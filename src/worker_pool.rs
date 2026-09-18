//! Experimental persistent CPU worker pool for deterministic matmul.
//!
//! This module stays outside the production model path until scaling is proven.
//! The production constructor caps workers to avoid oversubscription. Multi-
//! threaded execution partitions rows deterministically and preserves ascending
//! `k` accumulation for every output cell, so bit-for-bit equality with the
//! scalar reference kernel is expected.

use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};

#[derive(Default)]
struct InputBuffers {
    a: Vec<f32>,
    b: Vec<f32>,
}

enum InputSource {
    Shared {
        a: Arc<[f32]>,
        b: Arc<[f32]>,
    },
    Buffered,
}

struct MatmulJob {
    row_start: usize,
    row_end: usize,
    inner: usize,
    cols: usize,
    input: InputSource,
}

enum Message {
    Matmul(MatmulJob),
    Shutdown,
}

struct Worker {
    tx: mpsc::SyncSender<Message>,
    scratch: Arc<Mutex<Vec<f32>>>,
    join: Option<JoinHandle<()>>,
}

/// Persistent row-partitioned pool used to validate deterministic CPU scheduling.
///
/// `threads() == 1` stays on the scalar reference kernel. For N>1, each
/// worker owns a persistent scratch buffer and a dedicated command channel.
/// The pool serializes calls so the buffered slice API can safely reuse one
/// pair of input buffers without allocating them again after warm-up.
pub struct PersistentMatmulPool {
    requested_threads: usize,
    effective_threads: usize,
    workers: Vec<Worker>,
    inputs: Arc<RwLock<InputBuffers>>,
    call_lock: Mutex<()>,
    completion_rx: Mutex<mpsc::Receiver<usize>>,
}

impl PersistentMatmulPool {
    /// Create a pool capped by `available_parallelism()` and an Engine-wide
    /// ceiling of 8 workers.
    pub fn new(requested_threads: usize) -> Self {
        assert!(requested_threads > 0, "worker count must be positive");
        let available = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self::build(requested_threads, available)
    }

    fn build(requested_threads: usize, available_threads: usize) -> Self {
        assert!(requested_threads > 0, "worker count must be positive");
        assert!(available_threads > 0, "available worker count must be positive");

        let effective_threads = requested_threads.min(available_threads).min(8).max(1);
        let inputs = Arc::new(RwLock::new(InputBuffers::default()));
        let (completion_tx, completion_rx) = mpsc::sync_channel::<usize>(effective_threads);
        let mut workers = Vec::with_capacity(effective_threads);

        for worker_id in 0..effective_threads {
            let (tx, rx) = mpsc::sync_channel::<Message>(1);
            let scratch = Arc::new(Mutex::new(Vec::<f32>::new()));
            let worker_scratch = Arc::clone(&scratch);
            let worker_inputs = Arc::clone(&inputs);
            let done = completion_tx.clone();

            let join = thread::spawn(move || {
                while let Ok(message) = rx.recv() {
                    match message {
                        Message::Matmul(job) => {
                            run_job(job, &worker_inputs, &worker_scratch);
                            if done.send(worker_id).is_err() {
                                break;
                            }
                        }
                        Message::Shutdown => break,
                    }
                }
            });

            workers.push(Worker {
                tx,
                scratch,
                join: Some(join),
            });
        }
        drop(completion_tx);

        Self {
            requested_threads,
            effective_threads,
            workers,
            inputs,
            call_lock: Mutex::new(()),
            completion_rx: Mutex::new(completion_rx),
        }
    }

    #[cfg(test)]
    fn new_for_test(requested_threads: usize, available_threads: usize) -> Self {
        Self::build(requested_threads, available_threads)
    }

    pub fn requested_threads(&self) -> usize {
        self.requested_threads
    }

    pub fn threads(&self) -> usize {
        self.effective_threads
    }

    /// Zero-copy input path for callers that already own immutable shared
    /// matrices. Arc cloning is O(1); matrix contents are not copied per call.
    pub fn matmul_shared_into(
        &self,
        a: Arc<[f32]>,
        rows: usize,
        inner: usize,
        b: Arc<[f32]>,
        cols: usize,
        out: &mut [f32],
    ) {
        assert_eq!(a.len(), rows * inner);
        assert_eq!(b.len(), inner * cols);
        assert_eq!(out.len(), rows * cols);

        if self.effective_threads == 1 || rows <= 1 {
            crate::kernels::matmul_reference_into(&a, rows, inner, &b, cols, out);
            return;
        }

        let _call = self.call_lock.lock().expect("worker pool call mutex poisoned");
        self.dispatch(
            rows,
            inner,
            cols,
            out,
            |_, _| InputSource::Shared {
                a: Arc::clone(&a),
                b: Arc::clone(&b),
            },
        );
    }

    /// Slice compatibility path. Inputs are copied into pool-owned reusable
    /// buffers, eliminating the `to_vec()` allocations from the first
    /// prototype while keeping a safe persistent-worker design.
    pub fn matmul_slices_into(
        &self,
        a: &[f32],
        rows: usize,
        inner: usize,
        b: &[f32],
        cols: usize,
        out: &mut [f32],
    ) {
        assert_eq!(a.len(), rows * inner);
        assert_eq!(b.len(), inner * cols);
        assert_eq!(out.len(), rows * cols);

        if self.effective_threads == 1 || rows <= 1 {
            crate::kernels::matmul_reference_into(a, rows, inner, b, cols, out);
            return;
        }

        let _call = self.call_lock.lock().expect("worker pool call mutex poisoned");
        {
            let mut buffers = self.inputs.write().expect("worker input lock poisoned");
            buffers.a.resize(a.len(), 0.0);
            buffers.b.resize(b.len(), 0.0);
            buffers.a.copy_from_slice(a);
            buffers.b.copy_from_slice(b);
        }

        self.dispatch(rows, inner, cols, out, |_, _| InputSource::Buffered);
    }

    fn dispatch(
        &self,
        rows: usize,
        inner: usize,
        cols: usize,
        out: &mut [f32],
        mut input_for_worker: impl FnMut(usize, usize) -> InputSource,
    ) {
        out.fill(0.0);
        let chunk_count = self.effective_threads.min(rows);

        for worker_id in 0..chunk_count {
            let row_start = worker_id * rows / chunk_count;
            let row_end = (worker_id + 1) * rows / chunk_count;
            debug_assert!(row_start < row_end);
            let job = MatmulJob {
                row_start,
                row_end,
                inner,
                cols,
                input: input_for_worker(row_start, row_end),
            };
            self.workers[worker_id]
                .tx
                .send(Message::Matmul(job))
                .expect("worker pool stopped unexpectedly");
        }

        let completion_rx = self
            .completion_rx
            .lock()
            .expect("worker completion mutex poisoned");
        for _ in 0..chunk_count {
            let worker_id = completion_rx
                .recv()
                .expect("worker failed before reporting completion");
            let row_start = worker_id * rows / chunk_count;
            let row_end = (worker_id + 1) * rows / chunk_count;
            let scratch = self.workers[worker_id]
                .scratch
                .lock()
                .expect("worker scratch mutex poisoned");
            out[row_start * cols..row_end * cols]
                .copy_from_slice(&scratch[..(row_end - row_start) * cols]);
        }
    }
}

impl Drop for PersistentMatmulPool {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.tx.send(Message::Shutdown);
        }
        for worker in &mut self.workers {
            if let Some(join) = worker.join.take() {
                let _ = join.join();
            }
        }
    }
}

fn run_job(
    job: MatmulJob,
    buffered_inputs: &Arc<RwLock<InputBuffers>>,
    scratch: &Arc<Mutex<Vec<f32>>>,
) {
    let MatmulJob {
        row_start,
        row_end,
        inner,
        cols,
        input,
    } = job;

    match input {
        InputSource::Shared { a, b } => {
            run_rows(&a, &b, row_start, row_end, inner, cols, scratch);
        }
        InputSource::Buffered => {
            let inputs = buffered_inputs.read().expect("worker input lock poisoned");
            run_rows(
                &inputs.a,
                &inputs.b,
                row_start,
                row_end,
                inner,
                cols,
                scratch,
            );
        }
    }
}

fn run_rows(
    a: &[f32],
    b: &[f32],
    row_start: usize,
    row_end: usize,
    inner: usize,
    cols: usize,
    scratch: &Arc<Mutex<Vec<f32>>>,
) {
    let rows = row_end - row_start;
    let needed = rows * cols;
    let mut out = scratch.lock().expect("worker scratch mutex poisoned");
    out.resize(needed, 0.0);
    out[..needed].fill(0.0);

    for local_row in 0..rows {
        let global_row = row_start + local_row;
        let a_row = &a[global_row * inner..(global_row + 1) * inner];
        let out_row = &mut out[local_row * cols..(local_row + 1) * cols];
        for k in 0..inner {
            let av = a_row[k];
            let b_row = &b[k * cols..(k + 1) * cols];
            for j in 0..cols {
                out_row[j] += av * b_row[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PersistentMatmulPool;
    use std::sync::Arc;

    fn data(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let raw = ((i * 37 + salt * 19 + i / 5) % 113) as f32;
                (raw - 56.0) / 31.0
            })
            .collect()
    }

    fn reference(a: &[f32], rows: usize, inner: usize, b: &[f32], cols: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * cols];
        crate::kernels::matmul_reference_into(a, rows, inner, b, cols, &mut out);
        out
    }

    #[test]
    fn one_thread_is_exact_reference() {
        let (rows, inner, cols) = (7, 13, 11);
        let a = data(rows * inner, 3);
        let b = data(inner * cols, 7);
        let pool = PersistentMatmulPool::new_for_test(1, 8);
        let mut out = vec![f32::NAN; rows * cols];
        pool.matmul_slices_into(&a, rows, inner, &b, cols, &mut out);
        assert_eq!(out, reference(&a, rows, inner, &b, cols));
    }

    #[test]
    fn thread_policy_caps_requested_available_and_eight() {
        let p = PersistentMatmulPool::new_for_test(16, 32);
        assert_eq!(p.requested_threads(), 16);
        assert_eq!(p.threads(), 8);

        let p = PersistentMatmulPool::new_for_test(8, 3);
        assert_eq!(p.threads(), 3);
    }

    #[test]
    fn shared_and_buffered_paths_are_exact_across_thread_counts() {
        for (rows, inner, cols) in [
            (2, 3, 5),
            (7, 13, 11),
            (32, 32, 32),
            (32, 32, 96),
            (32, 96, 32),
            (32, 32, 100),
            (33, 17, 19),
        ] {
            let a = data(rows * inner, 5);
            let b = data(inner * cols, 11);
            let expected = reference(&a, rows, inner, &b, cols);
            let a_shared: Arc<[f32]> = Arc::from(a.clone());
            let b_shared: Arc<[f32]> = Arc::from(b.clone());

            for threads in [2, 3, 4, 8] {
                let pool = PersistentMatmulPool::new_for_test(threads, 8);
                let mut buffered = vec![f32::NAN; rows * cols];
                let mut shared = vec![f32::NAN; rows * cols];

                pool.matmul_slices_into(&a, rows, inner, &b, cols, &mut buffered);
                pool.matmul_shared_into(
                    Arc::clone(&a_shared),
                    rows,
                    inner,
                    Arc::clone(&b_shared),
                    cols,
                    &mut shared,
                );

                assert_eq!(buffered, expected);
                assert_eq!(shared, expected);
            }
        }
    }

    #[test]
    fn uneven_rows_are_partitioned_without_empty_or_reversed_chunks() {
        let (rows, inner, cols) = (33, 17, 19);
        let a = data(rows * inner, 23);
        let b = data(inner * cols, 29);
        let expected = reference(&a, rows, inner, &b, cols);
        let pool = PersistentMatmulPool::new_for_test(8, 8);
        let mut out = vec![f32::NAN; rows * cols];

        pool.matmul_slices_into(&a, rows, inner, &b, cols, &mut out);

        assert_eq!(out, expected);
    }

    #[test]
    fn repeated_calls_are_bit_exact_and_reuse_shape_capacity() {
        let pool = PersistentMatmulPool::new_for_test(4, 8);
        for &(rows, inner, cols) in &[(31, 29, 23), (9, 7, 11), (31, 29, 23)] {
            let a = data(rows * inner, 13);
            let b = data(inner * cols, 17);
            let expected = reference(&a, rows, inner, &b, cols);
            let mut out = vec![0.0; rows * cols];

            for _ in 0..12 {
                pool.matmul_slices_into(&a, rows, inner, &b, cols, &mut out);
                assert_eq!(out, expected);
            }
        }
    }

    #[test]
    fn pool_shuts_down_cleanly_after_many_jobs() {
        let pool = PersistentMatmulPool::new_for_test(3, 8);
        for round in 1..=16 {
            let rows = 3 + round;
            let inner = 5 + round % 4;
            let cols = 7 + round % 5;
            let a = data(rows * inner, round);
            let b = data(inner * cols, round + 1);
            let mut out = vec![0.0; rows * cols];
            pool.matmul_slices_into(&a, rows, inner, &b, cols, &mut out);
            assert_eq!(out, reference(&a, rows, inner, &b, cols));
        }
        drop(pool);
    }

    #[test]
    #[should_panic(expected = "worker count must be positive")]
    fn zero_workers_are_rejected() {
        let _ = PersistentMatmulPool::new_for_test(0, 8);
    }
}
