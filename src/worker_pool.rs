//! Experimental persistent CPU worker pool for deterministic matmul.
//!
//! This module is deliberately isolated from the production model path. It
//! validates the E3 scheduling contract before any hot-path integration:
//! workers persist across calls, row ranges are assigned deterministically,
//! and every output cell keeps the same ascending-`k` accumulation order as
//! the scalar/reference kernel.

use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

type Reply = mpsc::Sender<(usize, usize, Vec<f32>)>;

struct MatmulJob {
    row_start: usize,
    row_end: usize,
    inner: usize,
    cols: usize,
    a: Arc<Vec<f32>>,
    b: Arc<Vec<f32>>,
    reply: Reply,
}

enum Message {
    Matmul(MatmulJob),
    Shutdown,
}

/// Persistent pool used to validate deterministic CPU scheduling.
///
/// `threads == 1` intentionally stays on the reference kernel. Multi-threaded
/// execution partitions rows deterministically; workers never reduce into a
/// shared output buffer.
pub struct PersistentMatmulPool {
    threads: usize,
    tx: mpsc::Sender<Message>,
    workers: Vec<JoinHandle<()>>,
}

impl PersistentMatmulPool {
    pub fn new(threads: usize) -> Self {
        assert!(threads > 0, "worker count must be positive");
        let (tx, rx) = mpsc::channel::<Message>();
        let rx = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(threads);

        for _ in 0..threads {
            let rx = Arc::clone(&rx);
            workers.push(thread::spawn(move || loop {
                let message = {
                    let guard = rx.lock().expect("worker queue mutex poisoned");
                    guard.recv()
                };
                match message {
                    Ok(Message::Matmul(job)) => run_matmul_job(job),
                    Ok(Message::Shutdown) | Err(_) => break,
                }
            }));
        }

        Self {
            threads,
            tx,
            workers,
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Compute `A[rows, inner] * B[inner, cols]`.
    ///
    /// Multi-threaded mode uses fixed contiguous row chunks and deterministic
    /// assembly. Each output element accumulates `k=0..inner` in the same
    /// order as `matmul_reference_into`, so exact output is expected.
    pub fn matmul(&self, a: &[f32], rows: usize, inner: usize, b: &[f32], cols: usize) -> Vec<f32> {
        assert_eq!(a.len(), rows * inner);
        assert_eq!(b.len(), inner * cols);

        let mut out = vec![0.0; rows * cols];
        if self.threads == 1 || rows <= 1 {
            crate::kernels::matmul_reference_into(a, rows, inner, b, cols, &mut out);
            return out;
        }

        let chunk_count = self.threads.min(rows);
        let chunk_rows = rows.div_ceil(chunk_count);
        let a = Arc::new(a.to_vec());
        let b = Arc::new(b.to_vec());
        let (reply_tx, reply_rx) = mpsc::channel();
        let mut submitted = 0usize;

        for chunk in 0..chunk_count {
            let row_start = chunk * chunk_rows;
            let row_end = (row_start + chunk_rows).min(rows);
            if row_start >= row_end {
                continue;
            }
            let job = MatmulJob {
                row_start,
                row_end,
                inner,
                cols,
                a: Arc::clone(&a),
                b: Arc::clone(&b),
                reply: reply_tx.clone(),
            };
            self.tx.send(Message::Matmul(job)).expect("worker pool stopped unexpectedly");
            submitted += 1;
        }
        drop(reply_tx);

        // Arrival order is intentionally irrelevant. Chunks carry absolute row
        // indices and are copied into their predetermined output range.
        for _ in 0..submitted {
            let (row_start, row_end, chunk) = reply_rx.recv().expect("worker failed before replying");
            out[row_start * cols..row_end * cols].copy_from_slice(&chunk);
        }
        out
    }
}

impl Drop for PersistentMatmulPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            let _ = self.tx.send(Message::Shutdown);
        }
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

fn run_matmul_job(job: MatmulJob) {
    let rows = job.row_end - job.row_start;
    let mut chunk = vec![0.0; rows * job.cols];
    for local_row in 0..rows {
        let global_row = job.row_start + local_row;
        for k in 0..job.inner {
            let av = job.a[global_row * job.inner + k];
            let b_row = &job.b[k * job.cols..(k + 1) * job.cols];
            let out_row = &mut chunk[local_row * job.cols..(local_row + 1) * job.cols];
            for j in 0..job.cols {
                out_row[j] += av * b_row[j];
            }
        }
    }
    let _ = job.reply.send((job.row_start, job.row_end, chunk));
}

#[cfg(test)]
mod tests {
    use super::PersistentMatmulPool;

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
        let pool = PersistentMatmulPool::new(1);
        assert_eq!(pool.threads(), 1);
        assert_eq!(pool.matmul(&a, rows, inner, &b, cols), reference(&a, rows, inner, &b, cols));
    }

    #[test]
    fn fixed_partition_is_exact_across_thread_counts() {
        for (rows, inner, cols) in [(2, 3, 5), (7, 13, 11), (32, 32, 32), (33, 17, 19)] {
            let a = data(rows * inner, 5);
            let b = data(inner * cols, 11);
            let expected = reference(&a, rows, inner, &b, cols);
            for threads in [2, 3, 4, 8] {
                let pool = PersistentMatmulPool::new(threads);
                assert_eq!(pool.matmul(&a, rows, inner, &b, cols), expected);
            }
        }
    }

    #[test]
    fn repeated_calls_are_bit_exact() {
        let (rows, inner, cols) = (31, 29, 23);
        let a = data(rows * inner, 13);
        let b = data(inner * cols, 17);
        let pool = PersistentMatmulPool::new(4);
        let first = pool.matmul(&a, rows, inner, &b, cols);
        for _ in 0..20 {
            assert_eq!(pool.matmul(&a, rows, inner, &b, cols), first);
        }
    }

    #[test]
    fn pool_reuses_workers_across_jobs_and_shuts_down_cleanly() {
        let pool = PersistentMatmulPool::new(3);
        for round in 1..=12 {
            let rows = 3 + round;
            let inner = 5 + round % 4;
            let cols = 7 + round % 5;
            let a = data(rows * inner, round);
            let b = data(inner * cols, round + 1);
            assert_eq!(pool.matmul(&a, rows, inner, &b, cols), reference(&a, rows, inner, &b, cols));
        }
        drop(pool);
    }

    #[test]
    #[should_panic(expected = "worker count must be positive")]
    fn zero_workers_are_rejected() {
        let _ = PersistentMatmulPool::new(0);
    }
}
