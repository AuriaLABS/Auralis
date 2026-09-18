use auralis::kernels::matmul_reference_into;
use auralis::worker_pool::PersistentMatmulPool;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Shape {
    rows: usize,
    inner: usize,
    cols: usize,
}

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 43 + salt * 17 + i / 7) % 127) as f32;
            (raw - 63.0) / 37.0
        })
        .collect()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn time_reference(
    a: &[f32],
    b: &[f32],
    shape: Shape,
    warmup: usize,
    iters: usize,
    repeats: usize,
) -> f64 {
    let mut out = vec![0.0; shape.rows * shape.cols];
    for _ in 0..warmup {
        matmul_reference_into(a, shape.rows, shape.inner, b, shape.cols, &mut out);
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        for _ in 0..iters {
            matmul_reference_into(
                black_box(a),
                shape.rows,
                shape.inner,
                black_box(b),
                shape.cols,
                black_box(&mut out),
            );
        }
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64);
    }
    median(samples)
}

fn time_shared(
    pool: &PersistentMatmulPool,
    a: &Arc<[f32]>,
    b: &Arc<[f32]>,
    shape: Shape,
    warmup: usize,
    iters: usize,
    repeats: usize,
) -> f64 {
    let mut out = vec![0.0; shape.rows * shape.cols];
    for _ in 0..warmup {
        pool.matmul_shared_into(
            Arc::clone(a),
            shape.rows,
            shape.inner,
            Arc::clone(b),
            shape.cols,
            &mut out,
        );
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        for _ in 0..iters {
            pool.matmul_shared_into(
                black_box(Arc::clone(a)),
                shape.rows,
                shape.inner,
                black_box(Arc::clone(b)),
                shape.cols,
                black_box(&mut out),
            );
        }
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64);
    }
    median(samples)
}

fn time_buffered(
    pool: &PersistentMatmulPool,
    a: &[f32],
    b: &[f32],
    shape: Shape,
    warmup: usize,
    iters: usize,
    repeats: usize,
) -> f64 {
    let mut out = vec![0.0; shape.rows * shape.cols];
    for _ in 0..warmup {
        pool.matmul_slices_into(a, shape.rows, shape.inner, b, shape.cols, &mut out);
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        for _ in 0..iters {
            pool.matmul_slices_into(
                black_box(a),
                shape.rows,
                shape.inner,
                black_box(b),
                shape.cols,
                black_box(&mut out),
            );
        }
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64);
    }
    median(samples)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warmup = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(3usize);
    let iters = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20usize);
    let repeats = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(5usize);

    let shapes = [
        Shape { rows: 32, inner: 32, cols: 32 },
        Shape { rows: 32, inner: 32, cols: 96 },
        Shape { rows: 32, inner: 96, cols: 32 },
        Shape { rows: 32, inner: 32, cols: 100 },
        Shape { rows: 128, inner: 128, cols: 128 },
    ];

    println!(
        "worker_pool_v2_bench | warmup={} iters={} repeats={}",
        warmup, iters, repeats
    );

    for shape in shapes {
        let a = data(shape.rows * shape.inner, 3);
        let b = data(shape.inner * shape.cols, 11);
        let a_shared: Arc<[f32]> = Arc::from(a.clone());
        let b_shared: Arc<[f32]> = Arc::from(b.clone());
        let reference_us = time_reference(&a, &b, shape, warmup, iters, repeats);

        for requested in [1usize, 2, 4, 8] {
            let pool = PersistentMatmulPool::new(requested);
            let mut shared_out = vec![f32::NAN; shape.rows * shape.cols];
            let mut buffered_out = vec![f32::NAN; shape.rows * shape.cols];
            let mut expected = vec![0.0; shape.rows * shape.cols];
            matmul_reference_into(
                &a,
                shape.rows,
                shape.inner,
                &b,
                shape.cols,
                &mut expected,
            );
            pool.matmul_shared_into(
                Arc::clone(&a_shared),
                shape.rows,
                shape.inner,
                Arc::clone(&b_shared),
                shape.cols,
                &mut shared_out,
            );
            pool.matmul_slices_into(
                &a,
                shape.rows,
                shape.inner,
                &b,
                shape.cols,
                &mut buffered_out,
            );
            assert_eq!(shared_out, expected);
            assert_eq!(buffered_out, expected);

            let shared_us = time_shared(
                &pool,
                &a_shared,
                &b_shared,
                shape,
                warmup,
                iters,
                repeats,
            );
            let buffered_us =
                time_buffered(&pool, &a, &b, shape, warmup, iters, repeats);
            let effective = pool.threads();

            for (mode, candidate_us) in [("shared", shared_us), ("buffered", buffered_us)] {
                let speedup = reference_us / candidate_us;
                let efficiency = speedup / effective as f64;
                println!(
                    "worker_pool_v2_result | shape={}x{}x{} mode={} requested={} effective={} reference_us={:.3} candidate_us={:.3} ratio={:.4} speedup={:.4} efficiency={:.4}",
                    shape.rows,
                    shape.inner,
                    shape.cols,
                    mode,
                    requested,
                    effective,
                    reference_us,
                    candidate_us,
                    candidate_us / reference_us,
                    speedup,
                    efficiency,
                );
            }
        }
    }
}
