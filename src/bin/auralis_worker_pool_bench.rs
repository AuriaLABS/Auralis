use auralis::worker_pool::PersistentMatmulPool;
use std::hint::black_box;
use std::time::Instant;

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

fn run(threads: usize, rows: usize, inner: usize, cols: usize, warmup: usize, iters: usize, repeats: usize) -> f64 {
    let a = data(rows * inner, 3);
    let b = data(inner * cols, 11);
    let pool = PersistentMatmulPool::new(threads);

    for _ in 0..warmup {
        black_box(pool.matmul(black_box(&a), rows, inner, black_box(&b), cols));
    }

    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        for _ in 0..iters {
            black_box(pool.matmul(black_box(&a), rows, inner, black_box(&b), cols));
        }
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64);
    }
    median(samples)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warmup = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(5usize);
    let iters = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(40usize);
    let repeats = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(5usize);

    // Deliberately larger than the tiny training baseline so scheduling overhead
    // is visible but not the only cost. More shapes can be added after this
    // prototype establishes whether persistence is worthwhile.
    let (rows, inner, cols) = (128usize, 128usize, 128usize);
    let baseline = run(1, rows, inner, cols, warmup, iters, repeats);

    println!(
        "worker_pool_bench | shape={}x{}x{} warmup={} iters={} repeats={} baseline_us={:.3}",
        rows, inner, cols, warmup, iters, repeats, baseline
    );
    for threads in [1usize, 2, 4, 8] {
        let us = if threads == 1 {
            baseline
        } else {
            run(threads, rows, inner, cols, warmup, iters, repeats)
        };
        let speedup = baseline / us;
        let efficiency = speedup / threads as f64;
        println!(
            "worker_pool_result | threads={} median_us={:.3} speedup={:.4} efficiency={:.4}",
            threads, us, speedup, efficiency
        );
    }
}
