use auralis::cpu_matmul::matmul_blocked_cols_into;
use auralis::kernels::matmul_reference_into;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct Shape {
    rows: usize,
    inner: usize,
    cols: usize,
}

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 43 + salt * 29 + i / 11) % 131) as f32;
            (raw - 65.0) / 41.0
        })
        .collect()
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000_000.0
}

fn run_once(shape: Shape, tile: usize, iters: usize, candidate_first: bool) -> (Duration, Duration) {
    let a = data(shape.rows * shape.inner, 7);
    let b = data(shape.inner * shape.cols, 19);
    let mut reference = vec![0.0f32; shape.rows * shape.cols];
    let mut candidate = vec![0.0f32; shape.rows * shape.cols];

    matmul_reference_into(
        &a,
        shape.rows,
        shape.inner,
        &b,
        shape.cols,
        &mut reference,
    );
    matmul_blocked_cols_into(
        &a,
        shape.rows,
        shape.inner,
        &b,
        shape.cols,
        &mut candidate,
        tile,
    );
    assert_eq!(candidate, reference, "candidate must match reference exactly");

    let mut run_reference = || {
        let start = Instant::now();
        for _ in 0..iters {
            matmul_reference_into(
                black_box(&a),
                shape.rows,
                shape.inner,
                black_box(&b),
                shape.cols,
                black_box(&mut reference),
            );
        }
        let elapsed = start.elapsed();
        black_box(reference[reference.len() / 2]);
        elapsed
    };

    let mut run_candidate = || {
        let start = Instant::now();
        for _ in 0..iters {
            matmul_blocked_cols_into(
                black_box(&a),
                shape.rows,
                shape.inner,
                black_box(&b),
                shape.cols,
                black_box(&mut candidate),
                tile,
            );
        }
        let elapsed = start.elapsed();
        black_box(candidate[candidate.len() / 2]);
        elapsed
    };

    if candidate_first {
        let candidate_time = run_candidate();
        let reference_time = run_reference();
        (reference_time, candidate_time)
    } else {
        let reference_time = run_reference();
        let candidate_time = run_candidate();
        (reference_time, candidate_time)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warmup = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    let iters = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50usize);
    let repeats = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7usize);
    let tile = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(32usize);

    assert!(repeats > 0);
    assert!(iters > 0);
    assert!(tile > 0);

    let shapes = [
        Shape { rows: 32, inner: 32, cols: 32 },
        Shape { rows: 32, inner: 32, cols: 96 },
        Shape { rows: 32, inner: 96, cols: 32 },
        Shape { rows: 32, inner: 32, cols: 100 },
    ];

    println!(
        "auralis_matmul_blocked_bench warmup={} iters={} repeats={} tile={}",
        warmup, iters, repeats, tile
    );

    for shape in shapes {
        for i in 0..warmup {
            let _ = run_once(shape, tile, 1, i % 2 == 0);
        }

        let mut reference_samples = Vec::with_capacity(repeats);
        let mut candidate_samples = Vec::with_capacity(repeats);
        for rep in 0..repeats {
            let (r, c) = run_once(shape, tile, iters, rep % 2 == 0);
            reference_samples.push(r);
            candidate_samples.push(c);
        }

        let reference_median = median(reference_samples);
        let candidate_median = median(candidate_samples);
        let ratio = candidate_median.as_secs_f64() / reference_median.as_secs_f64();
        println!(
            "shape={}x{}x{} reference_us={:.3} candidate_us={:.3} candidate_over_reference={:.4}",
            shape.rows,
            shape.inner,
            shape.cols,
            micros(reference_median),
            micros(candidate_median),
            ratio,
        );
    }
}
