use auralis::kernels::matmul_reference_into;
use auralis::matmul_dispatch::{matmul_dispatch_into, MatmulBackend};
use std::hint::black_box;
use std::time::Instant;

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 41 + salt * 23 + i / 7) % 127) as f32;
            (raw - 63.0) / 37.0
        })
        .collect()
}

fn median_us(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn bench_one(
    rows: usize,
    inner: usize,
    cols: usize,
    backend: MatmulBackend,
    warmup: usize,
    iters: usize,
    repeats: usize,
) -> (f64, MatmulBackend) {
    let a = data(rows * inner, 3);
    let b = data(inner * cols, 11);
    let mut out = vec![0.0; rows * cols];
    let mut resolved = backend;

    for _ in 0..warmup {
        resolved = matmul_dispatch_into(&a, rows, inner, &b, cols, &mut out, backend);
        black_box(&out);
    }

    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        for _ in 0..iters {
            resolved = matmul_dispatch_into(&a, rows, inner, &b, cols, &mut out, backend);
            black_box(&out);
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }

    (median_us(samples), resolved)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warmup = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let iters = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(80);
    let repeats = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7);

    println!(
        "auralis_matmul_dispatch_bench warmup={} iters={} repeats={}",
        warmup, iters, repeats
    );

    for (rows, inner, cols) in [
        (32usize, 32usize, 32usize),
        (32, 32, 96),
        (32, 96, 32),
        (32, 32, 100),
    ] {
        let a = data(rows * inner, 3);
        let b = data(inner * cols, 11);
        let mut reference_out = vec![0.0; rows * cols];
        let mut auto_out = vec![0.0; rows * cols];
        matmul_reference_into(&a, rows, inner, &b, cols, &mut reference_out);
        let resolved = matmul_dispatch_into(
            &a,
            rows,
            inner,
            &b,
            cols,
            &mut auto_out,
            MatmulBackend::Auto,
        );
        assert_eq!(auto_out, reference_out, "dispatch output must stay exact");

        let (reference_us, _) = bench_one(
            rows,
            inner,
            cols,
            MatmulBackend::Reference,
            warmup,
            iters,
            repeats,
        );
        let (auto_us, auto_resolved) = bench_one(
            rows,
            inner,
            cols,
            MatmulBackend::Auto,
            warmup,
            iters,
            repeats,
        );
        assert_eq!(resolved, auto_resolved);

        println!(
            "dispatch_result | shape={}x{}x{} resolved={:?} reference_us={:.3} auto_us={:.3} auto_over_reference={:.4}",
            rows,
            inner,
            cols,
            resolved,
            reference_us,
            auto_us,
            auto_us / reference_us
        );
    }
}
