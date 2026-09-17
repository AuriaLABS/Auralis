use auralis::simd::{capabilities, matmul_simd_into, SimdBackend};
use std::hint::black_box;
use std::time::Instant;

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 53 + salt * 31 + i / 5) % 149) as f32;
            (raw - 74.0) / 47.0
        })
        .collect()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn run(
    backend: SimdBackend,
    rows: usize,
    inner: usize,
    cols: usize,
    warmup: usize,
    iters: usize,
    repeats: usize,
) -> f64 {
    let a = data(rows * inner, 3);
    let b = data(inner * cols, 11);
    let mut out = vec![0.0; rows * cols];
    for _ in 0..warmup {
        matmul_simd_into(backend, &a, rows, inner, &b, cols, &mut out).unwrap();
        black_box(&out);
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        for _ in 0..iters {
            matmul_simd_into(backend, &a, rows, inner, &b, cols, &mut out).unwrap();
            black_box(&out);
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    median(samples)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warmup = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    let iters = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100usize);
    let repeats = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7usize);
    let caps = capabilities();

    println!(
        "simd_bench | arch={} avx2={} warmup={} iters={} repeats={}",
        std::env::consts::ARCH,
        caps.avx2,
        warmup,
        iters,
        repeats
    );

    for (rows, inner, cols) in [
        (32usize, 32usize, 32usize),
        (32, 32, 96),
        (32, 96, 32),
        (32, 32, 100),
        (128, 128, 128),
    ] {
        let portable = run(
            SimdBackend::Portable,
            rows,
            inner,
            cols,
            warmup,
            iters,
            repeats,
        );
        let auto = run(
            SimdBackend::Auto,
            rows,
            inner,
            cols,
            warmup,
            iters,
            repeats,
        );
        println!(
            "simd_result | shape={}x{}x{} portable_us={:.3} auto_us={:.3} auto_over_portable={:.4} speedup={:.4}",
            rows,
            inner,
            cols,
            portable,
            auto,
            auto / portable,
            portable / auto
        );
    }
}
