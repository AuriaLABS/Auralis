use auralis::kernels::matmul_reference_into;
use auralis::matmul_dispatch::{
    matmul_dispatch_into, MatmulBackend, ResolvedMatmulBackend,
};
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
) -> (f64, ResolvedMatmulBackend) {
    let a = data(rows * inner, 3);
    let b = data(inner * cols, 11);
    let mut out = vec![0.0; rows * cols];
    let mut resolved = matmul_dispatch_into(backend, &a, rows, inner, &b, cols, &mut out);

    for _ in 0..warmup {
        resolved = matmul_dispatch_into(backend, &a, rows, inner, &b, cols, &mut out);
        black_box(&out);
    }

    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        for _ in 0..iters {
            resolved = matmul_dispatch_into(backend, &a, rows, inner, &b, cols, &mut out);
            black_box(&out);
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }

    (median_us(samples), resolved)
}

fn run_mix(backend: MatmulBackend) {
    // Approximate the dominant dense matmul mix of Config::tiny(vocab=100),
    // n_layer=2 and context length 32: Q/K/V/O per layer, FF up/down per
    // layer, then final projection to vocab.
    for layer in 0..2usize {
        for op in 0..4usize {
            run_shape_once(32, 32, 32, backend, 100 + layer * 10 + op);
        }
        run_shape_once(32, 32, 96, backend, 200 + layer * 10);
        run_shape_once(32, 96, 32, backend, 201 + layer * 10);
    }
    run_shape_once(32, 32, 100, backend, 300);
}

fn run_shape_once(rows: usize, inner: usize, cols: usize, backend: MatmulBackend, salt: usize) {
    let a = data(rows * inner, salt);
    let b = data(inner * cols, salt + 7);
    let mut out = vec![0.0; rows * cols];
    black_box(matmul_dispatch_into(
        backend,
        black_box(&a),
        rows,
        inner,
        black_box(&b),
        cols,
        &mut out,
    ));
    black_box(out);
}

fn bench_mix(
    backend: MatmulBackend,
    warmup: usize,
    iters: usize,
    repeats: usize,
) -> f64 {
    for _ in 0..warmup {
        run_mix(backend);
    }

    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        for _ in 0..iters {
            run_mix(backend);
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    median_us(samples)
}

fn assert_mix_exact() {
    for (rows, inner, cols, salt) in [
        (32usize, 32usize, 32usize, 101usize),
        (32, 32, 96, 201),
        (32, 96, 32, 202),
        (32, 32, 100, 301),
    ] {
        let a = data(rows * inner, salt);
        let b = data(inner * cols, salt + 7);
        let mut reference = vec![0.0; rows * cols];
        let mut auto = vec![0.0; rows * cols];
        matmul_dispatch_into(
            MatmulBackend::Reference,
            &a,
            rows,
            inner,
            &b,
            cols,
            &mut reference,
        );
        matmul_dispatch_into(
            MatmulBackend::Auto,
            &a,
            rows,
            inner,
            &b,
            cols,
            &mut auto,
        );
        assert_eq!(auto, reference, "mix shape={rows}x{inner}x{cols}");
    }
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
            MatmulBackend::Auto,
            &a,
            rows,
            inner,
            &b,
            cols,
            &mut auto_out,
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

    assert_mix_exact();
    let mix_iters = iters.min(20).max(1);
    let reference_mix_us = bench_mix(MatmulBackend::Reference, warmup, mix_iters, repeats);
    let auto_mix_us = bench_mix(MatmulBackend::Auto, warmup, mix_iters, repeats);
    println!(
        "dispatch_mix_result | profile=tiny_v100_l2_ctx32 reference_us={:.3} auto_us={:.3} auto_over_reference={:.4} speedup={:.4}",
        reference_mix_us,
        auto_mix_us,
        auto_mix_us / reference_mix_us,
        reference_mix_us / auto_mix_us
    );
}
