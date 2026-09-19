
use auralis::bench_result::{BenchComparison, BenchRunRecord};
use auralis::cpu_matmul::matmul_blocked_32_into;
use auralis::kernels::{matmul_reference_into, matmul_row_slices_into};
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    rows: usize,
    inner: usize,
    cols: usize,
    eligible_for_auto: bool,
}

#[derive(Clone, Copy)]
enum Backend {
    RowSlices,
    Blocked32,
}

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 41 + salt * 23 + i / 7) % 127) as f32;
            (raw - 63.0) / 37.0
        })
        .collect()
}

fn shapes() -> [Shape; 5] {
    [
        Shape { rows: 32, inner: 32, cols: 32, eligible_for_auto: true },
        Shape { rows: 32, inner: 32, cols: 96, eligible_for_auto: true },
        Shape { rows: 32, inner: 96, cols: 32, eligible_for_auto: true },
        Shape { rows: 32, inner: 32, cols: 100, eligible_for_auto: true },
        Shape { rows: 9, inner: 13, cols: 37, eligible_for_auto: false },
    ]
}

fn hardware() -> String {
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some((key, value)) = line.split_once(':') {
                if key.trim() == "model name" {
                    let value = value.trim();
                    if !value.is_empty() {
                        return value.to_string();
                    }
                }
            }
        }
    }
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

fn toolchain() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "rustc-unknown".to_string())
}

fn fingerprint(shape: Shape, iters: usize) -> u64 {
    let text = format!(
        "rows={} inner={} cols={} iters={} tile=32",
        shape.rows, shape.inner, shape.cols, iters
    );
    let mut h = 0xcbf29ce484222325u64;
    for byte in text.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn exact_outputs(shape: Shape) -> bool {
    let a = data(shape.rows * shape.inner, 3);
    let b = data(shape.inner * shape.cols, 11);
    let mut scalar = vec![f32::NAN; shape.rows * shape.cols];
    let mut rows = vec![f32::NAN; shape.rows * shape.cols];
    let mut blocked = vec![f32::NAN; shape.rows * shape.cols];

    matmul_reference_into(&a, shape.rows, shape.inner, &b, shape.cols, &mut scalar);
    matmul_row_slices_into(&a, shape.rows, shape.inner, &b, shape.cols, &mut rows);
    matmul_blocked_32_into(&a, shape.rows, shape.inner, &b, shape.cols, &mut blocked);

    rows == scalar && blocked == scalar
}

fn run_once(
    backend: Backend,
    shape: Shape,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    iters: usize,
) -> u64 {
    let started = Instant::now();
    for _ in 0..iters {
        match backend {
            Backend::RowSlices => matmul_row_slices_into(
                black_box(a),
                shape.rows,
                shape.inner,
                black_box(b),
                shape.cols,
                black_box(&mut *out),
            ),
            Backend::Blocked32 => matmul_blocked_32_into(
                black_box(a),
                shape.rows,
                shape.inner,
                black_box(b),
                shape.cols,
                black_box(&mut *out),
            ),
        }
    }
    black_box(out[out.len() / 2]);
    let total = started.elapsed().as_nanos();
    ((total / iters as u128).max(1)) as u64
}

fn benchmark_pair(
    shape: Shape,
    warmup: usize,
    iters: usize,
    repeats: usize,
    hardware: &str,
    toolchain: &str,
) -> (BenchRunRecord, BenchRunRecord, usize) {
    let a = data(shape.rows * shape.inner, 17);
    let b = data(shape.inner * shape.cols, 29);
    let mut scalar_out = vec![f32::NAN; shape.rows * shape.cols];
    let mut row_out = vec![f32::NAN; shape.rows * shape.cols];
    let mut blocked_out = vec![f32::NAN; shape.rows * shape.cols];

    matmul_reference_into(
        &a,
        shape.rows,
        shape.inner,
        &b,
        shape.cols,
        &mut scalar_out,
    );
    matmul_row_slices_into(
        &a,
        shape.rows,
        shape.inner,
        &b,
        shape.cols,
        &mut row_out,
    );
    matmul_blocked_32_into(
        &a,
        shape.rows,
        shape.inner,
        &b,
        shape.cols,
        &mut blocked_out,
    );
    assert_eq!(
        row_out, scalar_out,
        "RowSlices benchmark data must match scalar for {}x{}x{}",
        shape.rows, shape.inner, shape.cols
    );
    assert_eq!(
        blocked_out, scalar_out,
        "Blocked32 benchmark data must match scalar for {}x{}x{}",
        shape.rows, shape.inner, shape.cols
    );

    for i in 0..warmup {
        if i % 2 == 0 {
            run_once(Backend::RowSlices, shape, &a, &b, &mut row_out, 1);
            run_once(Backend::Blocked32, shape, &a, &b, &mut blocked_out, 1);
        } else {
            run_once(Backend::Blocked32, shape, &a, &b, &mut blocked_out, 1);
            run_once(Backend::RowSlices, shape, &a, &b, &mut row_out, 1);
        }
    }

    let mut row_raw = Vec::with_capacity(repeats);
    let mut blocked_raw = Vec::with_capacity(repeats);
    let mut blocked_wins = 0usize;

    for rep in 0..repeats {
        let (row_ns, blocked_ns) = if rep % 2 == 0 {
            let row = run_once(Backend::RowSlices, shape, &a, &b, &mut row_out, iters);
            let blocked = run_once(Backend::Blocked32, shape, &a, &b, &mut blocked_out, iters);
            (row, blocked)
        } else {
            let blocked = run_once(Backend::Blocked32, shape, &a, &b, &mut blocked_out, iters);
            let row = run_once(Backend::RowSlices, shape, &a, &b, &mut row_out, iters);
            (row, blocked)
        };
        if blocked_ns < row_ns {
            blocked_wins += 1;
        }
        row_raw.push(row_ns);
        blocked_raw.push(blocked_ns);
    }

    let benchmark_id = format!("matmul/{}x{}x{}", shape.rows, shape.inner, shape.cols);
    let config_fingerprint = fingerprint(shape, iters);

    let row = BenchRunRecord {
        benchmark_id: benchmark_id.clone(),
        backend: "row_slices".into(),
        hardware: hardware.into(),
        toolchain: toolchain.into(),
        config_fingerprint,
        warmup_iterations: warmup,
        repetitions: repeats,
        raw_ns: row_raw,
    };
    let blocked = BenchRunRecord {
        benchmark_id,
        backend: "blocked32".into(),
        hardware: hardware.into(),
        toolchain: toolchain.into(),
        config_fingerprint,
        warmup_iterations: warmup,
        repetitions: repeats,
        raw_ns: blocked_raw,
    };
    (row, blocked, blocked_wins)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warmup = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    let iters = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(80usize);
    let repeats = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7usize);
    assert!(iters > 0, "iters must be positive");
    assert!(repeats >= 3, "repeats must be at least 3");

    let hardware = hardware();
    let toolchain = toolchain();
    println!(
        "matmul_current_baseline_protocol | warmup={} iters={} repeats={} hardware={:?} toolchain={:?} promotion_ratio_max=0.95",
        warmup, iters, repeats, hardware, toolchain
    );

    for shape in shapes() {
        assert!(
            exact_outputs(shape),
            "scalar/rowslices/blocked mismatch for {}x{}x{}",
            shape.rows,
            shape.inner,
            shape.cols
        );

        let (row, blocked, blocked_wins) =
            benchmark_pair(shape, warmup, iters, repeats, &hardware, &toolchain);
        let row_summary = row.summary().expect("row_slices summary");
        let blocked_summary = blocked.summary().expect("blocked32 summary");
        let comparison = BenchComparison::compare(&row, &blocked).expect("compatible records");

        let repeated_win = blocked_wins * 3 >= repeats * 2;
        let iqr_limit = row_summary.iqr_ns * 2.0 + row_summary.median_ns * 0.02;
        let iqr_ok = blocked_summary.iqr_ns <= iqr_limit;
        let promote = shape.eligible_for_auto
            && comparison.candidate_over_reference <= 0.95
            && repeated_win
            && iqr_ok;

        println!("bench_record | {}", row.json().expect("row json"));
        println!("bench_record | {}", blocked.json().expect("blocked json"));
        println!("bench_comparison | {}", comparison.json());
        println!(
            concat!(
                "matmul_current_baseline | shape={}x{}x{} exact=true eligible={} ",
                "row_slices_median_ns={:.3} row_slices_iqr_ns={:.3} ",
                "blocked32_median_ns={:.3} blocked32_iqr_ns={:.3} ",
                "blocked_over_rowslices={:.4} speedup={:.4} ",
                "blocked_wins={}/{} repeated_win={} iqr_limit_ns={:.3} iqr_ok={} promote={}"
            ),
            shape.rows,
            shape.inner,
            shape.cols,
            shape.eligible_for_auto,
            row_summary.median_ns,
            row_summary.iqr_ns,
            blocked_summary.median_ns,
            blocked_summary.iqr_ns,
            comparison.candidate_over_reference,
            comparison.speedup,
            blocked_wins,
            repeats,
            repeated_win,
            iqr_limit,
            iqr_ok,
            promote,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_corpus_shapes_match_scalar_reference_exactly() {
        for shape in shapes() {
            assert!(exact_outputs(shape), "shape={}x{}x{}", shape.rows, shape.inner, shape.cols);
        }
    }

    #[test]
    fn irregular_shape_is_never_auto_eligible() {
        let irregular = shapes()
            .into_iter()
            .find(|s| (s.rows, s.inner, s.cols) == (9, 13, 37))
            .unwrap();
        assert!(!irregular.eligible_for_auto);
    }

    #[test]
    fn fingerprint_tracks_shape_and_protocol() {
        let a = Shape { rows: 32, inner: 32, cols: 32, eligible_for_auto: true };
        let b = Shape { rows: 32, inner: 32, cols: 96, eligible_for_auto: true };
        assert_ne!(fingerprint(a, 80), fingerprint(b, 80));
        assert_ne!(fingerprint(a, 80), fingerprint(a, 81));
    }
}
