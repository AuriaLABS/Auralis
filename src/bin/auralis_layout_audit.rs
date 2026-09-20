use auralis::bench_result::{BenchComparison, BenchRunRecord};
use auralis::kernels::matmul_row_slices_into;
use std::env;
use std::process::Command;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    rows: usize,
    inner: usize,
    cols: usize,
}

const CASES: &[Case] = &[
    Case { name: "qkv", rows: 32, inner: 32, cols: 32 },
    Case { name: "ffn_up", rows: 32, inner: 32, cols: 96 },
    Case { name: "ffn_down", rows: 32, inner: 96, cols: 32 },
    Case { name: "logits", rows: 32, inner: 32, cols: 100 },
    Case { name: "short_qkv", rows: 9, inner: 32, cols: 32 },
];

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 37 + salt * 19 + i / 3) % 101) as f32;
            (raw - 50.0) / 31.0
        })
        .collect()
}

fn pack_b_t(b: &[f32], inner: usize, cols: usize, bt: &mut [f32]) {
    assert_eq!(b.len(), inner * cols);
    assert_eq!(bt.len(), cols * inner);
    for k in 0..inner {
        let row = &b[k * cols..(k + 1) * cols];
        for j in 0..cols {
            bt[j * inner + k] = row[j];
        }
    }
}

fn matmul_packed_bt_into(
    a: &[f32],
    rows: usize,
    inner: usize,
    bt: &[f32],
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(bt.len(), cols * inner);
    assert_eq!(out.len(), rows * cols);

    for i in 0..rows {
        let a_row = &a[i * inner..(i + 1) * inner];
        let out_row = &mut out[i * cols..(i + 1) * cols];
        for j in 0..cols {
            let bt_row = &bt[j * inner..(j + 1) * inner];
            let mut sum = 0.0f32;
            for k in 0..inner {
                sum += a_row[k] * bt_row[k];
            }
            out_row[j] = sum;
        }
    }
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().max(1).min(u64::MAX as u128) as u64
}

fn run_rowslices(
    case: Case,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    iterations: usize,
) -> u64 {
    let started = Instant::now();
    for _ in 0..iterations {
        matmul_row_slices_into(a, case.rows, case.inner, b, case.cols, out);
    }
    elapsed_ns(started)
}

fn run_packed_kernel(
    case: Case,
    a: &[f32],
    bt: &[f32],
    out: &mut [f32],
    iterations: usize,
) -> u64 {
    let started = Instant::now();
    for _ in 0..iterations {
        matmul_packed_bt_into(a, case.rows, case.inner, bt, case.cols, out);
    }
    elapsed_ns(started)
}

fn run_packed_e2e(
    case: Case,
    a: &[f32],
    b: &[f32],
    bt: &mut [f32],
    out: &mut [f32],
    iterations: usize,
) -> u64 {
    let started = Instant::now();
    for _ in 0..iterations {
        pack_b_t(b, case.inner, case.cols, bt);
        matmul_packed_bt_into(a, case.rows, case.inner, bt, case.cols, out);
    }
    elapsed_ns(started)
}

fn hardware() -> String {
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some((key, value)) = line.split_once(':') {
                if key.trim() == "model name" && !value.trim().is_empty() {
                    return value.trim().to_string();
                }
            }
        }
    }
    format!("{}-{}", env::consts::ARCH, env::consts::OS)
}

fn toolchain() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "rustc-unavailable".into())
}

fn fingerprint(case: Case, warmup: usize, iterations: usize, repeats: usize) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in [
        case.rows as u64,
        case.inner as u64,
        case.cols as u64,
        warmup as u64,
        iterations as u64,
        repeats as u64,
    ] {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn record(
    case: Case,
    backend: &str,
    hw: &str,
    tc: &str,
    warmup: usize,
    repeats: usize,
    raw_ns: Vec<u64>,
    fp: u64,
) -> BenchRunRecord {
    BenchRunRecord {
        benchmark_id: format!("layout/{}", case.name),
        backend: backend.into(),
        hardware: hw.into(),
        toolchain: tc.into(),
        config_fingerprint: fp,
        warmup_iterations: warmup,
        repetitions: repeats,
        raw_ns,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let warmup = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    let iterations = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100usize);
    let repeats = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7usize);
    assert!(iterations > 0 && repeats >= 3);

    let hw = hardware();
    let tc = toolchain();

    for (case_index, &case) in CASES.iter().enumerate() {
        let a = data(case.rows * case.inner, case_index + 3);
        let b = data(case.inner * case.cols, case_index + 17);
        let mut bt = vec![0.0f32; case.cols * case.inner];
        let mut reference = vec![0.0f32; case.rows * case.cols];
        let mut packed = vec![0.0f32; case.rows * case.cols];

        pack_b_t(&b, case.inner, case.cols, &mut bt);
        matmul_row_slices_into(
            &a,
            case.rows,
            case.inner,
            &b,
            case.cols,
            &mut reference,
        );
        matmul_packed_bt_into(
            &a,
            case.rows,
            case.inner,
            &bt,
            case.cols,
            &mut packed,
        );
        assert_eq!(packed, reference, "packed candidate mismatch for {}", case.name);

        if warmup > 0 {
            let _ = run_rowslices(case, &a, &b, &mut reference, warmup);
            let _ = run_packed_kernel(case, &a, &bt, &mut packed, warmup);
            let _ = run_packed_e2e(case, &a, &b, &mut bt, &mut packed, warmup);
        }

        let mut row_raw = Vec::with_capacity(repeats);
        let mut packed_raw = Vec::with_capacity(repeats);
        let mut e2e_raw = Vec::with_capacity(repeats);

        for rep in 0..repeats {
            let order = rep % 3;
            let mut row = 0;
            let mut pk = 0;
            let mut e2e = 0;
            for mode in [order, (order + 1) % 3, (order + 2) % 3] {
                match mode {
                    0 => row = run_rowslices(case, &a, &b, &mut reference, iterations),
                    1 => pk = run_packed_kernel(case, &a, &bt, &mut packed, iterations),
                    _ => e2e = run_packed_e2e(case, &a, &b, &mut bt, &mut packed, iterations),
                }
            }
            assert_eq!(packed, reference, "output drift for {}", case.name);
            row_raw.push(row);
            packed_raw.push(pk);
            e2e_raw.push(e2e);
            println!(
                "layout_audit_run | case={} repetition={} rowslices_ns={} packed_kernel_ns={} packed_e2e_ns={} packed_kernel_over_rowslices={:.4} packed_e2e_over_rowslices={:.4} exact=true",
                case.name,
                rep + 1,
                row,
                pk,
                e2e,
                pk as f64 / row as f64,
                e2e as f64 / row as f64,
            );
        }

        let fp = fingerprint(case, warmup, iterations, repeats);
        let rowslices = record(
            case,
            "row_slices",
            &hw,
            &tc,
            warmup,
            repeats,
            row_raw,
            fp,
        );
        let packed_kernel = record(
            case,
            "packed_bt_persistent",
            &hw,
            &tc,
            warmup,
            repeats,
            packed_raw,
            fp,
        );
        let packed_e2e = record(
            case,
            "packed_bt_dynamic",
            &hw,
            &tc,
            warmup,
            repeats,
            e2e_raw,
            fp,
        );

        let kernel_cmp = BenchComparison::compare(&rowslices, &packed_kernel).unwrap();
        let e2e_cmp = BenchComparison::compare(&rowslices, &packed_e2e).unwrap();
        let macs = case.rows * case.inner * case.cols;
        let row_summary = rowslices.summary().unwrap();
        let ns_per_mac = row_summary.median_ns / (iterations * macs) as f64;

        println!("layout_audit_record | {}", rowslices.json().unwrap());
        println!("layout_audit_record | {}", packed_kernel.json().unwrap());
        println!("layout_audit_record | {}", packed_e2e.json().unwrap());
        println!(
            "layout_audit_summary | case={} shape={}x{}x{} iterations={} repeats={} row_median_ns={:.3} packed_kernel_over_rowslices={:.4} packed_e2e_over_rowslices={:.4} rowslices_ns_per_mac={:.6} exact=true promotion_candidate={}",
            case.name,
            case.rows,
            case.inner,
            case.cols,
            iterations,
            repeats,
            row_summary.median_ns,
            kernel_cmp.candidate_over_reference,
            e2e_cmp.candidate_over_reference,
            ns_per_mac,
            kernel_cmp.candidate_over_reference <= 0.95,
        );
    }
}
