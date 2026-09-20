use auralis::backend::{
    Backend, BackendId, DeviceId, MatrixMut, MatrixRef, OptimizedCpuBackend,
};
use auralis::bench_result::{BenchComparison, BenchRunRecord};
use auralis::kernels::matmul_row_slices_into;
use std::env;
use std::process::Command;
use std::time::Instant;

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 37 + salt * 19 + i / 3) % 101) as f32;
            (raw - 50.0) / 31.0
        })
        .collect()
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
    format!("{}-{}", env::consts::ARCH, env::consts::OS)
}

fn toolchain() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "rustc-unavailable".into())
}

fn fingerprint(iterations: usize) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for value in [32u64, 32, 32, iterations as u64, 1] {
        h ^= value;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn direct(a: &[f32], b: &[f32], out: &mut [f32], iterations: usize) -> u64 {
    let started = Instant::now();
    for _ in 0..iterations {
        matmul_row_slices_into(a, 32, 32, b, 32, out);
    }
    started.elapsed().as_nanos().max(1) as u64
}

fn boundary(
    backend: &dyn Backend,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    iterations: usize,
) -> u64 {
    let started = Instant::now();
    for _ in 0..iterations {
        backend
            .matmul(
                MatrixRef::new(a, 32, 32, DeviceId::Cpu).unwrap(),
                MatrixRef::new(b, 32, 32, DeviceId::Cpu).unwrap(),
                MatrixMut::new(out, 32, 32, DeviceId::Cpu).unwrap(),
            )
            .unwrap();
    }
    started.elapsed().as_nanos().max(1) as u64
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let warmup = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    let iterations = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200usize);
    let repeats = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7usize);
    assert!(iterations > 0 && repeats >= 3);

    let a = data(32 * 32, 3);
    let b = data(32 * 32, 17);
    let mut expected = vec![0.0f32; 32 * 32];
    let mut observed = vec![0.0f32; 32 * 32];
    matmul_row_slices_into(&a, 32, 32, &b, 32, &mut expected);

    let optimized = OptimizedCpuBackend;
    let backend: &dyn Backend = &optimized;
    backend
        .matmul(
            MatrixRef::new(&a, 32, 32, DeviceId::Cpu).unwrap(),
            MatrixRef::new(&b, 32, 32, DeviceId::Cpu).unwrap(),
            MatrixMut::new(&mut observed, 32, 32, DeviceId::Cpu).unwrap(),
        )
        .unwrap();
    assert_eq!(observed, expected);

    if warmup > 0 {
        let _ = direct(&a, &b, &mut expected, warmup);
        let _ = boundary(backend, &a, &b, &mut observed, warmup);
    }

    let mut direct_raw = Vec::with_capacity(repeats);
    let mut boundary_raw = Vec::with_capacity(repeats);
    for repetition in 0..repeats {
        let direct_first = repetition % 2 == 0;
        let (d, v) = if direct_first {
            (
                direct(&a, &b, &mut expected, iterations),
                boundary(backend, &a, &b, &mut observed, iterations),
            )
        } else {
            let v = boundary(backend, &a, &b, &mut observed, iterations);
            let d = direct(&a, &b, &mut expected, iterations);
            (d, v)
        };
        assert_eq!(observed, expected);
        direct_raw.push(d);
        boundary_raw.push(v);
        println!(
            "backend_boundary_run | repetition={} first={} direct_ns={} boundary_ns={} ratio={:.4} exact=true",
            repetition + 1,
            if direct_first { "direct" } else { "boundary" },
            d,
            v,
            v as f64 / d as f64,
        );
    }

    let hw = hardware();
    let tc = toolchain();
    let fp = fingerprint(iterations);
    let direct_record = BenchRunRecord {
        benchmark_id: "backend/matmul/32x32x32".into(),
        backend: "cpu-row-slices-direct".into(),
        hardware: hw.clone(),
        toolchain: tc.clone(),
        config_fingerprint: fp,
        warmup_iterations: warmup,
        repetitions: repeats,
        raw_ns: direct_raw,
    };
    let boundary_record = BenchRunRecord {
        benchmark_id: "backend/matmul/32x32x32".into(),
        backend: BackendId::OptimizedCpu.as_str().into(),
        hardware: hw,
        toolchain: tc,
        config_fingerprint: fp,
        warmup_iterations: warmup,
        repetitions: repeats,
        raw_ns: boundary_raw,
    };
    let comparison = BenchComparison::compare(&direct_record, &boundary_record).unwrap();

    println!("backend_boundary_record | {}", direct_record.json().unwrap());
    println!("backend_boundary_record | {}", boundary_record.json().unwrap());
    println!("backend_boundary_comparison | {}", comparison.json());
    println!(
        "backend_boundary_summary | warmup={} iterations={} repeats={} candidate_over_direct={:.4} exact=true backend={} device={}",
        warmup,
        iterations,
        repeats,
        comparison.candidate_over_reference,
        optimized.id().as_str(),
        optimized.device().label(),
    );
}
