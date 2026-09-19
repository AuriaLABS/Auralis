use auralis::bench_result::{BenchComparison, BenchRunRecord};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("auralis-bench-result-smoke"));
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    let reference = BenchRunRecord {
        benchmark_id: "matmul,\"tiny\"\ncase".into(),
        backend: "reference".into(),
        hardware: "cpu\rnode\u{0001}".into(),
        toolchain: "rustc\t1.98.1".into(),
        config_fingerprint: 0x1234,
        warmup_iterations: 3,
        repetitions: 3,
        raw_ns: vec![30, 10, 20],
    };
    let candidate = BenchRunRecord {
        benchmark_id: reference.benchmark_id.clone(),
        backend: "candidate".into(),
        hardware: reference.hardware.clone(),
        toolchain: reference.toolchain.clone(),
        config_fingerprint: reference.config_fingerprint,
        warmup_iterations: reference.warmup_iterations,
        repetitions: 3,
        raw_ns: vec![15, 10, 12],
    };

    let comparison = BenchComparison::compare(&reference, &candidate)?;

    fs::write(out_dir.join("run.json"), reference.json()?)
        .map_err(|e| format!("write run.json: {e}"))?;
    fs::write(
        out_dir.join("run.csv"),
        format!("{}{}", BenchRunRecord::csv_header(), reference.csv_row()?),
    )
    .map_err(|e| format!("write run.csv: {e}"))?;
    fs::write(out_dir.join("comparison.json"), comparison.json())
        .map_err(|e| format!("write comparison.json: {e}"))?;

    println!("bench_result_smoke | dir={}", out_dir.display());
    Ok(())
}
