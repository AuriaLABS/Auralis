//! Library-first benchmark execution for packaged Auralis binaries.
//!
//! This module deliberately does not spawn Cargo or depend on the repository
//! layout. Existing benchmark bins may wrap these runners for legacy output.

use crate::bench::{find, BenchRunnerKind};
use crate::bench_format::CatalogFormat;
use crate::bench_result::{BenchComparison, BenchRunRecord};
use crate::kernels::{matmul_reference_into, matmul_row_slices_into};
use crate::model::{Config, Gpt};
use crate::optim::Adam;
use crate::training::{train_step, train_step_reuse, TrainConfig, TrainWorkspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchProtocol {
    pub warmup: usize,
    pub iterations: usize,
    pub repetitions: usize,
}

impl BenchProtocol {
    pub fn validate(self) -> Result<Self, String> {
        if self.iterations == 0 {
            return Err("benchmark iterations must be positive".into());
        }
        if self.repetitions == 0 {
            return Err("benchmark repetitions must be positive".into());
        }
        Ok(self)
    }

    pub fn from_default_args(args: &str) -> Result<Self, String> {
        let values: Vec<usize> = args
            .split_whitespace()
            .map(|part| {
                part.parse::<usize>()
                    .map_err(|_| format!("invalid benchmark default argument {part:?}"))
            })
            .collect::<Result<_, _>>()?;
        if values.len() != 3 {
            return Err("runnable benchmark defaults must be: warmup iterations repetitions".into());
        }
        Self {
            warmup: values[0],
            iterations: values[1],
            repetitions: values[2],
        }
        .validate()
    }
}

#[derive(Clone, Debug)]
pub struct BenchExecution {
    pub benchmark_id: String,
    pub records: Vec<BenchRunRecord>,
    pub comparison: BenchComparison,
    pub legacy_text: String,
}

impl BenchExecution {
    pub fn render(&self, format: CatalogFormat) -> Result<String, String> {
        match format {
            CatalogFormat::Text => {
                let mut out = format!(
                    "bench-run | id={} records={} reference={} candidate={}\n",
                    self.benchmark_id,
                    self.records.len(),
                    self.comparison.reference_backend,
                    self.comparison.candidate_backend,
                );
                for record in &self.records {
                    let summary = record.summary()?;
                    out.push_str(&format!(
                        "bench-result | id={} backend={} median_ns={:.3} q1_ns={:.3} q3_ns={:.3} iqr_ns={:.3} repetitions={}\n",
                        record.benchmark_id,
                        record.backend,
                        summary.median_ns,
                        summary.q1_ns,
                        summary.q3_ns,
                        summary.iqr_ns,
                        record.repetitions,
                    ));
                }
                out.push_str(&format!(
                    "bench-comparison | id={} reference={} candidate={} candidate_over_reference={:.6} speedup={:.6}\n",
                    self.comparison.benchmark_id,
                    self.comparison.reference_backend,
                    self.comparison.candidate_backend,
                    self.comparison.candidate_over_reference,
                    self.comparison.speedup,
                ));
                Ok(out)
            }
            CatalogFormat::Json => {
                let mut out = String::new();
                for record in &self.records {
                    out.push_str(&record.json()?);
                    out.push('\n');
                }
                out.push_str(&self.comparison.json());
                out.push('\n');
                Ok(out)
            }
            CatalogFormat::Csv => {
                let mut out = BenchRunRecord::csv_header().to_string();
                for record in &self.records {
                    out.push_str(&record.csv_row()?);
                }
                Ok(out)
            }
        }
    }
}

pub fn default_protocol(id: &str) -> Result<BenchProtocol, String> {
    let spec = find(id).ok_or_else(|| format!("unknown bench {id}"))?;
    if spec.runner.is_none() {
        return Err(format!("bench {} is catalog-only and has no library runner", spec.id));
    }
    BenchProtocol::from_default_args(spec.default_args)
}

pub fn run(id: &str, protocol: BenchProtocol) -> Result<BenchExecution, String> {
    let protocol = protocol.validate()?;
    let spec = find(id).ok_or_else(|| format!("unknown bench {id}"))?;
    match spec.runner {
        Some(BenchRunnerKind::Matmul) => run_matmul(protocol),
        Some(BenchRunnerKind::Engine) => run_engine(protocol),
        None => Err(format!("bench {} is catalog-only and has no library runner", spec.id)),
    }
}

pub fn run_legacy(id: &str, protocol: BenchProtocol) -> Result<String, String> {
    run(id, protocol).map(|execution| execution.legacy_text)
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
    if let Ok(value) = env::var("AURALIS_TOOLCHAIN") {
        if !value.trim().is_empty() {
            return value;
        }
    }
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

fn hash_values(values: &[u64]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &value in values {
        h ^= value;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().max(1).min(u64::MAX as u128) as u64
}

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

// ---- Matmul runner -------------------------------------------------------

#[derive(Clone, Copy)]
enum MatmulMode {
    Reference,
    RowSlices,
}

impl MatmulMode {
    fn legacy_label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::RowSlices => "slices",
        }
    }
}

struct MatmulCase {
    rows: usize,
    inner: usize,
    cols: usize,
    weight: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

fn matmul_data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 37 + salt * 19 + i / 3) % 101) as f32;
            (raw - 50.0) / 31.0
        })
        .collect()
}

fn matmul_cases() -> Vec<MatmulCase> {
    [
        (32, 32, 100, 1),
        (32, 96, 32, 2),
        (32, 32, 96, 2),
        (32, 32, 32, 8),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (rows, inner, cols, weight))| MatmulCase {
        rows,
        inner,
        cols,
        weight,
        a: matmul_data(rows * inner, index + 3),
        b: matmul_data(inner * cols, index + 17),
    })
    .collect()
}

fn validate_matmul_exact(cases: &[MatmulCase]) -> bool {
    cases.iter().all(|case| {
        let mut reference = vec![f32::NAN; case.rows * case.cols];
        let mut sliced = vec![f32::NAN; case.rows * case.cols];
        matmul_reference_into(
            &case.a,
            case.rows,
            case.inner,
            &case.b,
            case.cols,
            &mut reference,
        );
        matmul_row_slices_into(
            &case.a,
            case.rows,
            case.inner,
            &case.b,
            case.cols,
            &mut sliced,
        );
        reference == sliced
    })
}

fn run_matmul_suite(mode: MatmulMode, iterations: usize, cases: &[MatmulCase]) -> u64 {
    let mut outputs: Vec<Vec<f32>> = cases
        .iter()
        .map(|case| vec![0.0; case.rows * case.cols])
        .collect();
    let mut checksum = 0.0f64;
    let started = Instant::now();
    for iteration in 0..iterations {
        for (case_index, case) in cases.iter().enumerate() {
            for repetition in 0..case.weight {
                let out = &mut outputs[case_index];
                match mode {
                    MatmulMode::Reference => matmul_reference_into(
                        &case.a,
                        case.rows,
                        case.inner,
                        &case.b,
                        case.cols,
                        out,
                    ),
                    MatmulMode::RowSlices => matmul_row_slices_into(
                        &case.a,
                        case.rows,
                        case.inner,
                        &case.b,
                        case.cols,
                        out,
                    ),
                }
                let probe = (iteration + repetition + case_index) % out.len();
                checksum += out[probe] as f64;
            }
        }
    }
    black_box(checksum);
    elapsed_ns(started)
}

fn run_matmul(protocol: BenchProtocol) -> Result<BenchExecution, String> {
    let cases = matmul_cases();
    if !validate_matmul_exact(&cases) {
        return Err("row-slices matmul does not match scalar reference exactly".into());
    }

    if protocol.warmup > 0 {
        let _ = run_matmul_suite(MatmulMode::Reference, protocol.warmup, &cases);
        let _ = run_matmul_suite(MatmulMode::RowSlices, protocol.warmup, &cases);
    }

    let calls_per_iteration: usize = cases.iter().map(|case| case.weight).sum();
    let mut reference_raw = Vec::with_capacity(protocol.repetitions);
    let mut slices_raw = Vec::with_capacity(protocol.repetitions);
    let mut speedups = Vec::with_capacity(protocol.repetitions);
    let mut legacy = String::new();

    for repetition in 0..protocol.repetitions {
        let reference_first = repetition % 2 == 0;
        let (reference, slices) = if reference_first {
            (
                run_matmul_suite(MatmulMode::Reference, protocol.iterations, &cases),
                run_matmul_suite(MatmulMode::RowSlices, protocol.iterations, &cases),
            )
        } else {
            let slices = run_matmul_suite(MatmulMode::RowSlices, protocol.iterations, &cases);
            let reference = run_matmul_suite(MatmulMode::Reference, protocol.iterations, &cases);
            (reference, slices)
        };
        let speedup = reference as f64 / slices as f64;
        reference_raw.push(reference);
        slices_raw.push(slices);
        speedups.push(speedup);
        legacy.push_str(&format!(
            "matmul_bench_run | repetition={} first={} iterations={} calls_per_iteration={} reference_ms={:.3} slices_ms={:.3} speedup={:.4} exact_output=true\n",
            repetition + 1,
            if reference_first { MatmulMode::Reference.legacy_label() } else { MatmulMode::RowSlices.legacy_label() },
            protocol.iterations,
            calls_per_iteration,
            reference as f64 / 1_000_000.0,
            slices as f64 / 1_000_000.0,
            speedup,
        ));
    }

    let hw = hardware();
    let tc = toolchain();
    let fingerprint = hash_values(&[
        0x6d61746d756c,
        protocol.warmup as u64,
        protocol.iterations as u64,
        protocol.repetitions as u64,
        calls_per_iteration as u64,
    ]);
    let reference_record = BenchRunRecord {
        benchmark_id: "matmul".into(),
        backend: "reference".into(),
        hardware: hw.clone(),
        toolchain: tc.clone(),
        config_fingerprint: fingerprint,
        warmup_iterations: protocol.warmup,
        repetitions: protocol.repetitions,
        raw_ns: reference_raw,
    };
    let slices_record = BenchRunRecord {
        benchmark_id: "matmul".into(),
        backend: "row_slices".into(),
        hardware: hw,
        toolchain: tc,
        config_fingerprint: fingerprint,
        warmup_iterations: protocol.warmup,
        repetitions: protocol.repetitions,
        raw_ns: slices_raw,
    };
    let reference_summary = reference_record.summary()?;
    let slices_summary = slices_record.summary()?;
    let median_speedup = median_f64(&mut speedups);
    legacy.push_str(&format!(
        "matmul_bench_summary | warmup_iterations={} measured_iterations={} repeats={} calls_per_iteration={} median_reference_ms={:.3} median_slices_ms={:.3} median_speedup={:.4} exact_output=true\n",
        protocol.warmup,
        protocol.iterations,
        protocol.repetitions,
        calls_per_iteration,
        reference_summary.median_ns / 1_000_000.0,
        slices_summary.median_ns / 1_000_000.0,
        median_speedup,
    ));
    let comparison = BenchComparison::compare(&reference_record, &slices_record)?;
    Ok(BenchExecution {
        benchmark_id: "matmul".into(),
        records: vec![reference_record, slices_record],
        comparison,
        legacy_text: legacy,
    })
}

// ---- Engine runner -------------------------------------------------------

#[derive(Clone, Copy)]
enum EngineMode {
    Reference,
    Reuse,
}

impl EngineMode {
    fn legacy_label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Reuse => "reuse",
        }
    }
}

struct EngineRun {
    elapsed_ns: u64,
    processed_tokens: u64,
    params: Vec<f32>,
    adam_t: i32,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
}

fn engine_model(seed: u64) -> Gpt {
    let mut rng = StdRng::seed_from_u64(seed);
    Gpt::new(Config::tiny(100), &mut rng)
}

fn engine_config() -> TrainConfig {
    TrainConfig {
        seed: 659_918,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: 1.0,
    }
}

fn engine_tokens() -> Vec<usize> {
    (0..4096)
        .map(|i| ((i * 37 + i / 7 + 11) % 100) as usize)
        .collect()
}

fn engine_step(
    mode: EngineMode,
    model: &mut Gpt,
    adam: &mut Adam,
    tokens: &[usize],
    cfg: TrainConfig,
    grads: &mut [f32],
    workspace: &mut TrainWorkspace,
) -> Result<usize, String> {
    let global_step = adam.t.max(0) as u64;
    let metrics = match mode {
        EngineMode::Reference => train_step(model, adam, tokens, cfg, global_step, grads),
        EngineMode::Reuse => train_step_reuse(
            model,
            adam,
            tokens,
            cfg,
            global_step,
            grads,
            workspace,
        ),
    }
    .map_err(str::to_string)?;
    Ok(metrics.tokens)
}

fn run_engine_mode(
    mode: EngineMode,
    protocol: BenchProtocol,
    tokens: &[usize],
) -> Result<EngineRun, String> {
    let mut model = engine_model(1234);
    let params = model.collect_params().len();
    let mut adam = Adam::new(params, 3e-3);
    let mut grads = vec![0.0; params];
    let mut workspace = TrainWorkspace::new(&model);
    let cfg = engine_config();

    for _ in 0..protocol.warmup {
        let _ = engine_step(
            mode,
            &mut model,
            &mut adam,
            tokens,
            cfg,
            &mut grads,
            &mut workspace,
        )?;
    }

    let started = Instant::now();
    let mut processed_tokens = 0u64;
    for _ in 0..protocol.iterations {
        processed_tokens += engine_step(
            mode,
            &mut model,
            &mut adam,
            tokens,
            cfg,
            &mut grads,
            &mut workspace,
        )? as u64;
    }
    let elapsed_ns = elapsed_ns(started);
    let params = model.collect_params();
    let (_, adam_t, m, v) = adam.export();
    Ok(EngineRun {
        elapsed_ns,
        processed_tokens,
        params,
        adam_t,
        adam_m: m.to_vec(),
        adam_v: v.to_vec(),
    })
}

fn exact_engine_state(a: &EngineRun, b: &EngineRun) -> bool {
    a.params == b.params
        && a.adam_t == b.adam_t
        && a.adam_m == b.adam_m
        && a.adam_v == b.adam_v
}

fn run_engine(protocol: BenchProtocol) -> Result<BenchExecution, String> {
    let tokens = engine_tokens();
    let mut reference_raw = Vec::with_capacity(protocol.repetitions);
    let mut reuse_raw = Vec::with_capacity(protocol.repetitions);
    let mut ratios = Vec::with_capacity(protocol.repetitions);
    let mut reference_ms = Vec::with_capacity(protocol.repetitions);
    let mut reuse_ms = Vec::with_capacity(protocol.repetitions);
    let mut legacy = String::new();

    for repetition in 0..protocol.repetitions {
        let reference_first = repetition % 2 == 0;
        let (reference, reuse) = if reference_first {
            (
                run_engine_mode(EngineMode::Reference, protocol, &tokens)?,
                run_engine_mode(EngineMode::Reuse, protocol, &tokens)?,
            )
        } else {
            let reuse = run_engine_mode(EngineMode::Reuse, protocol, &tokens)?;
            let reference = run_engine_mode(EngineMode::Reference, protocol, &tokens)?;
            (reference, reuse)
        };
        if !exact_engine_state(&reference, &reuse) {
            return Err(format!(
                "Engine candidate changed final training state in repetition {}",
                repetition + 1
            ));
        }
        if reference.processed_tokens != reuse.processed_tokens {
            return Err("Engine reference/reuse processed token mismatch".into());
        }

        let reference_seconds = reference.elapsed_ns as f64 / 1_000_000_000.0;
        let reuse_seconds = reuse.elapsed_ns as f64 / 1_000_000_000.0;
        let reference_rate = reference.processed_tokens as f64 / reference_seconds;
        let reuse_rate = reuse.processed_tokens as f64 / reuse_seconds;
        let ratio = reuse_rate / reference_rate;
        let reference_step_ms =
            reference.elapsed_ns as f64 / protocol.iterations as f64 / 1_000_000.0;
        let reuse_step_ms =
            reuse.elapsed_ns as f64 / protocol.iterations as f64 / 1_000_000.0;

        reference_raw.push(reference.elapsed_ns);
        reuse_raw.push(reuse.elapsed_ns);
        ratios.push(ratio);
        reference_ms.push(reference_step_ms);
        reuse_ms.push(reuse_step_ms);

        legacy.push_str(&format!(
            "engine_bench_run | repetition={} first={} warmup_steps={} measured_steps={} reference_tok_per_s={:.3} reuse_tok_per_s={:.3} ratio={:.4} reference_ms_per_step={:.4} reuse_ms_per_step={:.4} exact_state=true\n",
            repetition + 1,
            if reference_first { EngineMode::Reference.legacy_label() } else { EngineMode::Reuse.legacy_label() },
            protocol.warmup,
            protocol.iterations,
            reference_rate,
            reuse_rate,
            ratio,
            reference_step_ms,
            reuse_step_ms,
        ));
    }

    let hw = hardware();
    let tc = toolchain();
    let cfg = Config::tiny(100);
    let train = engine_config();
    let fingerprint = hash_values(&[
        0x656e67696e65,
        protocol.warmup as u64,
        protocol.iterations as u64,
        protocol.repetitions as u64,
        cfg.vocab as u64,
        cfg.n_embd as u64,
        cfg.n_head as u64,
        cfg.n_layer as u64,
        cfg.block as u64,
        cfg.n_ff as u64,
        train.seed,
        train.batch_size as u64,
        train.gradient_accumulation_steps as u64,
        train.grad_clip_norm.to_bits() as u64,
    ]);
    let reference_record = BenchRunRecord {
        benchmark_id: "engine".into(),
        backend: "reference".into(),
        hardware: hw.clone(),
        toolchain: tc.clone(),
        config_fingerprint: fingerprint,
        warmup_iterations: protocol.warmup,
        repetitions: protocol.repetitions,
        raw_ns: reference_raw,
    };
    let reuse_record = BenchRunRecord {
        benchmark_id: "engine".into(),
        backend: "reuse".into(),
        hardware: hw,
        toolchain: tc,
        config_fingerprint: fingerprint,
        warmup_iterations: protocol.warmup,
        repetitions: protocol.repetitions,
        raw_ns: reuse_raw,
    };

    let reference_summary = reference_record.summary()?;
    let reuse_summary = reuse_record.summary()?;
    let median_reference_rate = {
        let ns = reference_summary.median_ns;
        (protocol.iterations as f64 * 128.0) / (ns / 1_000_000_000.0)
    };
    let median_reuse_rate = {
        let ns = reuse_summary.median_ns;
        (protocol.iterations as f64 * 128.0) / (ns / 1_000_000_000.0)
    };
    let median_ratio = median_f64(&mut ratios);
    let median_reference_ms = median_f64(&mut reference_ms);
    let median_reuse_ms = median_f64(&mut reuse_ms);
    legacy.push_str(&format!(
        "engine_bench_summary | warmup_steps={} measured_steps={} repeats={} median_reference_tok_per_s={:.3} median_reuse_tok_per_s={:.3} median_ratio={:.4} median_reference_ms_per_step={:.4} median_reuse_ms_per_step={:.4} exact_state=true\n",
        protocol.warmup,
        protocol.iterations,
        protocol.repetitions,
        median_reference_rate,
        median_reuse_rate,
        median_ratio,
        median_reference_ms,
        median_reuse_ms,
    ));

    let comparison = BenchComparison::compare(&reference_record, &reuse_record)?;
    Ok(BenchExecution {
        benchmark_id: "engine".into(),
        records: vec![reference_record, reuse_record],
        comparison,
        legacy_text: legacy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_rejects_zero_work() {
        assert!(BenchProtocol {
            warmup: 0,
            iterations: 0,
            repetitions: 1,
        }
        .validate()
        .is_err());
        assert!(BenchProtocol {
            warmup: 0,
            iterations: 1,
            repetitions: 0,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn default_protocols_are_runnable() {
        assert_eq!(
            default_protocol("matmul").unwrap(),
            BenchProtocol {
                warmup: 5,
                iterations: 40,
                repetitions: 5,
            }
        );
        assert_eq!(
            default_protocol("engine").unwrap(),
            BenchProtocol {
                warmup: 3,
                iterations: 20,
                repetitions: 5,
            }
        );
    }

    #[test]
    fn tiny_library_runs_emit_records() {
        for id in ["matmul", "engine"] {
            let run = run(
                id,
                BenchProtocol {
                    warmup: 0,
                    iterations: 1,
                    repetitions: 1,
                },
            )
            .unwrap();
            assert_eq!(run.records.len(), 2);
            assert_eq!(run.records[0].benchmark_id, id);
            assert!(run.legacy_text.contains("_bench_summary |"));
        }
    }
}
