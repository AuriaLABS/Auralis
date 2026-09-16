use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use auralis::training::{train_step, train_step_reuse, TrainConfig, TrainWorkspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::time::Instant;

#[derive(Clone, Copy)]
enum BenchMode {
    Reference,
    Reuse,
}

impl BenchMode {
    fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Reuse => "reuse",
        }
    }
}

struct BenchRun {
    seconds: f64,
    processed_tokens: u64,
    final_params: Vec<f32>,
    adam_t: i32,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
}

fn make_model(seed: u64) -> Gpt {
    let mut rng = StdRng::seed_from_u64(seed);
    Gpt::new(Config::tiny(100), &mut rng)
}

fn bench_config() -> TrainConfig {
    TrainConfig {
        seed: 659_918,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: 1.0,
    }
}

fn token_stream() -> Vec<usize> {
    (0..4096)
        .map(|i| ((i * 37 + i / 7 + 11) % 100) as usize)
        .collect()
}

fn execute_step(
    mode: BenchMode,
    model: &mut Gpt,
    adam: &mut Adam,
    tokens: &[usize],
    cfg: TrainConfig,
    grads: &mut [f32],
    workspace: &mut TrainWorkspace,
) -> usize {
    let global_step = adam.t.max(0) as u64;
    let metrics = match mode {
        BenchMode::Reference => train_step(model, adam, tokens, cfg, global_step, grads),
        BenchMode::Reuse => {
            train_step_reuse(model, adam, tokens, cfg, global_step, grads, workspace)
        }
    }
    .expect("benchmark training step");
    metrics.tokens
}

fn run_mode(
    mode: BenchMode,
    warmup_steps: usize,
    measured_steps: usize,
    cfg: TrainConfig,
    tokens: &[usize],
) -> BenchRun {
    let mut model = make_model(1234);
    let params = model.collect_params().len();
    let mut adam = Adam::new(params, 3e-3);
    let mut grads = vec![0.0; params];
    let mut workspace = TrainWorkspace::new(&model);

    for _ in 0..warmup_steps {
        execute_step(
            mode,
            &mut model,
            &mut adam,
            tokens,
            cfg,
            &mut grads,
            &mut workspace,
        );
    }

    let started = Instant::now();
    let mut processed_tokens = 0u64;
    for _ in 0..measured_steps {
        processed_tokens += execute_step(
            mode,
            &mut model,
            &mut adam,
            tokens,
            cfg,
            &mut grads,
            &mut workspace,
        ) as u64;
    }
    let seconds = started.elapsed().as_secs_f64();

    let final_params = model.collect_params();
    let (_, adam_t, m, v) = adam.export();
    BenchRun {
        seconds,
        processed_tokens,
        final_params,
        adam_t,
        adam_m: m.to_vec(),
        adam_v: v.to_vec(),
    }
}

fn exact_state(a: &BenchRun, b: &BenchRun) -> bool {
    a.final_params == b.final_params
        && a.adam_t == b.adam_t
        && a.adam_m == b.adam_m
        && a.adam_v == b.adam_v
}

fn tokens_per_second(run: &BenchRun) -> f64 {
    if run.seconds > 0.0 {
        run.processed_tokens as f64 / run.seconds
    } else {
        f64::INFINITY
    }
}

fn milliseconds_per_step(run: &BenchRun, measured_steps: usize) -> f64 {
    1000.0 * run.seconds / measured_steps as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn parse_arg(index: usize, default: usize) -> usize {
    env::args()
        .nth(index)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() {
    let warmup_steps = parse_arg(1, 5);
    let measured_steps = parse_arg(2, 40);
    let repeats = parse_arg(3, 5);
    if measured_steps == 0 || repeats == 0 {
        eprintln!("benchmark measured_steps and repeats must be positive");
        std::process::exit(2);
    }

    let cfg = bench_config();
    let tokens = token_stream();
    let mut reference_rates = Vec::with_capacity(repeats);
    let mut reuse_rates = Vec::with_capacity(repeats);
    let mut ratios = Vec::with_capacity(repeats);
    let mut reference_ms = Vec::with_capacity(repeats);
    let mut reuse_ms = Vec::with_capacity(repeats);
    let mut all_exact = true;

    for repetition in 0..repeats {
        let reference_first = repetition % 2 == 0;
        let (reference, reuse) = if reference_first {
            (
                run_mode(
                    BenchMode::Reference,
                    warmup_steps,
                    measured_steps,
                    cfg,
                    &tokens,
                ),
                run_mode(
                    BenchMode::Reuse,
                    warmup_steps,
                    measured_steps,
                    cfg,
                    &tokens,
                ),
            )
        } else {
            let reuse = run_mode(
                BenchMode::Reuse,
                warmup_steps,
                measured_steps,
                cfg,
                &tokens,
            );
            let reference = run_mode(
                BenchMode::Reference,
                warmup_steps,
                measured_steps,
                cfg,
                &tokens,
            );
            (reference, reuse)
        };

        let exact = exact_state(&reference, &reuse);
        all_exact &= exact;
        let reference_rate = tokens_per_second(&reference);
        let reuse_rate = tokens_per_second(&reuse);
        let ratio = reuse_rate / reference_rate;
        let reference_step_ms = milliseconds_per_step(&reference, measured_steps);
        let reuse_step_ms = milliseconds_per_step(&reuse, measured_steps);

        reference_rates.push(reference_rate);
        reuse_rates.push(reuse_rate);
        ratios.push(ratio);
        reference_ms.push(reference_step_ms);
        reuse_ms.push(reuse_step_ms);

        println!(
            concat!(
                "engine_bench_run | repetition={} first={} warmup_steps={} measured_steps={} ",
                "reference_tok_per_s={:.3} reuse_tok_per_s={:.3} ratio={:.4} ",
                "reference_ms_per_step={:.4} reuse_ms_per_step={:.4} exact_state={}"
            ),
            repetition + 1,
            if reference_first {
                BenchMode::Reference.label()
            } else {
                BenchMode::Reuse.label()
            },
            warmup_steps,
            measured_steps,
            reference_rate,
            reuse_rate,
            ratio,
            reference_step_ms,
            reuse_step_ms,
            exact,
        );

        if !exact {
            eprintln!("Engine candidate changed the final training state in repetition {}", repetition + 1);
            std::process::exit(3);
        }
    }

    let median_reference_rate = median(&mut reference_rates);
    let median_reuse_rate = median(&mut reuse_rates);
    let median_ratio = median(&mut ratios);
    let median_reference_ms = median(&mut reference_ms);
    let median_reuse_ms = median(&mut reuse_ms);

    println!(
        concat!(
            "engine_bench_summary | warmup_steps={} measured_steps={} repeats={} ",
            "median_reference_tok_per_s={:.3} median_reuse_tok_per_s={:.3} median_ratio={:.4} ",
            "median_reference_ms_per_step={:.4} median_reuse_ms_per_step={:.4} exact_state={}"
        ),
        warmup_steps,
        measured_steps,
        repeats,
        median_reference_rate,
        median_reuse_rate,
        median_ratio,
        median_reference_ms,
        median_reuse_ms,
        all_exact,
    );
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn median_handles_odd_and_even_samples() {
        let mut odd = [3.0, 1.0, 2.0];
        let mut even = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut odd), 2.0);
        assert_eq!(median(&mut even), 2.5);
    }
}
