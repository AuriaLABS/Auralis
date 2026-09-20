use auralis::model::{Config, Gpt};
use auralis::optim::{Optimizer, OptimizerDiagnostics, OptimizerId};
use auralis::training::{train_step_reuse, TrainConfig, TrainWorkspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::time::Instant;

struct NoopOptimizer {
    n: usize,
    step: u64,
    empty: Vec<f32>,
}

impl NoopOptimizer {
    fn new(n: usize) -> Self {
        Self {
            n,
            step: 0,
            empty: Vec::new(),
        }
    }
}

impl Optimizer for NoopOptimizer {
    fn id(&self) -> OptimizerId {
        OptimizerId::Adam
    }

    fn learning_rate(&self) -> f32 {
        1e-3
    }

    fn set_learning_rate(&mut self, _learning_rate: f32) -> Result<(), &'static str> {
        Ok(())
    }

    fn global_step(&self) -> u64 {
        self.step
    }

    fn parameter_count(&self) -> usize {
        self.n
    }

    fn config_fingerprint(&self) -> u64 {
        0x96
    }

    fn state_fingerprint(&self) -> u64 {
        self.step
    }

    fn diagnostics(&self) -> OptimizerDiagnostics<'_> {
        OptimizerDiagnostics {
            first_name: "noop_1",
            first: &self.empty,
            second_name: "noop_2",
            second: &self.empty,
        }
    }

    fn update(
        &mut self,
        parameters: &mut [f32],
        gradients: &[f32],
    ) -> Result<(), &'static str> {
        if parameters.len() != self.n || gradients.len() != self.n {
            return Err("noop optimizer size mismatch");
        }
        self.step += 1;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Off,
    Forced,
}

struct Run {
    ns_per_step: f64,
    clip_count: usize,
    first_pre: f32,
    first_post: f32,
    final_params: Vec<f32>,
}

fn tokens() -> Vec<usize> {
    (0..8192)
        .map(|i| (i * 37 + i / 7 + 11) % 100)
        .collect()
}

fn run(mode: Mode, warmup: usize, measured: usize) -> Run {
    let mut rng = StdRng::seed_from_u64(0x96);
    let mut gpt = Gpt::new(Config::tiny(100), &mut rng);
    let n = gpt.collect_params().len();
    let mut optimizer = NoopOptimizer::new(n);
    let mut workspace = TrainWorkspace::new(&gpt);
    let mut grads = vec![0.0f32; n];
    let stream = tokens();
    let cfg = TrainConfig {
        seed: 659_918,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: match mode {
            Mode::Off => f32::MAX,
            Mode::Forced => 1e-4,
        },
    };

    for step in 0..warmup {
        let _ = train_step_reuse(
            &mut gpt,
            &mut optimizer,
            &stream,
            cfg,
            step as u64,
            &mut grads,
            &mut workspace,
        )
        .unwrap();
    }

    let started = Instant::now();
    let mut clip_count = 0usize;
    let mut first_pre = 0.0f32;
    let mut first_post = 0.0f32;
    for i in 0..measured {
        let metrics = train_step_reuse(
            &mut gpt,
            &mut optimizer,
            &stream,
            cfg,
            (warmup + i) as u64,
            &mut grads,
            &mut workspace,
        )
        .unwrap();
        if i == 0 {
            first_pre = metrics.grad_norm_before_clip;
            first_post = metrics.grad_norm_after_clip;
        }
        clip_count += usize::from(metrics.clip_applied);
        match mode {
            Mode::Off => {
                assert!(!metrics.clip_applied);
                assert_eq!(metrics.grad_scale, 1.0);
                assert_eq!(
                    metrics.grad_norm_before_clip.to_bits(),
                    metrics.grad_norm_after_clip.to_bits()
                );
            }
            Mode::Forced => {
                assert!(metrics.clip_applied);
                assert!(metrics.grad_norm_after_clip <= 1.0001e-4);
                assert!(metrics.grad_norm_after_clip < metrics.grad_norm_before_clip);
            }
        }
    }
    let elapsed = started.elapsed().as_nanos() as f64 / measured as f64;
    Run {
        ns_per_step: elapsed,
        clip_count,
        first_pre,
        first_post,
        final_params: gpt.collect_params(),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let warmup = args.get(1).and_then(|x| x.parse().ok()).unwrap_or(3usize);
    let measured = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(30usize);
    let repeats = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(5usize);
    assert!(measured > 0 && repeats >= 3);

    let mut off_raw = Vec::with_capacity(repeats);
    let mut forced_raw = Vec::with_capacity(repeats);
    let mut first_pre = 0.0f32;
    let mut first_post = 0.0f32;

    for rep in 0..repeats {
        let off_first = rep % 2 == 0;
        let (off, forced) = if off_first {
            (run(Mode::Off, warmup, measured), run(Mode::Forced, warmup, measured))
        } else {
            let forced = run(Mode::Forced, warmup, measured);
            let off = run(Mode::Off, warmup, measured);
            (off, forced)
        };

        assert_eq!(off.final_params, forced.final_params);
        assert_eq!(off.clip_count, 0);
        assert_eq!(forced.clip_count, measured);
        first_pre = forced.first_pre;
        first_post = forced.first_post;
        off_raw.push(off.ns_per_step);
        forced_raw.push(forced.ns_per_step);

        println!(
            "clipping_bench_run | repetition={} first={} off_ns_per_step={:.3} forced_ns_per_step={:.3} forced_over_off={:.4} off_clips={} forced_clips={} params_exact=true",
            rep + 1,
            if off_first { "off" } else { "forced" },
            off.ns_per_step,
            forced.ns_per_step,
            forced.ns_per_step / off.ns_per_step,
            off.clip_count,
            forced.clip_count,
        );
    }

    let off = median(&mut off_raw);
    let forced = median(&mut forced_raw);
    println!(
        "clipping_bench_summary | warmup={} measured_steps={} repeats={} off_ns_per_step={:.3} forced_ns_per_step={:.3} forced_over_off={:.4} forced_first_pre={:.6} forced_first_post={:.6} params_exact=true",
        warmup,
        measured,
        repeats,
        off,
        forced,
        forced / off,
        first_pre,
        first_post,
    );
}
