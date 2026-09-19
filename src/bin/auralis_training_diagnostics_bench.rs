use auralis::model::{Config, Gpt};
use auralis::numeric::Diagnostics;
use auralis::optim::Adam;
use auralis::training::{
    train_step_reuse, train_step_reuse_diagnostics, TrainConfig, TrainWorkspace,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn config() -> Config {
    Config {
        vocab: 100,
        n_embd: 32,
        n_head: 4,
        n_layer: 2,
        block: 32,
        n_ff: 96,
    }
}

fn train_config() -> TrainConfig {
    TrainConfig {
        seed: 659_918,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: 1.0,
    }
}

fn stream() -> Vec<usize> {
    (0..4096)
        .map(|i| (i * 37 + i / 3 + 11) % config().vocab)
        .collect()
}

fn model(seed: u64) -> Gpt {
    let mut rng = StdRng::seed_from_u64(seed);
    Gpt::new(config(), &mut rng)
}

fn assert_clean_equivalence(tokens: &[usize]) {
    let mut reference = model(0xA11CE_9001);
    let mut diagnosed = reference.clone();
    let n = reference.collect_params().len();
    let mut adam_reference = Adam::new(n, 3e-3);
    let mut adam_diagnosed = Adam::new(n, 3e-3);
    let mut grads_reference = vec![0.0; n];
    let mut grads_diagnosed = vec![0.0; n];
    let mut workspace_reference = TrainWorkspace::new(&reference);
    let mut workspace_diagnosed = TrainWorkspace::new(&diagnosed);

    let a = train_step_reuse(
        &mut reference,
        &mut adam_reference,
        tokens,
        train_config(),
        0,
        &mut grads_reference,
        &mut workspace_reference,
    )
    .expect("reference step");

    let (b, report) = train_step_reuse_diagnostics(
        &mut diagnosed,
        &mut adam_diagnosed,
        tokens,
        train_config(),
        0,
        &mut grads_diagnosed,
        &mut workspace_diagnosed,
        Diagnostics::on(),
    )
    .expect("diagnosed step");

    let report = report.expect("diagnostics enabled");
    assert!(report.loss.is_finite());
    assert!(report.pre_optimizer.first_fault().is_none());
    assert!(report.post_optimizer.first_fault().is_none());
    assert_eq!(a, b);
    assert_eq!(grads_reference, grads_diagnosed);
    assert_eq!(reference.collect_params(), diagnosed.collect_params());
    assert_eq!(adam_reference.export().1, adam_diagnosed.export().1);
    assert_eq!(adam_reference.export().2, adam_diagnosed.export().2);
    assert_eq!(adam_reference.export().3, adam_diagnosed.export().3);
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn time_baseline(tokens: &[usize], steps: usize, repeats: usize) -> f64 {
    let mut samples = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut gpt = model(0xA11CE_9100 + repeat as u64);
        let n = gpt.collect_params().len();
        let mut adam = Adam::new(n, 3e-3);
        let mut grads = vec![0.0; n];
        let mut workspace = TrainWorkspace::new(&gpt);
        let start = Instant::now();
        for step in 0..steps {
            train_step_reuse(
                &mut gpt,
                &mut adam,
                tokens,
                train_config(),
                step as u64,
                &mut grads,
                &mut workspace,
            )
            .expect("baseline step");
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / steps as f64);
    }
    median(samples)
}

fn time_diagnostics_off(tokens: &[usize], steps: usize, repeats: usize) -> f64 {
    let mut samples = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut gpt = model(0xA11CE_9100 + repeat as u64);
        let n = gpt.collect_params().len();
        let mut adam = Adam::new(n, 3e-3);
        let mut grads = vec![0.0; n];
        let mut workspace = TrainWorkspace::new(&gpt);
        let start = Instant::now();
        for step in 0..steps {
            let (_, report) = train_step_reuse_diagnostics(
                &mut gpt,
                &mut adam,
                tokens,
                train_config(),
                step as u64,
                &mut grads,
                &mut workspace,
                Diagnostics::off(),
            )
            .expect("diagnostics-off wrapper step");
            assert!(report.is_none());
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / steps as f64);
    }
    median(samples)
}

fn time_diagnostics(tokens: &[usize], steps: usize, repeats: usize) -> f64 {
    let mut samples = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut gpt = model(0xA11CE_9100 + repeat as u64);
        let n = gpt.collect_params().len();
        let mut adam = Adam::new(n, 3e-3);
        let mut grads = vec![0.0; n];
        let mut workspace = TrainWorkspace::new(&gpt);
        let start = Instant::now();
        for step in 0..steps {
            train_step_reuse_diagnostics(
                &mut gpt,
                &mut adam,
                tokens,
                train_config(),
                step as u64,
                &mut grads,
                &mut workspace,
                Diagnostics::on(),
            )
            .expect("diagnosed step");
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / steps as f64);
    }
    median(samples)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let steps = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12usize);
    let repeats = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    assert!(steps > 0 && repeats > 0);

    let tokens = stream();
    assert_clean_equivalence(&tokens);

    let baseline = time_baseline(&tokens, steps, repeats);
    let diagnostics_off = time_diagnostics_off(&tokens, steps, repeats);
    let diagnostics = time_diagnostics(&tokens, steps, repeats);
    println!(
        "training_diagnostics_bench | steps={} repeats={} baseline_ns_per_step={:.3} off_wrapper_ns_per_step={:.3} off_over_baseline={:.4} diagnostics_ns_per_step={:.3} diagnostics_over_baseline={:.4}",
        steps,
        repeats,
        baseline,
        diagnostics_off,
        diagnostics_off / baseline,
        diagnostics,
        diagnostics / baseline
    );
}
