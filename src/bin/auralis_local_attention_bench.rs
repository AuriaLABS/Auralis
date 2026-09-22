use auralis::brain_ab::{
    run_attention_window_experiment, AbProtocol, AbVariant, AttentionWindowExperimentResult,
    ATTENTION_WINDOW_AB_SCHEMA_VERSION,
};
use auralis::model::{Config, Gpt, NormalizationKind};
use auralis::optim::OptimizerId;
use auralis::position::PositionKind;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::hint::black_box;
use std::time::Instant;

fn quality_config() -> Config {
    Config {
        vocab: 32,
        n_embd: 32,
        n_head: 4,
        n_layer: 2,
        block: 32,
        n_ff: 96,
    }
}

fn scaling_config() -> Config {
    Config {
        block: 64,
        ..quality_config()
    }
}

fn tokens(vocab: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| (i * 29 + i / 3 + 7) % vocab)
        .collect()
}

fn variant(label: &str, cfg: Config) -> AbVariant {
    AbVariant {
        label: label.into(),
        config: cfg,
        normalization: NormalizationKind::LayerNorm,
        position: PositionKind::LearnedAbsolute,
        optimizer: OptimizerId::Adam,
    }
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn print_quality(label: &str, result: &AttentionWindowExperimentResult) {
    println!(
        "local_quality_protocol | label={} schema={} seed={} steps={} repeats={} tokens={}",
        label,
        result.schema_version,
        result.protocol.seed,
        result.protocol.steps,
        result.protocol.repeats,
        result.protocol.token_count,
    );
    for v in [&result.a, &result.b] {
        let mean_loss = v
            .measurements
            .iter()
            .map(|m| m.eval_loss as f64)
            .sum::<f64>()
            / v.measurements.len() as f64;
        let mean_ppl = v
            .measurements
            .iter()
            .map(|m| m.eval_perplexity as f64)
            .sum::<f64>()
            / v.measurements.len() as f64;
        let mean_tps = v
            .measurements
            .iter()
            .map(|m| m.tokens_per_second)
            .sum::<f64>()
            / v.measurements.len() as f64;
        println!(
            "local_quality | label={} variant={} window={} mean_eval_loss={:.6} mean_ppl={:.6} mean_tok_per_s={:.3} params={}",
            label,
            v.label,
            v.attention_window,
            mean_loss,
            mean_ppl,
            mean_tps,
            v.measurements[0].parameter_count,
        );
    }
}

fn measure_curve(cfg: Config, context: usize, window: usize, repeats: usize) {
    let mut rng = StdRng::seed_from_u64(0xA11CE_1114);
    let gpt = Gpt::new_with_attention_policy(
        cfg,
        NormalizationKind::LayerNorm,
        PositionKind::LearnedAbsolute,
        cfg.n_head,
        window,
        &mut rng,
    );
    let stream = tokens(cfg.vocab, context + 1);
    let x = &stream[..context];
    let y = &stream[1..context + 1];

    let mut dense_rng = StdRng::seed_from_u64(0xA11CE_1114);
    let dense = Gpt::new(cfg, &mut dense_rng);
    assert_eq!(gpt.collect_params(), dense.collect_params());
    if window == 0 || window >= context {
        assert_eq!(gpt.logits(x), dense.logits(x));
    }

    black_box(gpt.logits(x));
    let mut warm_grads = vec![0.0f32; gpt.collect_params().len()];
    black_box(gpt.backward_into(x, y, &mut warm_grads));

    let mut forward = Vec::with_capacity(repeats);
    let mut backward = Vec::with_capacity(repeats);
    let mut loss = f32::NAN;
    let mut max_grad = 0.0f32;

    for _ in 0..repeats {
        let started = Instant::now();
        black_box(gpt.logits(x));
        forward.push(started.elapsed().as_nanos().max(1) as u64);

        let mut grads = vec![0.0f32; gpt.collect_params().len()];
        let started = Instant::now();
        loss = black_box(gpt.backward_into(x, y, &mut grads));
        backward.push(started.elapsed().as_nanos().max(1) as u64);
        assert!(loss.is_finite());
        assert!(grads.iter().all(|g| g.is_finite()));
        max_grad = grads
            .iter()
            .fold(max_grad, |acc, &g| acc.max(g.abs()));
    }

    let slots = gpt.attention_probability_slots(context).unwrap();
    let bytes = gpt.attention_probability_bytes(context).unwrap();
    println!(
        "local_attention_curve | context={} window={} forward_ns={} backward_ns={} probability_slots_per_layer={} probability_bytes_model={} params={} loss={:.6} max_abs_grad={:.6}",
        context,
        window,
        median(&mut forward),
        median(&mut backward),
        slots,
        bytes,
        gpt.collect_params().len(),
        loss,
        max_grad,
    );
}

fn main() {
    let quality_cfg = quality_config();
    let protocol = AbProtocol {
        seed: 659_918,
        steps: 4,
        repeats: 3,
        batch_size: 2,
        gradient_accumulation_steps: 1,
        learning_rate: 3e-3,
        grad_clip_norm: 1.0,
        token_count: 256,
    };

    println!(
        "local_attention_protocol | schema={} dense_window=0 local_windows=8,16 quality_steps={} quality_repeats={} scaling_repeats=5",
        ATTENTION_WINDOW_AB_SCHEMA_VERSION,
        protocol.steps,
        protocol.repeats,
    );

    for window in [8usize, 16] {
        let result = run_attention_window_experiment(
            protocol,
            variant("dense", quality_cfg),
            0,
            variant(&format!("local-w{window}"), quality_cfg),
            window,
            quality_cfg.n_head,
        )
        .unwrap();
        print_quality(&format!("dense_vs_w{window}"), &result);
    }

    let cfg = scaling_config();
    for context in [8usize, 16, 32, 64] {
        for window in [0usize, 8, 16] {
            measure_curve(cfg, context, window, 5);
        }
    }
}
