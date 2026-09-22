use auralis::brain_ab::{
    run_attention_head_experiment, AbProtocol, AbVariant, ATTENTION_HEAD_AB_SCHEMA_VERSION,
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

fn decode_config() -> Config {
    Config {
        block: 64,
        ..quality_config()
    }
}

fn tokens(vocab: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| (i * 17 + i / 7 + 3) % vocab)
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

fn decode_curve(cfg: Config, n_kv_head: usize, context: usize, repeats: usize) {
    assert!(context < cfg.block);
    let mut rng = StdRng::seed_from_u64(0xA11CE_1400);
    let gpt = Gpt::new_with_attention_heads(
        cfg,
        NormalizationKind::LayerNorm,
        PositionKind::LearnedAbsolute,
        n_kv_head,
        &mut rng,
    );
    let stream = tokens(cfg.vocab, context + 1);
    let prefix = &stream[..context];
    let next = stream[context];

    let baseline_prefix = gpt.logits(prefix);
    let mut check_cache = gpt.new_kv_cache();
    let cached_prefix = gpt.prefill_kv_cache(prefix, &mut check_cache).unwrap();
    assert_eq!(cached_prefix, baseline_prefix);

    let full_with_next = gpt.logits(&stream);
    let expected_next = &full_with_next[context * cfg.vocab..(context + 1) * cfg.vocab];
    let cached_next = gpt.decode_kv_cached(next, &mut check_cache).unwrap();
    assert_eq!(cached_next, expected_next);

    let mut full_prefill_times = Vec::with_capacity(repeats);
    let mut cache_prefill_times = Vec::with_capacity(repeats);
    let mut uncached_next_times = Vec::with_capacity(repeats);
    let mut cached_next_times = Vec::with_capacity(repeats);
    let mut logical_bytes = 0usize;
    let mut allocated_bytes = 0usize;

    for _ in 0..repeats {
        let started = Instant::now();
        black_box(gpt.logits(prefix));
        full_prefill_times.push(started.elapsed().as_nanos().max(1) as u64);

        let mut cache = gpt.new_kv_cache();
        let started = Instant::now();
        black_box(gpt.prefill_kv_cache(prefix, &mut cache).unwrap());
        cache_prefill_times.push(started.elapsed().as_nanos().max(1) as u64);
        logical_bytes = cache.logical_bytes();
        allocated_bytes = cache.allocated_bytes();

        let started = Instant::now();
        black_box(gpt.logits(&stream));
        uncached_next_times.push(started.elapsed().as_nanos().max(1) as u64);

        let started = Instant::now();
        black_box(gpt.decode_kv_cached(next, &mut cache).unwrap());
        cached_next_times.push(started.elapsed().as_nanos().max(1) as u64);
    }

    let full_prefill_ns = median(&mut full_prefill_times);
    let cache_prefill_ns = median(&mut cache_prefill_times);
    let uncached_next_ns = median(&mut uncached_next_times);
    let cached_next_ns = median(&mut cached_next_times);
    println!(
        "gqa_decode | context={} n_head={} n_kv_head={} params={} kv_width={} logical_bytes={} allocated_bytes={} full_prefill_ns={} cache_prefill_ns={} uncached_next_ns={} cached_next_ns={} decode_speedup={:.4} exact=true",
        context,
        cfg.n_head,
        n_kv_head,
        gpt.collect_params().len(),
        gpt.kv_width(),
        logical_bytes,
        allocated_bytes,
        full_prefill_ns,
        cache_prefill_ns,
        uncached_next_ns,
        cached_next_ns,
        uncached_next_ns as f64 / cached_next_ns as f64,
    );
}

fn print_quality(label: &str, result: &auralis::brain_ab::AttentionHeadExperimentResult) {
    println!(
        "gqa_quality_protocol | label={} schema={} seed={} steps={} repeats={} tokens={}",
        label,
        result.schema_version,
        result.protocol.seed,
        result.protocol.steps,
        result.protocol.repeats,
        result.protocol.token_count,
    );
    for variant in [&result.a, &result.b] {
        let mean_loss = variant
            .measurements
            .iter()
            .map(|m| m.eval_loss as f64)
            .sum::<f64>()
            / variant.measurements.len() as f64;
        let mean_ppl = variant
            .measurements
            .iter()
            .map(|m| m.eval_perplexity as f64)
            .sum::<f64>()
            / variant.measurements.len() as f64;
        let mean_tps = variant
            .measurements
            .iter()
            .map(|m| m.tokens_per_second)
            .sum::<f64>()
            / variant.measurements.len() as f64;
        let params = variant.measurements[0].parameter_count;
        println!(
            "gqa_quality | label={} variant={} n_head={} n_kv_head={} mean_eval_loss={:.6} mean_ppl={:.6} mean_tok_per_s={:.3} params={}",
            label,
            variant.label,
            variant.config.n_head,
            variant.n_kv_head,
            mean_loss,
            mean_ppl,
            mean_tps,
            params,
        );
    }
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
        "gqa_mqa_protocol | attention_head_schema={} n_head={} variants=mha:4,gqa:2,mqa:1 quality_steps={} quality_repeats={} decode_repeats=5 decode_block=64",
        ATTENTION_HEAD_AB_SCHEMA_VERSION,
        quality_cfg.n_head,
        protocol.steps,
        protocol.repeats,
    );

    let mha_vs_gqa = run_attention_head_experiment(
        protocol,
        variant("mha", quality_cfg),
        4,
        variant("gqa", quality_cfg),
        2,
    )
    .unwrap();
    let mha_vs_mqa = run_attention_head_experiment(
        protocol,
        variant("mha", quality_cfg),
        4,
        variant("mqa", quality_cfg),
        1,
    )
    .unwrap();
    print_quality("mha_vs_gqa", &mha_vs_gqa);
    print_quality("mha_vs_mqa", &mha_vs_mqa);

    let decode_cfg = decode_config();
    for context in [8usize, 16, 32] {
        for n_kv_head in [4usize, 2, 1] {
            decode_curve(decode_cfg, n_kv_head, context, 5);
        }
    }
}
