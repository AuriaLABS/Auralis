use auralis::model::{Config, Gpt};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct Row {
    context: usize,
    uncached_ns: u64,
    cached_ns: u64,
    logical_bytes: usize,
    allocated_bytes: usize,
    params: usize,
}

fn config() -> Config {
    Config {
        vocab: 64,
        n_embd: 32,
        n_head: 4,
        n_layer: 2,
        block: 64,
        n_ff: 96,
    }
}

fn tokens(vocab: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| (i * 37 + i / 5 + 11) % vocab)
        .collect()
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn measure(gpt: &Gpt, stream: &[usize], context: usize, repeats: usize) -> Row {
    let prefix = &stream[..context];

    let baseline = gpt.logits(prefix);
    let mut correctness_cache = gpt.new_kv_cache();
    let cached = gpt
        .prefill_kv_cache(prefix, &mut correctness_cache)
        .expect("kv-cache correctness prefill");
    assert_eq!(cached, baseline, "cached/uncached mismatch at context {context}");

    // Warm both paths once before recording hosted-runner timing.
    for end in 1..=context {
        black_box(gpt.logits(&prefix[..end]));
    }
    let mut warm_cache = gpt.new_kv_cache();
    for &token in prefix {
        black_box(
            gpt.decode_kv_cached(token, &mut warm_cache)
                .expect("kv-cache warm decode"),
        );
    }

    let mut uncached = Vec::with_capacity(repeats);
    let mut cached_times = Vec::with_capacity(repeats);
    let mut logical_bytes = 0usize;
    let mut allocated_bytes = 0usize;

    for _ in 0..repeats {
        let started = Instant::now();
        for end in 1..=context {
            black_box(gpt.logits(&prefix[..end]));
        }
        uncached.push(started.elapsed().as_nanos().max(1) as u64);

        let mut cache = gpt.new_kv_cache();
        let started = Instant::now();
        for &token in prefix {
            black_box(
                gpt.decode_kv_cached(token, &mut cache)
                    .expect("kv-cache decode"),
            );
        }
        cached_times.push(started.elapsed().as_nanos().max(1) as u64);
        logical_bytes = cache.logical_bytes();
        allocated_bytes = cache.allocated_bytes();
        assert_eq!(cache.len(), context);
    }

    Row {
        context,
        uncached_ns: median(&mut uncached),
        cached_ns: median(&mut cached_times),
        logical_bytes,
        allocated_bytes,
        params: gpt.collect_params().len(),
    }
}

fn main() {
    let cfg = config();
    let mut rng = StdRng::seed_from_u64(0xA11CE_4000);
    let gpt = Gpt::new(cfg, &mut rng);
    let stream = tokens(cfg.vocab, cfg.block);
    let repeats = 5usize;

    println!(
        "kv_cache_protocol | schema=1 repeats={} vocab={} width={} heads={} layers={} block={} ff={} comparison=growing_prefix_decode",
        repeats,
        cfg.vocab,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_layer,
        cfg.block,
        cfg.n_ff,
    );

    for context in [8usize, 16, 32, 64] {
        let row = measure(&gpt, &stream, context, repeats);
        let speedup = row.uncached_ns as f64 / row.cached_ns as f64;
        println!(
            "kv_cache_curve | context={} uncached_total_ns={} cached_total_ns={} uncached_ns_per_token={:.3} cached_ns_per_token={:.3} speedup={:.4} logical_bytes={} allocated_bytes={} params={} exact=true",
            row.context,
            row.uncached_ns,
            row.cached_ns,
            row.uncached_ns as f64 / row.context as f64,
            row.cached_ns as f64 / row.context as f64,
            speedup,
            row.logical_bytes,
            row.allocated_bytes,
            row.params,
        );
    }
}
