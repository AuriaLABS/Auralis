use auralis::memory::{
    ExternalMemory, InMemoryExternalMemory, MemoryClass, MemoryQuery, MemoryWrite,
};
use auralis::memory_integration::MemoryInferenceMode;
use auralis::model::{Config, Gpt};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::time::Instant;

fn controlled_model() -> Gpt {
    let cfg = Config {
        vocab: 4,
        n_embd: 4,
        n_head: 1,
        n_layer: 1,
        block: 4,
        n_ff: 8,
    };
    let mut rng = StdRng::seed_from_u64(42);
    let mut gpt = Gpt::new(cfg, &mut rng);
    let mut params = gpt.collect_params();
    params.fill(0.0);

    let w_out_len = cfg.n_embd * cfg.vocab;
    let w_out_start = params.len() - cfg.vocab - w_out_len;
    for i in 0..cfg.vocab.min(cfg.n_embd) {
        params[w_out_start + i * cfg.vocab + i] = 1.0;
    }
    gpt.write_params(&params);
    gpt
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
}

fn last_prediction(logits: &[f32], vocab: usize) -> usize {
    argmax(&logits[logits.len() - vocab..])
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let iterations = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100usize);
    let repeats = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7usize);
    assert!(iterations > 0 && repeats >= 3);

    let gpt = controlled_model();
    let input = [0usize, 0];
    let baseline = gpt.logits(&input);

    let (off, off_trace) = gpt
        .logits_with_memory(&input, None, &MemoryInferenceMode::off())
        .expect("memory off");
    assert_eq!(off, baseline);
    assert!(!off_trace.enabled);

    let empty = InMemoryExternalMemory::new(8).unwrap();
    let empty_mode = MemoryInferenceMode::on(MemoryQuery::exact("facts", "missing"), 1.0);
    let (empty_logits, empty_trace) = gpt
        .logits_with_memory(&input, Some(&empty), &empty_mode)
        .expect("empty memory");
    assert_eq!(empty_logits, baseline);
    assert_eq!(empty_trace.hits, 0);

    let mut memory = InMemoryExternalMemory::new(8).unwrap();
    for target in 1usize..=3 {
        let mut value = vec![0.0f32; gpt.cfg.n_embd];
        value[target] = 1.0;
        memory
            .write(MemoryWrite::new(
                "facts",
                format!("target-{target}"),
                value,
                MemoryClass::Persistent,
            ))
            .unwrap();
    }
    let before_len = memory.len();
    let before_fingerprint = memory.snapshot_fingerprint().unwrap();

    let baseline_prediction = last_prediction(&baseline, gpt.cfg.vocab);
    let mut off_correct = 0usize;
    let mut on_correct = 0usize;
    let mut last_trace = None;

    for target in 1usize..=3 {
        off_correct += usize::from(baseline_prediction == target);
        let mode = MemoryInferenceMode::on(
            MemoryQuery::exact("facts", format!("target-{target}")),
            1.0,
        );
        let (logits, trace) = gpt
            .logits_with_memory(&input, Some(&memory), &mode)
            .expect("memory on");
        let prediction = last_prediction(&logits, gpt.cfg.vocab);
        on_correct += usize::from(prediction == target);
        assert_eq!(trace.hits, 1);
        assert_eq!(trace.fused_width, gpt.cfg.n_embd);
        last_trace = Some(trace);
    }

    let after_fingerprint = memory.snapshot_fingerprint().unwrap();
    assert_eq!(memory.len(), before_len);
    assert_eq!(after_fingerprint, before_fingerprint);
    assert_eq!(off_correct, 0);
    assert_eq!(on_correct, 3);

    println!(
        "memory_model_task | cases=3 baseline_prediction={} off_correct={} on_correct={} off_accuracy={:.6} on_accuracy={:.6} off_exact=true empty_exact=true writes_unchanged=true",
        baseline_prediction,
        off_correct,
        on_correct,
        off_correct as f64 / 3.0,
        on_correct as f64 / 3.0,
    );
    let trace = last_trace.unwrap();
    println!("{}", trace.human());
    println!("memory_integration_json | {}", trace.json());

    let mode = MemoryInferenceMode::on(MemoryQuery::exact("facts", "target-1"), 1.0);
    let mut off_raw = Vec::with_capacity(repeats);
    let mut on_raw = Vec::with_capacity(repeats);
    for rep in 0..repeats {
        let off_first = rep % 2 == 0;

        let run_off = || {
            let started = Instant::now();
            for _ in 0..iterations {
                let (logits, trace) = gpt
                    .logits_with_memory(&input, None, &MemoryInferenceMode::off())
                    .unwrap();
                std::hint::black_box((&logits, trace));
            }
            started.elapsed().as_nanos().max(1) as u64
        };
        let run_on = || {
            let started = Instant::now();
            for _ in 0..iterations {
                let (logits, trace) = gpt
                    .logits_with_memory(&input, Some(&memory), &mode)
                    .unwrap();
                std::hint::black_box((&logits, trace));
            }
            started.elapsed().as_nanos().max(1) as u64
        };

        let (off_ns, on_ns) = if off_first {
            (run_off(), run_on())
        } else {
            let on_ns = run_on();
            let off_ns = run_off();
            (off_ns, on_ns)
        };
        off_raw.push(off_ns);
        on_raw.push(on_ns);
        println!(
            "memory_model_run | repetition={} first={} off_ns={} on_ns={} on_over_off={:.4}",
            rep + 1,
            if off_first { "off" } else { "on" },
            off_ns,
            on_ns,
            on_ns as f64 / off_ns as f64,
        );
    }

    let off_ns = median(&mut off_raw);
    let on_ns = median(&mut on_raw);
    let snapshot_bytes = memory.encode().unwrap().len();
    println!(
        "memory_model_cost | iterations={} repeats={} off_median_ns={} on_median_ns={} on_over_off={:.4} memory_records={} estimated_heap_bytes={} snapshot_bytes={} snapshot_fingerprint={:016x}",
        iterations,
        repeats,
        off_ns,
        on_ns,
        on_ns as f64 / off_ns as f64,
        memory.len(),
        memory.estimated_heap_bytes(),
        snapshot_bytes,
        memory.snapshot_fingerprint().unwrap(),
    );
}
