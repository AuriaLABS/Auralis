use auralis::optim::{Adam, AdamW, Lion, Optimizer};
use std::env;
use std::time::Instant;

fn params(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + i / 7 + 11) % 211) as f32 - 105.0) / 53.0)
        .collect()
}

fn grads(n: usize, step: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 29 + step * 17 + i / 5 + 3) % 197) as f32;
            (raw - 98.0) / 4096.0
        })
        .collect()
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn run_one(
    label: &str,
    make: impl Fn() -> Box<dyn Optimizer>,
    parameter_count: usize,
    state_bytes: usize,
    warmup: usize,
    iterations: usize,
    repeats: usize,
) {
    let mut raw = Vec::with_capacity(repeats);
    let mut fingerprints = Vec::with_capacity(repeats);

    for rep in 0..repeats {
        let mut optimizer = make();
        let mut p = params(parameter_count);

        for step in 0..warmup {
            let g = grads(parameter_count, step);
            optimizer.update(&mut p, &g).unwrap();
        }

        let started = Instant::now();
        for step in 0..iterations {
            let g = grads(parameter_count, warmup + step);
            optimizer.update(&mut p, &g).unwrap();
        }
        let ns = started.elapsed().as_nanos().max(1) as u64;
        raw.push(ns);
        fingerprints.push(optimizer.state_fingerprint());
        println!(
            "optimizer_variant_run | optimizer={} repetition={} total_ns={} ns_per_update={:.3} state_bytes={} global_step={} state_fingerprint={:016x}",
            label,
            rep + 1,
            ns,
            ns as f64 / iterations as f64,
            state_bytes,
            optimizer.global_step(),
            optimizer.state_fingerprint(),
        );
    }

    assert!(
        fingerprints.windows(2).all(|pair| pair[0] == pair[1]),
        "{label} must be deterministic across repetitions"
    );
    let med = median(&mut raw);
    println!(
        "optimizer_variant_summary | optimizer={} parameters={} warmup={} iterations={} repeats={} median_ns_per_update={:.3} state_bytes={} deterministic=true",
        label,
        parameter_count,
        warmup,
        iterations,
        repeats,
        med as f64 / iterations as f64,
        state_bytes,
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let parameter_count = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(28_580usize);
    let warmup = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5usize);
    let iterations = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(50usize);
    let repeats = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(5usize);
    assert!(parameter_count > 0 && iterations > 0 && repeats >= 3);

    let lr = 3e-3;
    run_one(
        "adam",
        || Box::new(Adam::new(parameter_count, lr)),
        parameter_count,
        parameter_count * 8,
        warmup,
        iterations,
        repeats,
    );
    run_one(
        "adamw",
        || Box::new(AdamW::new(parameter_count, lr, 0.01)),
        parameter_count,
        parameter_count * 8,
        warmup,
        iterations,
        repeats,
    );
    run_one(
        "lion",
        || Box::new(Lion::new(parameter_count, lr, 0.01)),
        parameter_count,
        parameter_count * 4,
        warmup,
        iterations,
        repeats,
    );
}
