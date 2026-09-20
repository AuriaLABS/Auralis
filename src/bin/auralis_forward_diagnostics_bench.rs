
use auralis::model::{Config, Gpt};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::hint::black_box;
use std::time::Instant;

fn model() -> Gpt {
    let mut rng = StdRng::seed_from_u64(0xA11CE_222);
    Gpt::new(Config::tiny(100), &mut rng)
}

fn tokens() -> Vec<usize> {
    (0..32).map(|i| (i * 13 + 7) % 100).collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn assert_equivalent(gpt: &Gpt, tokens: &[usize]) {
    let baseline = gpt.logits(tokens);
    let (diagnosed, report) = gpt.forward_diagnostics(tokens).expect("clean diagnostics");
    assert_eq!(diagnosed, baseline);
    assert!(report.first_fault().is_none());
    assert_eq!(report.tensors.len(), gpt.cfg.n_layer * 9 + 2);
}

fn time_baseline(gpt: &Gpt, tokens: &[usize], calls: usize, repeats: usize) -> f64 {
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        for _ in 0..calls {
            black_box(gpt.logits(black_box(tokens)));
        }
        samples.push(started.elapsed().as_secs_f64() * 1e9 / calls as f64);
    }
    median(samples)
}

fn time_diagnostics(gpt: &Gpt, tokens: &[usize], calls: usize, repeats: usize) -> f64 {
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        for _ in 0..calls {
            let (logits, report) = gpt
                .forward_diagnostics(black_box(tokens))
                .expect("clean diagnostics");
            black_box(logits);
            black_box(report);
        }
        samples.push(started.elapsed().as_secs_f64() * 1e9 / calls as f64);
    }
    median(samples)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let calls = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12usize);
    let repeats = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    assert!(calls > 0 && repeats >= 3);

    let gpt = model();
    let tokens = tokens();
    assert_equivalent(&gpt, &tokens);

    let baseline = time_baseline(&gpt, &tokens, calls, repeats);
    let diagnostics = time_diagnostics(&gpt, &tokens, calls, repeats);
    println!(
        "forward_diagnostics_bench | calls={} repeats={} baseline_ns_per_call={:.3} diagnostics_ns_per_call={:.3} diagnostics_over_baseline={:.4} summaries={}",
        calls,
        repeats,
        baseline,
        diagnostics,
        diagnostics / baseline,
        gpt.cfg.n_layer * 9 + 2
    );
}
