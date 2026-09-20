use auralis::eval::evaluate_tokens_reference;
use auralis::model::{Config, Gpt};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::process::Command;
use std::time::Instant;

fn peak_rss_kib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                return rest.split_whitespace().find_map(|v| v.parse().ok());
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn tokens(n: usize, vocab: usize) -> Vec<usize> {
    (0..n).map(|i| (i * 37 + i / 7 + 11) % vocab).collect()
}

fn child(context: usize, iterations: usize, repeats: usize) {
    let cfg = Config::tiny(100);
    assert!(context > 0 && context <= cfg.block);
    let mut rng = StdRng::seed_from_u64(659_918);
    let gpt = Gpt::new(cfg, &mut rng);
    let input = tokens(context, cfg.vocab);
    let eval_stream = tokens(context + 1, cfg.vocab);

    for _ in 0..5 {
        let logits = gpt.logits(&input);
        std::hint::black_box(logits);
    }

    let mut raw = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        for _ in 0..iterations {
            let logits = gpt.logits(&input);
            std::hint::black_box(logits);
        }
        raw.push(started.elapsed().as_nanos().max(1) as u64);
    }
    let median_ns = median(&mut raw);
    let seconds = median_ns as f64 / 1_000_000_000.0;
    let tok_per_s = (context * iterations) as f64 / seconds;
    let eval = evaluate_tokens_reference(&gpt, &eval_stream).expect("context eval");
    let rss = peak_rss_kib()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".into());

    println!(
        "position_context_child | context={} block={} iterations={} repeats={} median_ns={} tok_per_s={:.3} peak_rss_kib={} input_bytes={} logits_bytes={} position_capacity_bytes={} eval_loss={:.6} ppl={:.6}",
        context,
        cfg.block,
        iterations,
        repeats,
        median_ns,
        tok_per_s,
        rss,
        context * cfg.n_embd * 4,
        context * cfg.vocab * 4,
        cfg.block * cfg.n_embd * 4,
        eval.mean_loss,
        eval.perplexity,
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--child") {
        let context: usize = args.get(2).and_then(|v| v.parse().ok()).expect("context");
        let iterations: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(30);
        let repeats: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(5);
        child(context, iterations, repeats);
        return;
    }

    let iterations = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(30usize);
    let repeats = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5usize);
    assert!(iterations > 0 && repeats >= 3);
    let exe = env::current_exe().expect("current exe");

    println!(
        "position_context_protocol | model_block=32 n_embd=32 n_head=4 n_layer=2 n_ff=96 iterations={} repeats={} seed=659918",
        iterations, repeats
    );
    for context in [4usize, 8, 16, 32] {
        let out = Command::new(&exe)
            .args([
                "--child",
                &context.to_string(),
                &iterations.to_string(),
                &repeats.to_string(),
            ])
            .output()
            .expect("position context child");
        if !out.status.success() {
            eprint!("{}", String::from_utf8_lossy(&out.stderr));
            std::process::exit(out.status.code().unwrap_or(2));
        }
        print!("{}", String::from_utf8_lossy(&out.stdout));
    }
}
