use auralis::model::{Config, Gpt, NormalizationKind};
use auralis::optim::{Adam, Optimizer};
use auralis::position::PositionKind;
use auralis::training::{train_step_reuse, TrainConfig, TrainWorkspace};
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

fn stream(count: usize, vocab: usize) -> Vec<usize> {
    (0..count)
        .map(|i| (i * 37 + i / 7 + 11) % vocab)
        .collect()
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn parse_policy(value: &str) -> PositionKind {
    match value {
        "learned_absolute" => PositionKind::LearnedAbsolute,
        "alibi" => PositionKind::Alibi,
        other => panic!("unknown policy {other}"),
    }
}

fn child(
    policy: PositionKind,
    context: usize,
    train_steps: usize,
    iterations: usize,
    repeats: usize,
) {
    let cfg = Config {
        vocab: 8,
        n_embd: 8,
        n_head: 2,
        n_layer: 2,
        block: 32,
        n_ff: 16,
    };
    assert!(context > 0 && context <= cfg.block);

    let mut rng = StdRng::seed_from_u64(659_918);
    let mut gpt = Gpt::new_with_policies(cfg, NormalizationKind::LayerNorm, policy, &mut rng);
    let n = gpt.collect_params().len();
    let mut adam = Adam::new(n, 3e-3);
    let mut grads = vec![0.0f32; n];
    let mut workspace = TrainWorkspace::new(&gpt);
    let tokens = stream(2048, cfg.vocab);
    let train_cfg = TrainConfig {
        seed: 659_918,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: 1.0,
    };

    for _ in 0..train_steps {
        let step = adam.global_step();
        train_step_reuse(
            &mut gpt,
            &mut adam,
            &tokens,
            train_cfg,
            step,
            &mut grads,
            &mut workspace,
        )
        .expect("ALiBi context training");
    }

    let eval = stream(context + 1, cfg.vocab);
    let x = &eval[..context];
    let y = &eval[1..];
    let loss = gpt.loss(x, y);
    let ppl = loss.exp();

    for _ in 0..5 {
        std::hint::black_box(gpt.logits(x));
    }
    let mut raw = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(gpt.logits(x));
        }
        raw.push(started.elapsed().as_nanos().max(1) as u64);
    }
    let median_ns = median(&mut raw);
    let seconds = median_ns as f64 / 1_000_000_000.0;
    let tok_per_s = (context * iterations) as f64 / seconds;
    let rss = peak_rss_kib()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".into());

    println!(
        "alibi_context_child | policy={} context={} block={} train_steps={} optimizer_step={} iterations={} repeats={} median_ns={} tok_per_s={:.3} peak_rss_kib={} params={} loss={:.6} ppl={:.6}",
        policy.as_str(),
        context,
        cfg.block,
        train_steps,
        adam.global_step(),
        iterations,
        repeats,
        median_ns,
        tok_per_s,
        rss,
        n,
        loss,
        ppl,
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--child") {
        let policy = parse_policy(args.get(2).expect("policy"));
        let context: usize = args.get(3).and_then(|v| v.parse().ok()).expect("context");
        let train_steps: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(6);
        let iterations: usize = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(40);
        let repeats: usize = args.get(6).and_then(|v| v.parse().ok()).unwrap_or(5);
        child(policy, context, train_steps, iterations, repeats);
        return;
    }

    let train_steps = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(6usize);
    let iterations = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(40usize);
    let repeats = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(5usize);
    assert!(train_steps > 0 && iterations > 0 && repeats >= 3);

    println!(
        "alibi_context_protocol | seed=659918 vocab=8 n_embd=8 n_head=2 n_layer=2 block=32 n_ff=16 batch=2 accum=2 lr=0.003 clip=1 train_steps={} iterations={} repeats={}",
        train_steps, iterations, repeats
    );

    let exe = env::current_exe().expect("current exe");
    for context in [4usize, 8, 16, 32] {
        let order = if context % 16 == 0 {
            ["alibi", "learned_absolute"]
        } else {
            ["learned_absolute", "alibi"]
        };
        for policy in order {
            let output = Command::new(&exe)
                .args([
                    "--child",
                    policy,
                    &context.to_string(),
                    &train_steps.to_string(),
                    &iterations.to_string(),
                    &repeats.to_string(),
                ])
                .output()
                .expect("context child");
            if !output.status.success() {
                eprint!("{}", String::from_utf8_lossy(&output.stderr));
                std::process::exit(output.status.code().unwrap_or(2));
            }
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }
    }
}
