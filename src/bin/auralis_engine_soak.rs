use auralis::checkpoint;
use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use auralis::tokenizer::{AnyTok, CharTokenizer};
use auralis::training::{train_step_reuse, TrainConfig, TrainWorkspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone)]
struct WindowSample {
    tok_per_s: f64,
    rss_kib: u64,
}

fn model() -> Gpt {
    let mut rng = StdRng::seed_from_u64(0xA11CE_224);
    Gpt::new(Config::tiny(100), &mut rng)
}

fn train_config() -> TrainConfig {
    TrainConfig {
        seed: 659_918,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: 1.0,
    }
}

fn token_stream() -> Vec<usize> {
    (0..8192)
        .map(|i| (i * 37 + i / 7 + 11) % 100)
        .collect()
}

fn tokenizer(vocab: usize) -> AnyTok {
    let itos: Vec<char> = (0..vocab)
        .map(|i| char::from_u32(0x1000 + i as u32).expect("valid synthetic char"))
        .collect();
    let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    AnyTok::Char(CharTokenizer { stoi, itos })
}

fn proc_kib(field: &str) -> u64 {
    let Ok(text) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            return rest
                .split_whitespace()
                .find_map(|part| part.parse::<u64>().ok())
                .unwrap_or(0);
        }
    }
    0
}

fn current_rss_kib() -> u64 {
    proc_kib("VmRSS:")
}

fn peak_rss_kib() -> u64 {
    proc_kib("VmHWM:")
}

fn percentile(values: &[f64], p: f64) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[index]
}

fn median(values: &[f64]) -> f64 {
    percentile(values, 0.5)
}

fn fnv_mix(mut h: u64, value: u64) -> u64 {
    h ^= value;
    h.wrapping_mul(0x100000001b3)
}

fn state_fingerprint(gpt: &Gpt, adam: &Adam) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for value in gpt.collect_params() {
        h = fnv_mix(h, value.to_bits() as u64);
    }
    let (lr, t, m, v) = adam.export();
    h = fnv_mix(h, lr.to_bits() as u64);
    h = fnv_mix(h, t as u32 as u64);
    for &value in m {
        h = fnv_mix(h, value.to_bits() as u64);
    }
    for &value in v {
        h = fnv_mix(h, value.to_bits() as u64);
    }
    h
}

fn config_fingerprint(steps: usize, window_steps: usize) -> u64 {
    let cfg = Config::tiny(100);
    let train = train_config();
    let values = [
        cfg.vocab as u64,
        cfg.n_embd as u64,
        cfg.n_head as u64,
        cfg.n_layer as u64,
        cfg.block as u64,
        cfg.n_ff as u64,
        train.seed,
        train.batch_size as u64,
        train.gradient_accumulation_steps as u64,
        train.grad_clip_norm.to_bits() as u64,
        steps as u64,
        window_steps as u64,
    ];
    values.into_iter().fold(0xcbf29ce484222325u64, fnv_mix)
}

fn hardware() -> String {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some((key, value)) = line.split_once(':') {
                if key.trim() == "model name" {
                    let value = value.trim();
                    if !value.is_empty() {
                        return value.to_string();
                    }
                }
            }
        }
    }
    format!("{}-{}", env::consts::ARCH, env::consts::OS)
}

fn toolchain() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "rustc-unknown".to_string())
}

fn execute_windows(
    gpt: &mut Gpt,
    adam: &mut Adam,
    tokens: &[usize],
    cfg: TrainConfig,
    grads: &mut [f32],
    workspace: &mut TrainWorkspace,
    steps: usize,
    window_steps: usize,
    windows: &mut Vec<WindowSample>,
) -> f32 {
    let mut completed = 0usize;
    let mut last_loss = f32::NAN;
    while completed < steps {
        let chunk = window_steps.min(steps - completed);
        let started = Instant::now();
        let mut chunk_tokens = 0u64;
        for _ in 0..chunk {
            let global_step = adam.t.max(0) as u64;
            let metrics = train_step_reuse(
                gpt,
                adam,
                tokens,
                cfg,
                global_step,
                grads,
                workspace,
            )
            .expect("soak training step");
            chunk_tokens += metrics.tokens as u64;
            last_loss = metrics.loss;
        }
        let seconds = started.elapsed().as_secs_f64();
        let tok_per_s = if seconds > 0.0 {
            chunk_tokens as f64 / seconds
        } else {
            f64::INFINITY
        };
        windows.push(WindowSample {
            tok_per_s,
            rss_kib: current_rss_kib(),
        });
        completed += chunk;
    }
    last_loss
}

fn child_run(
    mode: &str,
    steps: usize,
    window_steps: usize,
    split_at: usize,
    checkpoint_path: &Path,
) {
    let cfg = train_config();
    let tokens = token_stream();
    let mut tok = tokenizer(100);
    let mut gpt = model();
    let n_params = gpt.collect_params().len();
    let mut adam = Adam::new(n_params, 3e-3);
    let mut grads = vec![0.0f32; n_params];
    let mut workspace = TrainWorkspace::new(&gpt);
    let initial_rss = current_rss_kib();
    let mut windows = Vec::new();

    let started = Instant::now();
    let final_loss = match mode {
        "continuous" => execute_windows(
            &mut gpt,
            &mut adam,
            &tokens,
            cfg,
            &mut grads,
            &mut workspace,
            steps,
            window_steps,
            &mut windows,
        ),
        "split" => {
            let first = split_at;
            let second = steps - first;
            let first_loss = execute_windows(
                &mut gpt,
                &mut adam,
                &tokens,
                cfg,
                &mut grads,
                &mut workspace,
                first,
                window_steps,
                &mut windows,
            );
            let mid = checkpoint_path.with_extension("mid.bin");
            checkpoint::save_full(&mid, &gpt, &tok, Some(&adam))
                .expect("save split midpoint checkpoint");
            let (loaded_gpt, loaded_tok, loaded_adam) =
                checkpoint::load_full(&mid).expect("load split midpoint checkpoint");
            let _ = fs::remove_file(&mid);

            gpt = loaded_gpt;
            tok = loaded_tok;
            adam = loaded_adam.expect("split checkpoint must contain Adam");
            grads = vec![0.0f32; gpt.collect_params().len()];
            workspace = TrainWorkspace::new(&gpt);

            if second == 0 {
                first_loss
            } else {
                execute_windows(
                    &mut gpt,
                    &mut adam,
                    &tokens,
                    cfg,
                    &mut grads,
                    &mut workspace,
                    second,
                    window_steps,
                    &mut windows,
                )
            }
        }
        other => panic!("unknown child mode {other}"),
    };
    let total_seconds = started.elapsed().as_secs_f64();

    checkpoint::save_full(checkpoint_path, &gpt, &tok, Some(&adam))
        .expect("save final soak checkpoint");

    let rates: Vec<f64> = windows.iter().map(|w| w.tok_per_s).collect();
    let first_tps = *rates.first().expect("at least one window");
    let last_tps = *rates.last().expect("at least one window");
    let median_tps = median(&rates);
    let p25_tps = percentile(&rates, 0.25);
    let p75_tps = percentile(&rates, 0.75);
    let last_first_ratio = last_tps / first_tps;
    let final_rss = current_rss_kib();
    let peak_rss = peak_rss_kib();
    let max_window_rss = windows.iter().map(|w| w.rss_kib).max().unwrap_or(final_rss);
    let rss_bounded = initial_rss == 0
        || (final_rss <= initial_rss.saturating_add(8192)
            && peak_rss <= initial_rss.saturating_add(16384));
    let checkpoint_bytes = fs::metadata(checkpoint_path)
        .expect("final checkpoint metadata")
        .len();
    let fingerprint = state_fingerprint(&gpt, &adam);
    let finite_loss = final_loss.is_finite();
    let tokens_per_step = cfg
        .effective_batch_size()
        .expect("effective batch")
        .checked_mul(gpt.cfg.block)
        .expect("tokens per step");
    let total_tokens = tokens_per_step
        .checked_mul(steps)
        .expect("total soak tokens");
    let ms_per_step = 1000.0 * total_seconds / steps as f64;

    println!(
        concat!(
            "engine_soak_child | mode={} steps={} window_steps={} windows={} tokens={} ",
            "total_seconds={:.6} ms_per_step={:.6} first_tok_per_s={:.3} last_tok_per_s={:.3} ",
            "median_tok_per_s={:.3} p25_tok_per_s={:.3} p75_tok_per_s={:.3} ",
            "last_first_ratio={:.4} initial_rss_kib={} final_rss_kib={} ",
            "peak_rss_kib={} max_window_rss_kib={} rss_bounded={} ",
            "final_loss={:.9} finite_loss={} adam_t={} checkpoint_bytes={} state_fingerprint={:016x}"
        ),
        mode,
        steps,
        window_steps,
        windows.len(),
        total_tokens,
        total_seconds,
        ms_per_step,
        first_tps,
        last_tps,
        median_tps,
        p25_tps,
        p75_tps,
        last_first_ratio,
        initial_rss,
        final_rss,
        peak_rss,
        max_window_rss,
        rss_bounded,
        final_loss,
        finite_loss,
        adam.t,
        checkpoint_bytes,
        fingerprint,
    );
}

fn parse_record(text: &str) -> HashMap<String, String> {
    let line = text
        .lines()
        .find(|line| line.starts_with("engine_soak_child | "))
        .expect("child emitted engine_soak_child record");
    line.split_once('|')
        .expect("child record separator")
        .1
        .split_whitespace()
        .filter_map(|part| part.split_once('='))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn value_f64(record: &HashMap<String, String>, key: &str) -> f64 {
    record
        .get(key)
        .unwrap_or_else(|| panic!("missing {key}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid {key}"))
}

fn value_bool(record: &HashMap<String, String>, key: &str) -> bool {
    record
        .get(key)
        .unwrap_or_else(|| panic!("missing {key}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid {key}"))
}

fn exact_final_state(a: &Path, b: &Path) -> bool {
    let (gpt_a, tok_a, adam_a) = checkpoint::load_full(a).expect("load continuous checkpoint");
    let (gpt_b, tok_b, adam_b) = checkpoint::load_full(b).expect("load split checkpoint");
    let Some(adam_a) = adam_a else { return false };
    let Some(adam_b) = adam_b else { return false };
    let (lr_a, t_a, m_a, v_a) = adam_a.export();
    let (lr_b, t_b, m_b, v_b) = adam_b.export();
    gpt_a.collect_params() == gpt_b.collect_params()
        && tok_a.vocab_size() == tok_b.vocab_size()
        && tok_a.kind() == tok_b.kind()
        && lr_a == lr_b
        && t_a == t_b
        && m_a == m_b
        && v_a == v_b
}

fn run_child_process(
    exe: &Path,
    mode: &str,
    steps: usize,
    window_steps: usize,
    split_at: usize,
    checkpoint: &Path,
) -> String {
    let output = Command::new(exe)
        .arg("--child")
        .arg(mode)
        .arg(steps.to_string())
        .arg(window_steps.to_string())
        .arg(split_at.to_string())
        .arg(checkpoint)
        .output()
        .expect("spawn soak child");
    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stdout));
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        panic!("soak child {mode} failed");
    }
    String::from_utf8(output.stdout).expect("child stdout utf8")
}

fn parent_run(steps: usize, window_steps: usize) {
    assert!(steps >= 2, "soak steps must be at least 2");
    assert!(window_steps > 0, "window_steps must be positive");
    let split_at = steps / 2;
    let pid = std::process::id();
    let temp = env::temp_dir();
    let continuous_path = temp.join(format!("auralis-soak-{pid}-continuous.bin"));
    let split_path = temp.join(format!("auralis-soak-{pid}-split.bin"));
    let exe = env::current_exe().expect("current soak executable");

    println!(
        "engine_soak_protocol | steps={} window_steps={} split_at={} config_fingerprint={:016x} hardware={:?} toolchain={:?} revision={}",
        steps,
        window_steps,
        split_at,
        config_fingerprint(steps, window_steps),
        hardware(),
        toolchain(),
        env::var("AURALIS_COMMIT_SHA").unwrap_or_else(|_| "unknown".into()),
    );

    let continuous = run_child_process(
        &exe,
        "continuous",
        steps,
        window_steps,
        split_at,
        &continuous_path,
    );
    let split = run_child_process(
        &exe,
        "split",
        steps,
        window_steps,
        split_at,
        &split_path,
    );
    print!("{continuous}");
    print!("{split}");

    let c = parse_record(&continuous);
    let s = parse_record(&split);
    let exact_state = exact_final_state(&continuous_path, &split_path);
    let checkpoint_bytes_exact =
        fs::read(&continuous_path).expect("read continuous checkpoint")
            == fs::read(&split_path).expect("read split checkpoint");
    let finite_loss = value_bool(&c, "finite_loss") && value_bool(&s, "finite_loss");
    let continuous_rss_bounded = value_bool(&c, "rss_bounded");
    let split_rss_bounded = value_bool(&s, "rss_bounded");
    let continuous_ratio = value_f64(&c, "last_first_ratio");
    let split_ratio = value_f64(&s, "last_first_ratio");

    println!(
        concat!(
            "engine_soak_summary | steps={} window_steps={} split_at={} ",
            "exact_state={} checkpoint_bytes_exact={} finite_loss={} ",
            "continuous_last_first_ratio={:.4} split_last_first_ratio={:.4} ",
            "continuous_rss_bounded={} split_rss_bounded={} ",
            "continuous_fingerprint={} split_fingerprint={}"
        ),
        steps,
        window_steps,
        split_at,
        exact_state,
        checkpoint_bytes_exact,
        finite_loss,
        continuous_ratio,
        split_ratio,
        continuous_rss_bounded,
        split_rss_bounded,
        c.get("state_fingerprint").expect("continuous fingerprint"),
        s.get("state_fingerprint").expect("split fingerprint"),
    );

    let _ = fs::remove_file(&continuous_path);
    let _ = fs::remove_file(&split_path);

    if !exact_state
        || !checkpoint_bytes_exact
        || !finite_loss
        || !continuous_rss_bounded
        || !split_rss_bounded
        || continuous_ratio < 0.85
        || split_ratio < 0.85
    {
        std::process::exit(3);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--child") {
        let mode = args.get(2).expect("child mode");
        let steps = args.get(3).and_then(|s| s.parse().ok()).expect("child steps");
        let window_steps = args
            .get(4)
            .and_then(|s| s.parse().ok())
            .expect("child window_steps");
        let split_at = args
            .get(5)
            .and_then(|s| s.parse().ok())
            .expect("child split_at");
        let checkpoint = PathBuf::from(args.get(6).expect("child checkpoint"));
        child_run(mode, steps, window_steps, split_at, &checkpoint);
        return;
    }

    let steps = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300usize);
    let window_steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50usize);
    parent_run(steps, window_steps);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_is_deterministic() {
        let values = [4.0, 1.0, 3.0, 2.0, 5.0];
        assert_eq!(percentile(&values, 0.25), 2.0);
        assert_eq!(median(&values), 3.0);
        assert_eq!(percentile(&values, 0.75), 4.0);
    }

    #[test]
    fn fingerprints_change_with_protocol() {
        assert_ne!(config_fingerprint(300, 50), config_fingerprint(301, 50));
        assert_ne!(config_fingerprint(300, 50), config_fingerprint(300, 25));
    }

    #[test]
    fn synthetic_tokenizer_matches_model_vocab() {
        assert_eq!(tokenizer(100).vocab_size(), 100);
    }
}
