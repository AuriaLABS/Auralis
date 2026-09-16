use auralis::agent::Agent;
use auralis::bpe::BpeTokenizer;
use auralis::checkpoint;
use auralis::eval::evaluate_tokens_reference;
use auralis::experiment::{fingerprint_bytes, split_text, ExperimentIdentity, TokenSplit};
use auralis::gradcheck;
use auralis::manifest::{self, ExperimentManifest};
use auralis::metrics::Throughput;
use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use auralis::run_config::RunConfig;
use auralis::tokenizer::{AnyTok, CharTokenizer};
use auralis::training::train_step;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

fn load_corpus() -> String {
    let base = {
        let mut found = String::new();
        for p in ["data/corpus.txt", "auralis/data/corpus.txt"] {
            if let Ok(s) = fs::read_to_string(p) {
                if !s.trim().is_empty() {
                    found = s;
                    break;
                }
            }
        }
        if found.is_empty() {
            "Auralis es una inteligencia escrita en Rust.\n".repeat(20)
        } else {
            found
        }
    };
    let anchor = "Auralis es una inteligencia.\nAuralis recuerda el contexto.\nEl siguiente carácter importa.\n";
    format!("{base}\n{}", anchor.repeat(8))
}

fn sample_prompt(
    gpt: &Gpt,
    tok: &AnyTok,
    prompt: &str,
    n: usize,
    temp: f32,
    rng: &mut impl Rng,
) -> String {
    let mut out = tok.encode(prompt);
    if out.is_empty() {
        out = tok.encode("A");
    }
    gpt.generate(&mut out, n, temp, rng);
    tok.decode(&out)
}

fn new_model(train_text: &str, run: &RunConfig) -> (Gpt, AnyTok) {
    let tok = AnyTok::Bpe(BpeTokenizer::fit(train_text, run.bpe_merges));
    let mut rng = StdRng::seed_from_u64(run.seed);
    let gpt = Gpt::new(Config::tiny(tok.vocab_size()), &mut rng);
    (gpt, tok)
}

fn encode_split(tok: &AnyTok, raw: &auralis::experiment::TextSplit) -> TokenSplit {
    TokenSplit {
        train: tok.encode(&raw.train),
        validation: tok.encode(&raw.validation),
        test: tok.encode(&raw.test),
    }
}

fn report_holdout(name: &str, gpt: &Gpt, tokens: &[usize]) {
    match evaluate_tokens_reference(gpt, tokens) {
        Ok(m) => println!(
            "{name} | loss={:.4} ppl={:.3} predictions={} windows={}",
            m.mean_loss, m.perplexity, m.predicted_tokens, m.windows
        ),
        Err(e) => eprintln!("{name} no evaluable: {e}"),
    }
}

fn validate_resume_manifest(
    ckpt: &Path,
    gpt: &Gpt,
    tok: &AnyTok,
    adam: Option<&Adam>,
    run: &RunConfig,
    dataset_fingerprint: u64,
) -> Result<(), String> {
    let Some(adam) = adam else {
        return Err("checkpoint de reanudación sin estado Adam; no es una reanudación exacta".into());
    };
    let manifest_path = manifest::manifest_path(ckpt);
    if !manifest_path.exists() {
        eprintln!(
            "aviso: checkpoint legacy sin manifiesto; no se puede demostrar identidad exacta del experimento"
        );
        return Ok(());
    }

    let m = manifest::load_manifest(ckpt)?;
    m.validate_resume(gpt, tok, run, dataset_fingerprint)?;
    let optimizer_step = adam.t.max(0) as u64;
    if m.global_step != optimizer_step {
        return Err(format!(
            "resume mismatch for global_step: manifest={} adam={}",
            m.global_step, optimizer_step
        ));
    }
    println!(
        "manifest | revision={} global_step={} verified=true",
        m.code_revision, m.global_step
    );
    Ok(())
}

fn train(steps: usize, ckpt: &Path, fresh: bool, run: RunConfig) -> Result<(), String> {
    run.validate()
        .map_err(|e| format!("configuración inválida: {e}"))?;

    let text = load_corpus();
    let dataset_fingerprint = fingerprint_bytes(text.as_bytes());
    let raw_split = split_text(&text, run.split_config())
        .map_err(|e| format!("dataset inválido: {e}"))?;

    let (mut gpt, tok, saved_adam, resumed) = if !fresh && ckpt.exists() {
        let (g, t, a) = checkpoint::load_full(ckpt).map_err(|e| {
            format!(
                "checkpoint incompatible ({e}); abortado para evitar sobrescribir un experimento. Usa train-fresh para empezar de cero"
            )
        })?;
        println!("reanuda {} adam={}", ckpt.display(), a.is_some());
        (g, t, a, true)
    } else {
        let (g, t) = new_model(&raw_split.train, &run);
        (g, t, None, false)
    };

    let split = encode_split(&tok, &raw_split);
    let identity = ExperimentIdentity::from_raw(run.seed, &text, &split);
    let total_tokens = split.total_len();

    println!(
        "Auralis train | tok={} vocab={} tokens={} chars={} steps={} batch={} seed={} lr={} clip={} split={:.3}/{:.3}/{:.3} bpe_merges={}",
        tok.kind(),
        tok.vocab_size(),
        total_tokens,
        text.chars().count(),
        steps,
        run.batch_size,
        run.seed,
        run.learning_rate,
        run.grad_clip_norm,
        run.train_fraction,
        run.validation_fraction,
        1.0 - run.train_fraction - run.validation_fraction,
        run.bpe_merges,
    );
    println!("experiment | {}", identity.line());

    if resumed {
        validate_resume_manifest(
            ckpt,
            &gpt,
            &tok,
            saved_adam.as_ref(),
            &run,
            dataset_fingerprint,
        )?;
    }

    if split.train.len() <= gpt.cfg.block {
        return Err(format!(
            "split de entrenamiento demasiado corto para block={}",
            gpt.cfg.block
        ));
    }

    let n_params = gpt.collect_params().len();
    let mut grads = vec![0.0; n_params];
    let mut adam = if resumed {
        saved_adam.ok_or_else(|| "resume validado sin estado Adam".to_string())?
    } else {
        Adam::new(n_params, run.learning_rate)
    };
    if adam.export().2.len() != n_params {
        return Err("estado Adam incompatible con el número de parámetros".into());
    }

    let train_cfg = run.train_config();
    train_cfg
        .validate()
        .map_err(|e| format!("configuración de step inválida: {e}"))?;

    println!("params={} adam.t={}", n_params, adam.t);
    let t0 = Instant::now();
    let mut processed_tokens = 0u64;

    for local_step in 1..=steps {
        let global_step = adam.t.max(0) as u64;
        let metrics = train_step(
            &mut gpt,
            &mut adam,
            &split.train,
            train_cfg,
            global_step,
            &mut grads,
        )
        .map_err(|e| format!("entrenamiento abortado en step {global_step}: {e}"))?;
        processed_tokens += metrics.tokens as u64;

        if local_step == 1 || local_step % 20 == 0 || local_step == steps {
            let mut eval_rng = StdRng::seed_from_u64(
                run.seed ^ metrics.global_step.wrapping_mul(0x9e3779b97f4a7c15),
            );
            let sample = sample_prompt(&gpt, &tok, "Auralis es", 24, 0.2, &mut eval_rng);
            println!(
                "step {:4} global={} loss {:.4} grad {:.4} clip {:.3} sample={}",
                local_step,
                metrics.global_step,
                metrics.loss,
                metrics.grad_norm_before_clip,
                metrics.grad_scale,
                sample
            );
        }
    }

    let throughput = Throughput::from_duration(processed_tokens, t0.elapsed());
    println!(
        "train | tiempo={:.3}s tokens={} {:.0} tok/s adam.t={}",
        throughput.elapsed_seconds,
        throughput.tokens,
        throughput.tokens_per_second,
        adam.t
    );

    report_holdout("validation", &gpt, &split.validation);
    report_holdout("test", &gpt, &split.test);

    checkpoint::save_full(ckpt, &gpt, &tok, Some(&adam))
        .map_err(|e| format!("no se pudo guardar {ckpt:?}: {e}"))?;

    let manifest = ExperimentManifest::capture(
        &gpt,
        &tok,
        &run,
        dataset_fingerprint,
        adam.t.max(0) as u64,
    );
    let manifest_path = manifest::save_manifest(ckpt, &manifest)
        .map_err(|e| format!("checkpoint guardado pero falló el manifiesto: {e}"))?;
    println!(
        "checkpoint → {} | manifest → {}",
        ckpt.display(),
        manifest_path.display()
    );
    Ok(())
}

fn run_train_or_exit(steps: usize, ckpt: &Path, fresh: bool, run: RunConfig) {
    if let Err(e) = train(steps, ckpt, fresh, run) {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

fn eval_ckpt(ckpt: &Path) {
    let (gpt, tok) = match checkpoint::load(ckpt) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("carga {ckpt:?}: {e}");
            return;
        }
    };
    let mut rng = rand::thread_rng();
    for prompt in ["Auralis es", "Hola", "El gato", "Pregunta"] {
        println!("-- {prompt:?} --");
        for temp in [0.3, 0.6] {
            let s = sample_prompt(&gpt, &tok, prompt, 36, temp, &mut rng);
            println!("T={temp:.1}  {s}");
        }
    }
}

fn chat(ckpt: &Path) {
    let (gpt, tok) = match checkpoint::load(ckpt) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("carga {ckpt:?}: {e}\nEntrena antes: auralis train 80");
            return;
        }
    };
    let mut agent = Agent::new();
    let mut rng = rand::thread_rng();
    println!("Auralis chat  (escribe /salir)");
    let stdin = io::stdin();
    loop {
        print!("tú> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/salir" || line == "/quit" {
            break;
        }
        let reply = agent.reply(line, &gpt, &tok, &mut rng, 60);
        println!("auralis> {reply}");
    }
}

fn run_check() {
    let text = load_corpus();
    let tok = CharTokenizer::fit(&text);
    let ids = tok.encode(&text);
    let mut rng = StdRng::seed_from_u64(RunConfig::default().seed);
    let cfg = gradcheck::tiny_check_config(tok.vocab_size());
    let block = cfg.block.min(ids.len().saturating_sub(2).max(2));
    let gpt = Gpt::new(Config { block, ..cfg }, &mut rng);
    let x = &ids[0..block];
    let y = &ids[1..block + 1];
    println!(
        "gradcheck | params={} block={} vocab={}",
        gpt.collect_params().len(),
        block,
        tok.vocab_size()
    );
    let report = gradcheck::check_random_params(&gpt, x, y, 24, 1e-3, &mut rng);
    let ok = report.ok_with(1e-3, 0.25);
    println!(
        "checked={} max_abs={:.4e} max_rel={:.4} mean_rel={:.4} ok={}",
        report.checked, report.max_abs_err, report.max_rel_err, report.mean_rel_err, ok
    );
    if !ok {
        std::process::exit(2);
    }
}

fn run_bpe() {
    let text = load_corpus();
    let bpe = BpeTokenizer::fit(&text, 80);
    let ids = bpe.encode(&text);
    println!(
        "bpe | merges={} vocab={} chars={} tokens={}",
        bpe.merges.len(),
        bpe.vocab_size(),
        text.chars().count(),
        ids.len()
    );
    for (i, (a, b)) in bpe.merges.iter().take(8).enumerate() {
        println!("  merge {i}: {a:?} + {b:?} -> {:?}", format!("{a}{b}"));
    }
    let sample = text.chars().take(80).collect::<String>();
    println!("roundtrip: {}", bpe.decode(&bpe.encode(&sample)) == sample);
}

fn parse_train_args(args: &[String]) -> (usize, &Path, RunConfig) {
    let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(80);
    let ckpt = args
        .get(3)
        .map(Path::new)
        .unwrap_or_else(|| Path::new("auralis.bin"));
    let defaults = RunConfig::default();
    let seed = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(defaults.seed);
    let batch = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(defaults.batch_size);
    (steps, ckpt, defaults.with_cli_overrides(seed, batch))
}

fn usage() {
    eprintln!(
        "Auralis\n  auralis train [steps] [checkpoint] [seed] [batch]\n  auralis train-fresh [steps] [checkpoint] [seed] [batch]\n  auralis eval [checkpoint]\n  auralis chat [checkpoint]\n  auralis check\n  auralis bpe"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let ckpt_default = Path::new("auralis.bin");
    match args.get(1).map(|s| s.as_str()) {
        Some("train") => {
            let (steps, ckpt, run) = parse_train_args(&args);
            run_train_or_exit(steps, ckpt, false, run);
        }
        Some("train-fresh") => {
            let (steps, ckpt, run) = parse_train_args(&args);
            run_train_or_exit(steps, ckpt, true, run);
        }
        Some("chat") => chat(args.get(2).map(Path::new).unwrap_or(ckpt_default)),
        Some("check") => run_check(),
        Some("bpe") => run_bpe(),
        Some("eval") => eval_ckpt(args.get(2).map(Path::new).unwrap_or(ckpt_default)),
        Some("-h" | "--help" | "help") => usage(),
        Some(n) if n.chars().all(|c| c.is_ascii_digit()) => run_train_or_exit(
            n.parse().unwrap(),
            ckpt_default,
            false,
            RunConfig::default(),
        ),
        None => run_train_or_exit(80, ckpt_default, false, RunConfig::default()),
        _ => usage(),
    }
}
