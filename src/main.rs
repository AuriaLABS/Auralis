use auralis::agent::Agent;
use auralis::architecture::ArchitectureConfig;
use auralis::bench;
use auralis::bench_format::{self, CatalogFormat};
use auralis::bench_runner::{self, BenchProtocol};
use auralis::bpe::BpeTokenizer;
use auralis::checkpoint;
use auralis::eval::{evaluate_tokens_reference, EvalMetrics};
use auralis::experiment::{fingerprint_bytes, split_text, ExperimentIdentity, TokenSplit};
use auralis::gradcheck;
use auralis::inspect::inspect_checkpoint;
use auralis::manifest::{self, ExperimentManifest};
use auralis::metrics::Throughput;
use auralis::model::{Config, Gpt};
use auralis::numeric::{self, Diagnostics};
use auralis::optim::Adam;
use auralis::release::{check_release, default_root, ReleaseManifest, DEFAULT_RELEASE_ARTIFACTS};
use auralis::run_config::RunConfig;
use auralis::scheduler::{scheduler_path, SchedulerConfig};
use auralis::sec::scan_tree;
use auralis::tokenizer::{AnyTok, CharTokenizer};
use auralis::training::{
    train_step_reuse, train_step_reuse_diagnostics, TrainWorkspace,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
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

fn new_model(
    train_text: &str,
    run: &RunConfig,
    architecture: ArchitectureConfig,
) -> Result<(Gpt, AnyTok), String> {
    let tok = AnyTok::Bpe(BpeTokenizer::fit(train_text, run.bpe_merges));
    let mut rng = StdRng::seed_from_u64(run.seed);
    let cfg = architecture.model_config(tok.vocab_size())?;
    let gpt = Gpt::new(cfg, &mut rng);
    Ok((gpt, tok))
}

fn encode_split(tok: &AnyTok, raw: &auralis::experiment::TextSplit) -> TokenSplit {
    TokenSplit {
        train: tok.encode(&raw.train),
        validation: tok.encode(&raw.validation),
        test: tok.encode(&raw.test),
    }
}

fn evaluate_holdout(name: &str, gpt: &Gpt, tokens: &[usize]) -> Result<EvalMetrics, String> {
    let m = evaluate_tokens_reference(gpt, tokens)
        .map_err(|e| format!("{name} no evaluable: {e}"))?;
    println!(
        "{name} | loss={:.4} ppl={:.3} predictions={} windows={}",
        m.mean_loss, m.perplexity, m.predicted_tokens, m.windows
    );
    Ok(m)
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
        return Err(format!(
            "checkpoint {} no tiene manifiesto Foundation; reanudación rechazada. Usa train-fresh para iniciar un experimento reproducible",
            ckpt.display()
        ));
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

fn resolve_scheduler(
    ckpt: &Path,
    fresh: bool,
    requested: Option<SchedulerConfig>,
) -> Result<(SchedulerConfig, bool), String> {
    let sidecar = scheduler_path(ckpt);
    if fresh || !ckpt.exists() {
        return Ok((requested.unwrap_or_default(), requested.is_some()));
    }

    if sidecar.exists() {
        let persisted = SchedulerConfig::load(&sidecar)?;
        if let Some(requested) = requested {
            if requested != persisted {
                return Err(format!(
                    "scheduler mismatch on resume: persisted={} requested={}",
                    persisted.line(),
                    requested.line()
                ));
            }
        }
        return Ok((persisted, true));
    }

    if let Some(requested) = requested {
        if requested != SchedulerConfig::default() {
            return Err(
                "checkpoint has no scheduler metadata; non-constant scheduler cannot start mid-run"
                    .into(),
            );
        }
    }
    Ok((SchedulerConfig::default(), false))
}

fn train(
    steps: usize,
    ckpt: &Path,
    fresh: bool,
    run: RunConfig,
    diagnostics: bool,
    architecture: Option<ArchitectureConfig>,
    requested_scheduler: Option<SchedulerConfig>,
) -> Result<(), String> {
    run.validate()
        .map_err(|e| format!("configuración inválida: {e}"))?;
    print!("{}", run.effective_report());

    let (scheduler, persist_scheduler) =
        resolve_scheduler(ckpt, fresh, requested_scheduler)?;
    println!("{}", scheduler.line());

    let effective_batch = run
        .effective_batch_size()
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
        if let Some(expected) = architecture {
            if !expected.matches_model(g.cfg) {
                return Err(format!(
                    "architecture mismatch on resume: checkpoint={} requested={}",
                    ArchitectureConfig {
                        n_embd: g.cfg.n_embd,
                        n_head: g.cfg.n_head,
                        n_layer: g.cfg.n_layer,
                        block: g.cfg.block,
                        n_ff: g.cfg.n_ff,
                    }.line(),
                    expected.line(),
                ));
            }
        }
        println!("reanuda {} adam={}", ckpt.display(), a.is_some());
        (g, t, a, true)
    } else {
        let selected = architecture.unwrap_or_default();
        let (g, t) = new_model(&raw_split.train, &run, selected)?;
        (g, t, None, false)
    };

    let split = encode_split(&tok, &raw_split);
    let identity = ExperimentIdentity::from_raw(run.seed, &text, &split);
    let total_tokens = split.total_len();

    println!(
        "Auralis train | tok={} vocab={} tokens={} chars={} steps={} batch={} accum={} effective_batch={} seed={} lr={} clip={} split={:.3}/{:.3}/{:.3} bpe_merges={}",
        tok.kind(),
        tok.vocab_size(),
        total_tokens,
        text.chars().count(),
        steps,
        run.batch_size,
        run.gradient_accumulation_steps,
        effective_batch,
        run.seed,
        run.learning_rate,
        run.grad_clip_norm,
        run.train_fraction,
        run.validation_fraction,
        1.0 - run.train_fraction - run.validation_fraction,
        run.bpe_merges,
    );
    println!("experiment | {}", identity.line());
    println!(
        "{}",
        ArchitectureConfig {
            n_embd: gpt.cfg.n_embd,
            n_head: gpt.cfg.n_head,
            n_layer: gpt.cfg.n_layer,
            block: gpt.cfg.block,
            n_ff: gpt.cfg.n_ff,
        }
        .line()
    );

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
    let mut workspace = TrainWorkspace::new(&gpt);

    println!("params={} adam.t={}", n_params, adam.t);
    let t0 = Instant::now();
    let mut processed_tokens = 0u64;

    if diagnostics {
        println!("numeric | training_diagnostics=on");
    }

    for local_step in 1..=steps {
        let global_step = adam.t.max(0) as u64;
        let scheduled_lr = scheduler.learning_rate(run.learning_rate, global_step)?;
        adam.lr = scheduled_lr;
        let metrics = if diagnostics {
            let (metrics, report) = train_step_reuse_diagnostics(
                &mut gpt,
                &mut adam,
                &split.train,
                train_cfg,
                global_step,
                &mut grads,
                &mut workspace,
                Diagnostics::on(),
            )
            .map_err(|e| format!("diagnóstico numérico abortó step {global_step}: {e}"))?;
            if let Some(report) = report {
                println!(
                    "numeric_train | global_step={} loss_max_abs={:.6} grad_l2={:.6} grad_max_abs={:.6} pre_slices={} post_slices={}",
                    metrics.global_step,
                    report.loss.max_abs,
                    report.pre_optimizer.gradient.l2.unwrap_or(f32::NAN),
                    report.pre_optimizer.gradient.max_abs,
                    report.pre_optimizer.slices.len(),
                    report.post_optimizer.slices.len(),
                );
            }
            metrics
        } else {
            train_step_reuse(
                &mut gpt,
                &mut adam,
                &split.train,
                train_cfg,
                global_step,
                &mut grads,
                &mut workspace,
            )
            .map_err(|e| format!("entrenamiento abortado en step {global_step}: {e}"))?
        };
        processed_tokens += metrics.tokens as u64;

        if local_step == 1 || local_step % 20 == 0 || local_step == steps {
            let mut eval_rng = StdRng::seed_from_u64(
                run.seed ^ metrics.global_step.wrapping_mul(0x9e3779b97f4a7c15),
            );
            let sample = sample_prompt(&gpt, &tok, "Auralis es", 24, 0.2, &mut eval_rng);
            println!(
                "step {:4} global={} loss {:.4} grad {:.4} clip {:.3} lr={:.8} microbatches={} effective_batch={} sample={}",
                local_step,
                metrics.global_step,
                metrics.loss,
                metrics.grad_norm_before_clip,
                metrics.grad_scale,
                adam.lr,
                metrics.microbatches,
                metrics.effective_batch_size,
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

    let validation = evaluate_holdout("validation", &gpt, &split.validation)?;
    let test = evaluate_holdout("test", &gpt, &split.test)?;

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
    let scheduler_sidecar = scheduler_path(ckpt);
    if persist_scheduler {
        scheduler
            .save(&scheduler_sidecar)
            .map_err(|e| format!("checkpoint guardado pero falló scheduler metadata: {e}"))?;
    } else if fresh && scheduler_sidecar.exists() {
        fs::remove_file(&scheduler_sidecar)
            .map_err(|e| format!("no se pudo limpiar scheduler metadata obsoleta: {e}"))?;
    }
    let checkpoint_bytes = fs::metadata(ckpt)
        .map_err(|e| format!("no se pudo medir checkpoint {}: {e}", ckpt.display()))?
        .len();
    let manifest_bytes = fs::metadata(&manifest_path)
        .map_err(|e| format!("no se pudo medir manifiesto {}: {e}", manifest_path.display()))?
        .len();

    println!(
        "checkpoint → {} | manifest → {}",
        ckpt.display(),
        manifest_path.display()
    );
    if persist_scheduler {
        println!("scheduler_metadata → {}", scheduler_sidecar.display());
    }
    println!(
        "run_summary | params={} optimizer_steps={} batch={} accum={} effective_batch={} train_tokens={} train_seconds={:.6} tok_per_s={:.3} validation_loss={:.6} validation_ppl={:.6} test_loss={:.6} test_ppl={:.6} checkpoint_bytes={} manifest_bytes={}",
        n_params,
        adam.t,
        run.batch_size,
        run.gradient_accumulation_steps,
        effective_batch,
        throughput.tokens,
        throughput.elapsed_seconds,
        throughput.tokens_per_second,
        validation.mean_loss,
        validation.perplexity,
        test.mean_loss,
        test.perplexity,
        checkpoint_bytes,
        manifest_bytes,
    );
    Ok(())
}

fn run_train_or_exit(
    steps: usize,
    ckpt: &Path,
    fresh: bool,
    run: RunConfig,
    diagnostics: bool,
    architecture: Option<ArchitectureConfig>,
    scheduler: Option<SchedulerConfig>,
) {
    if let Err(e) = train(
        steps,
        ckpt,
        fresh,
        run,
        diagnostics,
        architecture,
        scheduler,
    ) {
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

fn print_config(path: Option<&Path>) {
    let run = match path {
        Some(p) => match RunConfig::load(p) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        },
        None => RunConfig::default(),
    };
    if let Err(e) = run.validate() {
        eprintln!("error: configuración inválida: {e}");
        std::process::exit(2);
    }
    print!("{}", run.effective_report());
}

fn run_inspect(args: &[String]) {
    let mut json = false;
    let mut path = PathBuf::from("auralis.bin");
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--json" {
            json = true;
            i += 1;
            continue;
        }
        path = PathBuf::from(&args[i]);
        i += 1;
    }
    match inspect_checkpoint(&path) {
        Ok(report) => {
            if json {
                println!("{}", report.json());
            } else {
                print!("{}", report.human());
            }
            if report.manifest_present && !report.manifest_ok {
                std::process::exit(2);
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}

fn run_release_check(args: &[String]) {
    let mut json = false;
    let mut root = default_root();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--json" {
            json = true;
            i += 1;
            continue;
        }
        root = PathBuf::from(&args[i]);
        i += 1;
    }
    let report = check_release(&root);
    if json {
        println!("{}", report.json());
    } else {
        print!("{}", report.human());
    }
    if !report.automated_pass {
        std::process::exit(2);
    }
}

fn run_release_manifest(args: &[String]) {
    let mut verify: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut root = default_root();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--verify" {
            let path = args.get(i + 1).expect("--verify requires a path");
            verify = Some(PathBuf::from(path));
            i += 2;
            continue;
        }
        if args[i] == "--out" {
            let path = args.get(i + 1).expect("--out requires a path");
            out = Some(PathBuf::from(path));
            i += 2;
            continue;
        }
        root = PathBuf::from(&args[i]);
        i += 1;
    }

    if let Some(path) = verify {
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: cannot read {}: {e}", path.display());
                std::process::exit(2);
            }
        };
        match ReleaseManifest::decode(&text).and_then(|m| m.verify(&root).map(|_| m)) {
            Ok(m) => {
                println!(
                    "release-manifest | verified=true artifacts={} revision={}",
                    m.artifacts.len(),
                    m.code_revision
                );
            }
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        }
        return;
    }

    match ReleaseManifest::capture(&root, DEFAULT_RELEASE_ARTIFACTS) {
        Ok(m) => {
            let encoded = m.encode();
            if let Some(path) = out {
                if let Err(e) = fs::write(&path, &encoded) {
                    eprintln!("error: cannot write {}: {e}", path.display());
                    std::process::exit(2);
                }
                println!(
                    "release-manifest → {} artifacts={}",
                    path.display(),
                    m.artifacts.len()
                );
            } else {
                print!("{encoded}");
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}

fn run_sec_audit(args: &[String]) {
    let root = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("src"));
    match scan_tree(&root) {
        Ok(report) => {
            print!("{}", report.human());
            if !report.ok() {
                std::process::exit(2);
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}

fn parse_bench_run_protocol(rest: &[&str], id: &str) -> Result<BenchProtocol, String> {
    let mut protocol = bench_runner::default_protocol(id)?;
    let mut i = 2usize;
    while i < rest.len() {
        let flag = rest[i];
        let value = rest
            .get(i + 1)
            .ok_or_else(|| format!("{flag} requires an integer value"))?
            .parse::<usize>()
            .map_err(|_| format!("{flag} requires an integer value"))?;
        match flag {
            "--warmup" => protocol.warmup = value,
            "--iterations" | "--steps" => protocol.iterations = value,
            "--repeats" => protocol.repetitions = value,
            other => return Err(format!("unknown bench run option {other}")),
        }
        i += 2;
    }
    protocol.validate()
}

fn run_bench(args: &[String]) {
    let json = args.iter().any(|a| a == "--json");
    let csv = args.iter().any(|a| a == "--csv");
    let fmt = match CatalogFormat::from_flags(json, csv) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let rest: Vec<&str> = args
        .iter()
        .skip(2)
        .map(|s| s.as_str())
        .filter(|s| *s != "--json" && *s != "--csv")
        .collect();
    let command = rest.first().copied().unwrap_or("list");

    if command == "run" {
        let Some(id) = rest.get(1).copied() else {
            eprintln!("error: bench run requires an id");
            std::process::exit(2);
        };
        let protocol = match parse_bench_run_protocol(&rest, id) {
            Ok(protocol) => protocol,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        };
        let execution = match bench_runner::run(id, protocol) {
            Ok(execution) => execution,
            Err(e) => {
                eprintln!("benchmark failed: {e}");
                std::process::exit(3);
            }
        };
        match execution.render(fmt) {
            Ok(out) => print!("{out}"),
            Err(e) => {
                eprintln!("benchmark output failed: {e}");
                std::process::exit(3);
            }
        }
        return;
    }

    let id = rest.get(1).copied();
    match bench_format::render(command, id, fmt) {
        Ok(out) => print!("{out}"),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}


#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct TrainCliOptions {
    diagnostics: bool,
    architecture: Option<ArchitectureConfig>,
    scheduler: Option<SchedulerConfig>,
}

fn parse_token_ids(spec: &str) -> Result<Vec<usize>, String> {
    if spec.trim().is_empty() {
        return Err("--tokens requires at least one token id".into());
    }
    spec.split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return Err("empty token id in --tokens".into());
            }
            part.parse::<usize>()
                .map_err(|_| format!("invalid token id {part:?}"))
        })
        .collect()
}

fn numeric_forward(rest: &[String]) -> Result<(), String> {
    let checkpoint_path = rest
        .first()
        .ok_or("numeric forward requires CHECKPOINT")?;
    let mut token_spec: Option<&str> = None;
    let mut i = 1usize;
    while i < rest.len() {
        match rest[i].as_str() {
            "--tokens" => {
                token_spec = Some(
                    rest.get(i + 1)
                        .ok_or("--tokens requires a comma-separated value")?
                        .as_str(),
                );
                i += 2;
            }
            other => return Err(format!("unknown numeric forward option {other}")),
        }
    }
    let tokens = parse_token_ids(token_spec.ok_or("numeric forward requires --tokens IDS")?)?;
    let (gpt, _) = checkpoint::load(checkpoint_path)
        .map_err(|e| format!("cannot load checkpoint {checkpoint_path}: {e}"))?;
    if tokens.len() > gpt.cfg.block {
        return Err(format!(
            "token count {} exceeds model block {}",
            tokens.len(),
            gpt.cfg.block
        ));
    }
    if let Some((index, token)) = tokens
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token >= gpt.cfg.vocab)
    {
        return Err(format!(
            "token id {token} at position {index} exceeds vocab {}",
            gpt.cfg.vocab
        ));
    }

    let (_, report) = gpt
        .forward_diagnostics(&tokens)
        .map_err(|e| format!("numerical fault: {e}"))?;
    println!(
        "numeric_forward | checkpoint={} tokens={} summaries={} finite=true",
        checkpoint_path,
        tokens.len(),
        report.tensors.len()
    );
    for summary in report.tensors {
        let layer = summary
            .layer
            .map(|layer| layer.to_string())
            .unwrap_or_else(|| "-".into());
        println!(
            "numeric_tensor | layer={} name={} stage={:?} len={} nans={} infs={} max_abs={:.6}",
            layer,
            summary.name,
            summary.stage,
            summary.scan.len,
            summary.scan.nans,
            summary.scan.infs,
            summary.scan.max_abs,
        );
    }
    Ok(())
}

fn numeric_command(args: &[String]) -> Result<i32, String> {
    let rest = &args[2..];
    if rest.first().map(String::as_str) == Some("forward") {
        return numeric_forward(&rest[1..]).map(|_| 0);
    }
    let out = numeric::cli(rest)?;
    print!("{}", out.text);
    Ok(if out.finite { 0 } else { 3 })
}

fn run_numeric(args: &[String]) {
    match numeric_command(args) {
        Ok(0) => {}
        Ok(code) => std::process::exit(code),
        Err(e) if e.starts_with("numerical fault:") => {
            eprintln!("error: {e}");
            std::process::exit(3);
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}

fn parse_train_args(
    args: &[String],
) -> Result<(usize, PathBuf, RunConfig, TrainCliOptions), String> {
    let mut config_path: Option<&str> = None;
    let mut model_config_path: Option<&str> = None;
    let mut scheduler_config_path: Option<&str> = None;
    let mut diagnostics = false;
    let mut positional: Vec<&str> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--config" {
            let path = args.get(i + 1).ok_or("--config requires a path")?;
            config_path = Some(path.as_str());
            i += 2;
            continue;
        }
        if args[i] == "--model-config" {
            let path = args.get(i + 1).ok_or("--model-config requires a path")?;
            model_config_path = Some(path.as_str());
            i += 2;
            continue;
        }
        if args[i] == "--scheduler-config" {
            let path = args.get(i + 1).ok_or("--scheduler-config requires a path")?;
            scheduler_config_path = Some(path.as_str());
            i += 2;
            continue;
        }
        if args[i] == "--diagnostics" {
            diagnostics = true;
            i += 1;
            continue;
        }
        if args[i].starts_with("--") {
            return Err(format!("unknown training option {}", args[i]));
        }
        positional.push(args[i].as_str());
        i += 1;
    }

    if positional.len() > 5 {
        return Err("too many positional training arguments".into());
    }

    let mut run = match config_path {
        Some(path) => RunConfig::load(path)?,
        None => RunConfig::default(),
    };
    run.validate()
        .map_err(|e| format!("configuración inválida: {e}"))?;

    let steps = positional
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);
    let ckpt = positional
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("auralis.bin"));
    if let Some(seed) = positional.get(2).and_then(|s| s.parse().ok()) {
        run.seed = seed;
    }
    if let Some(batch) = positional.get(3).and_then(|s| s.parse().ok()) {
        run.batch_size = batch;
    }
    if let Some(accum) = positional.get(4).and_then(|s| s.parse().ok()) {
        run.gradient_accumulation_steps = accum;
    }
    run.validate()
        .map_err(|e| format!("configuración inválida: {e}"))?;
    let architecture = match model_config_path {
        Some(path) => Some(ArchitectureConfig::load(path)?),
        None => None,
    };
    let scheduler = match scheduler_config_path {
        Some(path) => Some(SchedulerConfig::load(path)?),
        None => None,
    };
    Ok((
        steps,
        ckpt,
        run,
        TrainCliOptions {
            diagnostics,
            architecture,
            scheduler,
        },
    ))
}

fn usage() {
    eprintln!(
        "Auralis\n  auralis train [steps] [checkpoint] [seed] [batch] [accum] [--config FILE] [--model-config FILE] [--scheduler-config FILE] [--diagnostics]\n  auralis train-fresh [steps] [checkpoint] [seed] [batch] [accum] [--config FILE] [--model-config FILE] [--scheduler-config FILE] [--diagnostics]\n  auralis config [FILE]\n  auralis inspect [checkpoint] [--json]\n  auralis release-check [ROOT] [--json]\n  auralis release-manifest [ROOT] [--out FILE] [--verify FILE]\n  auralis sec-audit [SRC_ROOT]\n  auralis bench list [--json|--csv]\n  auralis bench describe ID [--json|--csv]\n  auralis bench run ID [--warmup N] [--iterations N] [--repeats N] [--json|--csv]\n  auralis numeric [VALUES|--fixture NAME]\n  auralis numeric forward CHECKPOINT --tokens 1,2,3\n  auralis eval [checkpoint]\n  auralis chat [checkpoint]\n  auralis check\n  auralis bpe"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let ckpt_default = Path::new("auralis.bin");
    match args.get(1).map(|s| s.as_str()) {
        Some("train") => match parse_train_args(&args) {
            Ok((steps, ckpt, run, options)) => {
                run_train_or_exit(
                    steps,
                    &ckpt,
                    false,
                    run,
                    options.diagnostics,
                    options.architecture,
                    options.scheduler,
                )
            },
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        },
        Some("train-fresh") => match parse_train_args(&args) {
            Ok((steps, ckpt, run, options)) => {
                run_train_or_exit(
                    steps,
                    &ckpt,
                    true,
                    run,
                    options.diagnostics,
                    options.architecture,
                    options.scheduler,
                )
            },
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        },
        Some("config") => print_config(args.get(2).map(Path::new)),
        Some("inspect") => run_inspect(&args),
        Some("release-check") => run_release_check(&args),
        Some("release-manifest") => run_release_manifest(&args),
        Some("sec-audit") => run_sec_audit(&args),
        Some("bench") => run_bench(&args),
        Some("numeric") => run_numeric(&args),
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
            false,
            None,
            None,
        ),
        None => run_train_or_exit(
            80,
            ckpt_default,
            false,
            RunConfig::default(),
            false,
            None,
            None,
        ),
        _ => usage(),
    }
}


#[cfg(test)]
mod cli_tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn diagnostics_flag_is_ephemeral_and_not_positional() {
        let args = strings(&[
            "auralis",
            "train",
            "3",
            "out.bin",
            "11",
            "2",
            "4",
            "--diagnostics",
        ]);
        let (steps, ckpt, run, options) = parse_train_args(&args).unwrap();
        assert_eq!(steps, 3);
        assert_eq!(ckpt, PathBuf::from("out.bin"));
        assert_eq!(run.seed, 11);
        assert_eq!(run.batch_size, 2);
        assert_eq!(run.gradient_accumulation_steps, 4);
        assert!(options.diagnostics);
        assert!(options.architecture.is_none());
        assert!(options.scheduler.is_none());
    }

    #[test]
    fn normal_training_cli_keeps_diagnostics_off() {
        let args = strings(&["auralis", "train", "3", "out.bin"]);
        let (_, _, _, options) = parse_train_args(&args).unwrap();
        assert!(!options.diagnostics);
        assert!(options.architecture.is_none());
        assert!(options.scheduler.is_none());
    }

    #[test]
    fn unknown_training_option_is_rejected() {
        let args = strings(&["auralis", "train", "--unknown"]);
        assert!(parse_train_args(&args).is_err());
    }

    #[test]
    fn token_id_parser_is_strict() {
        assert_eq!(parse_token_ids("1, 2,3").unwrap(), vec![1, 2, 3]);
        assert!(parse_token_ids("").is_err());
        assert!(parse_token_ids("1,,3").is_err());
        assert!(parse_token_ids("1,x").is_err());
    }
}
