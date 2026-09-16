use auralis::agent::Agent;
use auralis::bpe::BpeTokenizer;
use auralis::gradcheck;
use auralis::checkpoint;
use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use auralis::tokenizer::{AnyTok, CharTokenizer};
use rand::Rng;
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

fn sample_prompt(gpt: &Gpt, tok: &AnyTok, prompt: &str, n: usize, temp: f32, rng: &mut impl Rng) -> String {
    let mut out = tok.encode(prompt);
    if out.is_empty() {
        out = tok.encode("A");
    }
    gpt.generate(&mut out, n, temp, rng);
    tok.decode(&out)
}

fn train(steps: usize, ckpt: &Path, fresh: bool) {
    let text = load_corpus();
    let mut rng = rand::thread_rng();
    let (mut gpt, tok, saved_adam) = if !fresh && ckpt.exists() {
        match checkpoint::load_full(ckpt) {
            Ok((g, t, a)) => {
                println!("reanuda {} adam={}", ckpt.display(), a.is_some());
                (g, t, a)
            }
            Err(e) => {
                eprintln!("checkpoint ilegible ({e}); entreno desde cero");
                let tok = AnyTok::Bpe(BpeTokenizer::fit(&text, 64));
                let gpt = Gpt::new(Config::tiny(tok.vocab_size()), &mut rng);
                (gpt, tok, None)
            }
        }
    } else {
        let tok = AnyTok::Bpe(BpeTokenizer::fit(&text, 64));
        let gpt = Gpt::new(Config::tiny(tok.vocab_size()), &mut rng);
        (gpt, tok, None)
    };
    let ids_all = tok.encode(&text);
    println!(
        "Auralis train | tok={} vocab={} tokens={} chars={} steps={}",
        tok.kind(), tok.vocab_size(), ids_all.len(), text.chars().count(), steps
    );
    let block = gpt.cfg.block;
    let mut params = gpt.collect_params();
    let mut grads = vec![0.0; params.len()];
    let mut adam = saved_adam
        .filter(|a| a.export().2.len() == params.len())
        .unwrap_or_else(|| Adam::new(params.len(), 3e-3));
    println!("adam.t={}", adam.t);
    if ids_all.len() <= block + 1 {
        eprintln!("corpus demasiado corto");
        return;
    }
    let mut anchors = Vec::new();
    let span = 12.min(ids_all.len());
    for i in 0..ids_all.len().saturating_sub(span) {
        let piece = tok.decode(&ids_all[i..i + span]);
        if piece.contains("Auralis es una") {
            let s = i.saturating_sub(2);
            if s + block + 1 < ids_all.len() {
                anchors.push(s);
            }
        }
    }
    println!("ventanas ancla={}", anchors.len());
    let t0 = Instant::now();
    for step in 1..=steps {
        let max_start = ids_all.len() - block - 1;
        let start = if !anchors.is_empty() && rng.gen_bool(0.12) {
            anchors[rng.gen_range(0..anchors.len())].min(max_start)
        } else {
            rng.gen_range(0..max_start)
        };
        let x = &ids_all[start..start + block];
        let y = &ids_all[start + 1..start + block + 1];
        let loss = gpt.backward_into(x, y, &mut grads);
        let gnorm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        if gnorm > 1.0 {
            let s = 1.0 / gnorm;
            for g in grads.iter_mut() { *g *= s; }
        }
        adam.step(&mut params, &grads);
        gpt.write_params(&params);
        if step == 1 || step % 20 == 0 || step == steps {
            let eval = sample_prompt(&gpt, &tok, "Auralis es", 24, 0.2, &mut rng);
            let hit = eval.contains("inteligencia");
            println!("step {step:4}  loss {loss:.4}  hit={hit}  eval={eval}");
        }
    }
    if let Err(e) = checkpoint::save_full(ckpt, &gpt, &tok, Some(&adam)) {
        eprintln!("no se pudo guardar {ckpt:?}: {e}");
    } else {
        println!("checkpoint → {}", ckpt.display());
    }
    let secs = t0.elapsed().as_secs_f32().max(1e-6);
    let toks = steps as f32 * block as f32;
    println!("tiempo={:.1}s  {:.0} tok/s  adam.t={}", secs, toks / secs, adam.t);
}

fn eval_ckpt(ckpt: &Path) {
    let (gpt, tok) = match checkpoint::load(ckpt) {
        Ok(v) => v,
        Err(e) => { eprintln!("carga {ckpt:?}: {e}"); return; }
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
        Err(e) => { eprintln!("carga {ckpt:?}: {e}\nEntrena antes: auralis train 80"); return; }
    };
    let mut agent = Agent::new();
    let mut rng = rand::thread_rng();
    println!("Auralis chat  (escribe /salir)");
    let stdin = io::stdin();
    loop {
        print!("tú> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() { break; }
        let line = line.trim();
        if line.is_empty() { continue; }
        if line == "/salir" || line == "/quit" { break; }
        let reply = agent.reply(line, &gpt, &tok, &mut rng, 60);
        println!("auralis> {reply}");
    }
}

fn run_check() {
    let text = load_corpus();
    let tok = CharTokenizer::fit(&text);
    let ids = tok.encode(&text);
    let mut rng = rand::thread_rng();
    let cfg = gradcheck::tiny_check_config(tok.vocab_size());
    let block = cfg.block.min(ids.len().saturating_sub(2).max(2));
    let mut gpt = Gpt::new(Config { block, ..cfg }, &mut rng);
    let x = &ids[0..block];
    let y = &ids[1..block + 1];
    println!("gradcheck | params={} block={} vocab={}", gpt.collect_params().len(), block, tok.vocab_size());
    let report = gradcheck::check_random_params(&gpt, x, y, 6, 1e-3, &mut rng);
    println!("checked={}  max_abs={:.4e}  max_rel={:.4}  mean_rel={:.4}  ok={}",
        report.checked, report.max_abs_err, report.max_rel_err, report.mean_rel_err, report.ok(0.25));
}

fn run_bpe() {
    let text = load_corpus();
    let bpe = BpeTokenizer::fit(&text, 80);
    let ids = bpe.encode(&text);
    println!("bpe | merges={} vocab={} chars={} tokens={}", bpe.merges.len(), bpe.vocab_size(), text.chars().count(), ids.len());
    for (i, (a, b)) in bpe.merges.iter().take(8).enumerate() {
        println!("  merge {i}: {a:?} + {b:?} -> {:?}", format!("{a}{b}"));
    }
    let sample = text.chars().take(80).collect::<String>();
    println!("roundtrip: {}", bpe.decode(&bpe.encode(&sample)) == sample);
}

fn usage() {
    eprintln!("Auralis v0\n  auralis train|train-fresh|chat|eval|check|bpe");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let ckpt_default = Path::new("auralis.bin");
    match args.get(1).map(|s| s.as_str()) {
        Some("train") => {
            let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(80);
            let ckpt = args.get(3).map(Path::new).unwrap_or(ckpt_default);
            train(steps, ckpt, false);
        }
        Some("train-fresh") => {
            let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(80);
            let ckpt = args.get(3).map(Path::new).unwrap_or(ckpt_default);
            train(steps, ckpt, true);
        }
        Some("chat") => chat(args.get(2).map(Path::new).unwrap_or(ckpt_default)),
        Some("check") => run_check(),
        Some("bpe") => run_bpe(),
        Some("eval") => eval_ckpt(args.get(2).map(Path::new).unwrap_or(ckpt_default)),
        Some("-h" | "--help" | "help") => usage(),
        Some(n) if n.chars().all(|c| c.is_ascii_digit()) => train(n.parse().unwrap(), ckpt_default, false),
        None => train(80, ckpt_default, false),
        _ => usage(),
    }
}
