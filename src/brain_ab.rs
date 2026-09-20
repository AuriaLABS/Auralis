//! Reproducible A/B architecture experiment harness for Brain.
//!
//! Both variants share one protocol, seed, token stream and optimization
//! budget. The harness reports measurements; it does not choose a winner.

use crate::checkpoint;
use crate::eval::evaluate_tokens_reference;
use crate::manifest::{build_revision, fingerprint_params};
use crate::model::{Config, Gpt};
use crate::optim::Adam;
use crate::tokenizer::{AnyTok, CharTokenizer};
use crate::training::{train_step_reuse, TrainConfig, TrainWorkspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

pub const BRAIN_AB_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbProtocol {
    pub seed: u64,
    pub steps: usize,
    pub repeats: usize,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f32,
    pub grad_clip_norm: f32,
    pub token_count: usize,
}

impl Default for AbProtocol {
    fn default() -> Self {
        Self {
            seed: 659_918,
            steps: 20,
            repeats: 3,
            batch_size: 2,
            gradient_accumulation_steps: 2,
            learning_rate: 3e-3,
            grad_clip_norm: 1.0,
            token_count: 4096,
        }
    }
}

impl AbProtocol {
    pub fn validate(self) -> Result<Self, String> {
        if self.steps == 0 || self.repeats == 0 {
            return Err("A/B steps and repeats must be positive".into());
        }
        if self.batch_size == 0 || self.gradient_accumulation_steps == 0 {
            return Err("A/B batch and accumulation must be positive".into());
        }
        if self.token_count < 32 {
            return Err("A/B token_count must be at least 32".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("A/B learning_rate must be finite and positive".into());
        }
        if !self.grad_clip_norm.is_finite() || self.grad_clip_norm <= 0.0 {
            return Err("A/B grad_clip_norm must be finite and positive".into());
        }
        Ok(self)
    }

    fn train_config(self) -> TrainConfig {
        TrainConfig {
            seed: self.seed,
            batch_size: self.batch_size,
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            grad_clip_norm: self.grad_clip_norm,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbVariant {
    pub label: String,
    pub config: Config,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AbMeasurement {
    pub repetition: usize,
    pub final_train_loss: f32,
    pub eval_loss: f32,
    pub eval_perplexity: f32,
    pub tokens_per_second: f64,
    pub parameter_count: usize,
    pub parameter_bytes: usize,
    pub optimizer_state_bytes: usize,
    pub checkpoint_bytes: u64,
    pub state_fingerprint: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AbVariantResult {
    pub label: String,
    pub config: Config,
    pub measurements: Vec<AbMeasurement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AbExperimentResult {
    pub schema_version: u32,
    pub code_revision: String,
    pub protocol: AbProtocol,
    pub token_fingerprint: u64,
    pub a: AbVariantResult,
    pub b: AbVariantResult,
}

impl AbExperimentResult {
    pub fn a_vs_a_reproducible(&self) -> bool {
        if self.a.config != self.b.config {
            return false;
        }
        self.a
            .measurements
            .iter()
            .zip(&self.b.measurements)
            .all(|(a, b)| {
                a.final_train_loss.to_bits() == b.final_train_loss.to_bits()
                    && a.eval_loss.to_bits() == b.eval_loss.to_bits()
                    && a.eval_perplexity.to_bits() == b.eval_perplexity.to_bits()
                    && a.parameter_count == b.parameter_count
                    && a.parameter_bytes == b.parameter_bytes
                    && a.optimizer_state_bytes == b.optimizer_state_bytes
                    && a.checkpoint_bytes == b.checkpoint_bytes
                    && a.state_fingerprint == b.state_fingerprint
            })
    }

    pub fn human(&self) -> String {
        let mut out = format!(
            "brain_ab | schema={} revision={} seed={} steps={} repeats={} batch={} accum={} lr={} clip={} tokens={} token_fingerprint={:016x}\n",
            self.schema_version,
            self.code_revision,
            self.protocol.seed,
            self.protocol.steps,
            self.protocol.repeats,
            self.protocol.batch_size,
            self.protocol.gradient_accumulation_steps,
            self.protocol.learning_rate,
            self.protocol.grad_clip_norm,
            self.protocol.token_count,
            self.token_fingerprint,
        );
        for variant in [&self.a, &self.b] {
            for m in &variant.measurements {
                out.push_str(&format!(
                    "brain_ab_run | variant={} repetition={} n_embd={} n_head={} n_layer={} block={} n_ff={} train_loss={:.6} eval_loss={:.6} ppl={:.6} tok_per_s={:.3} params={} parameter_bytes={} optimizer_state_bytes={} checkpoint_bytes={} state_fingerprint={:016x}\n",
                    variant.label,
                    m.repetition,
                    variant.config.n_embd,
                    variant.config.n_head,
                    variant.config.n_layer,
                    variant.config.block,
                    variant.config.n_ff,
                    m.final_train_loss,
                    m.eval_loss,
                    m.eval_perplexity,
                    m.tokens_per_second,
                    m.parameter_count,
                    m.parameter_bytes,
                    m.optimizer_state_bytes,
                    m.checkpoint_bytes,
                    m.state_fingerprint,
                ));
            }
        }
        out.push_str(&format!(
            "brain_ab_summary | a={} b={} a_vs_a_reproducible={} comparable_budget=true\n",
            self.a.label,
            self.b.label,
            self.a_vs_a_reproducible(),
        ));
        out
    }

    pub fn json(&self) -> String {
        fn cfg(c: Config) -> String {
            format!(
                "{{\"vocab\":{},\"n_embd\":{},\"n_head\":{},\"n_layer\":{},\"block\":{},\"n_ff\":{}}}",
                c.vocab, c.n_embd, c.n_head, c.n_layer, c.block, c.n_ff
            )
        }
        fn measurement(m: &AbMeasurement) -> String {
            format!(
                concat!(
                    "{{\"repetition\":{},\"final_train_loss\":{},\"eval_loss\":{},",
                    "\"eval_perplexity\":{},\"tokens_per_second\":{},",
                    "\"parameter_count\":{},\"parameter_bytes\":{},",
                    "\"optimizer_state_bytes\":{},\"checkpoint_bytes\":{},",
                    "\"state_fingerprint\":\"{:016x}\"}}"
                ),
                m.repetition,
                m.final_train_loss,
                m.eval_loss,
                m.eval_perplexity,
                m.tokens_per_second,
                m.parameter_count,
                m.parameter_bytes,
                m.optimizer_state_bytes,
                m.checkpoint_bytes,
                m.state_fingerprint,
            )
        }
        fn variant(v: &AbVariantResult) -> String {
            let ms = v
                .measurements
                .iter()
                .map(measurement)
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"label\":\"{}\",\"config\":{},\"measurements\":[{}]}}",
                escape_json(&v.label),
                cfg(v.config),
                ms,
            )
        }
        format!(
            concat!(
                "{{\"schema_version\":{},\"code_revision\":\"{}\",",
                "\"protocol\":{{\"seed\":{},\"steps\":{},\"repeats\":{},",
                "\"batch_size\":{},\"gradient_accumulation_steps\":{},",
                "\"learning_rate\":{},\"grad_clip_norm\":{},\"token_count\":{}}},",
                "\"token_fingerprint\":\"{:016x}\",\"a\":{},\"b\":{},",
                "\"a_vs_a_reproducible\":{}}}"
            ),
            self.schema_version,
            escape_json(&self.code_revision),
            self.protocol.seed,
            self.protocol.steps,
            self.protocol.repeats,
            self.protocol.batch_size,
            self.protocol.gradient_accumulation_steps,
            self.protocol.learning_rate,
            self.protocol.grad_clip_norm,
            self.protocol.token_count,
            self.token_fingerprint,
            variant(&self.a),
            variant(&self.b),
            self.a_vs_a_reproducible(),
        )
    }
}

pub fn run_experiment(
    protocol: AbProtocol,
    a: AbVariant,
    b: AbVariant,
) -> Result<AbExperimentResult, String> {
    let protocol = protocol.validate()?;
    validate_variants(&a, &b, protocol.token_count)?;
    let tokens = token_stream(a.config.vocab, protocol.token_count);
    let token_fingerprint = fingerprint_tokens(&tokens);

    let mut a_measurements = Vec::with_capacity(protocol.repeats);
    let mut b_measurements = Vec::with_capacity(protocol.repeats);
    for repetition in 0..protocol.repeats {
        if repetition % 2 == 0 {
            a_measurements.push(run_variant(&a, protocol, &tokens, repetition + 1)?);
            b_measurements.push(run_variant(&b, protocol, &tokens, repetition + 1)?);
        } else {
            b_measurements.push(run_variant(&b, protocol, &tokens, repetition + 1)?);
            a_measurements.push(run_variant(&a, protocol, &tokens, repetition + 1)?);
        }
    }

    Ok(AbExperimentResult {
        schema_version: BRAIN_AB_SCHEMA_VERSION,
        code_revision: build_revision().to_string(),
        protocol,
        token_fingerprint,
        a: AbVariantResult {
            label: a.label,
            config: a.config,
            measurements: a_measurements,
        },
        b: AbVariantResult {
            label: b.label,
            config: b.config,
            measurements: b_measurements,
        },
    })
}

fn validate_variants(a: &AbVariant, b: &AbVariant, token_count: usize) -> Result<(), String> {
    if a.label.trim().is_empty() || b.label.trim().is_empty() {
        return Err("A/B variant labels must not be empty".into());
    }
    if a.config.vocab != b.config.vocab {
        return Err("A/B variants must use the same vocabulary".into());
    }
    for (label, cfg) in [(&a.label, a.config), (&b.label, b.config)] {
        if cfg.vocab <= 1
            || cfg.n_embd == 0
            || cfg.n_head == 0
            || cfg.n_layer == 0
            || cfg.block == 0
            || cfg.n_ff == 0
            || cfg.n_embd % cfg.n_head != 0
        {
            return Err(format!("invalid A/B model config for {label}"));
        }
        if token_count <= cfg.block {
            return Err(format!("A/B token budget too short for {label} block"));
        }
    }
    Ok(())
}

fn run_variant(
    variant: &AbVariant,
    protocol: AbProtocol,
    tokens: &[usize],
    repetition: usize,
) -> Result<AbMeasurement, String> {
    let mut rng = StdRng::seed_from_u64(protocol.seed);
    let mut gpt = Gpt::new(variant.config, &mut rng);
    let parameter_count = gpt.collect_params().len();
    let mut adam = Adam::new(parameter_count, protocol.learning_rate);
    let mut grads = vec![0.0f32; parameter_count];
    let mut workspace = TrainWorkspace::new(&gpt);
    let cfg = protocol.train_config();
    let started = Instant::now();
    let mut processed_tokens = 0u64;
    let mut final_train_loss = f32::NAN;

    for _ in 0..protocol.steps {
        let global_step = adam.t.max(0) as u64;
        let metrics = train_step_reuse(
            &mut gpt,
            &mut adam,
            tokens,
            cfg,
            global_step,
            &mut grads,
            &mut workspace,
        )
        .map_err(|e| format!("A/B training failed for {}: {e}", variant.label))?;
        processed_tokens += metrics.tokens as u64;
        final_train_loss = metrics.loss;
    }
    let elapsed = started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
    let eval = evaluate_tokens_reference(&gpt, tokens)
        .map_err(|e| format!("A/B eval failed for {}: {e}", variant.label))?;
    let params = gpt.collect_params();
    let state_fingerprint = training_state_fingerprint(&params, &adam);

    let tok = synthetic_tokenizer(variant.config.vocab)?;
    let path = temporary_checkpoint_path(&variant.label, repetition);
    checkpoint::save_full(&path, &gpt, &tok, Some(&adam))
        .map_err(|e| format!("A/B checkpoint failed for {}: {e}", variant.label))?;
    let checkpoint_bytes = fs::metadata(&path)
        .map_err(|e| format!("A/B checkpoint metadata failed: {e}"))?
        .len();
    let _ = fs::remove_file(&path);

    Ok(AbMeasurement {
        repetition,
        final_train_loss,
        eval_loss: eval.mean_loss,
        eval_perplexity: eval.perplexity,
        tokens_per_second: processed_tokens as f64 / elapsed,
        parameter_count,
        parameter_bytes: parameter_count.saturating_mul(4),
        optimizer_state_bytes: parameter_count.saturating_mul(8),
        checkpoint_bytes,
        state_fingerprint,
    })
}

fn training_state_fingerprint(params: &[f32], adam: &Adam) -> u64 {
    let mut h = fingerprint_params(params);
    let (lr, t, m, v) = adam.export();
    for byte in lr.to_bits().to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for byte in t.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for values in [m, v] {
        for &value in values {
            for byte in value.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}

fn token_stream(vocab: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| (i.wrapping_mul(37).wrapping_add(i / 7).wrapping_add(11)) % vocab)
        .collect()
}

fn fingerprint_tokens(tokens: &[usize]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &token in tokens {
        for byte in (token as u64).to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn synthetic_tokenizer(vocab: usize) -> Result<AnyTok, String> {
    if vocab < 2 || vocab > 95 {
        return Err("A/B synthetic tokenizer currently supports vocab 2..=95".into());
    }
    let itos: Vec<char> = (0..vocab)
        .map(|i| char::from_u32(32 + i as u32).unwrap())
        .collect();
    let stoi: HashMap<char, usize> = itos.iter().copied().enumerate().map(|(i, c)| (c, i)).collect();
    Ok(AnyTok::Char(CharTokenizer { stoi, itos }))
}

fn temporary_checkpoint_path(label: &str, repetition: usize) -> PathBuf {
    let safe = label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    std::env::temp_dir().join(format!(
        "auralis-ab-{}-{}-{}-{}.bin",
        std::process::id(),
        safe,
        repetition,
        std::thread::current().name().unwrap_or("main"),
    ))
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Config {
        Config {
            vocab: 8,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 8,
            n_ff: 16,
        }
    }

    #[test]
    fn a_vs_a_is_reproducible_except_wall_clock() {
        let result = run_experiment(
            AbProtocol {
                steps: 2,
                repeats: 2,
                token_count: 128,
                ..AbProtocol::default()
            },
            AbVariant { label: "A".into(), config: tiny() },
            AbVariant { label: "B".into(), config: tiny() },
        ).unwrap();
        assert!(result.a_vs_a_reproducible());
        assert_eq!(result.a.measurements[0].state_fingerprint, result.a.measurements[1].state_fingerprint);
        assert!(result.json().contains("\"a_vs_a_reproducible\":true"));
        assert!(result.human().contains("comparable_budget=true"));
    }

    #[test]
    fn mismatched_vocab_is_rejected() {
        let mut other = tiny();
        other.vocab = 9;
        assert!(run_experiment(
            AbProtocol { steps: 1, repeats: 1, token_count: 64, ..AbProtocol::default() },
            AbVariant { label: "A".into(), config: tiny() },
            AbVariant { label: "B".into(), config: other },
        ).is_err());
    }
}
