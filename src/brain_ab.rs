//! Reproducible A/B architecture experiment harness for Brain.
//!
//! Both variants share one protocol, seed, token stream and optimization
//! budget. The harness reports measurements; it does not choose a winner.

use crate::checkpoint;
use crate::eval::evaluate_tokens_reference;
use crate::manifest::{build_revision, fingerprint_params};
use crate::model::{Config, Gpt, NormalizationKind};
use crate::position::PositionKind;
use crate::optim::{Adam, AdamW, Lion, Optimizer, OptimizerId};
use crate::tokenizer::{AnyTok, CharTokenizer};
use crate::training::{train_step_reuse, TrainConfig, TrainWorkspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

pub const BRAIN_AB_SCHEMA_VERSION: u32 = 4;
pub const ATTENTION_HEAD_AB_SCHEMA_VERSION: u32 = 1;
pub const ATTENTION_WINDOW_AB_SCHEMA_VERSION: u32 = 1;

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
    pub normalization: NormalizationKind,
    pub position: PositionKind,
    pub optimizer: OptimizerId,
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
    pub optimizer_state_serialized_bytes: usize,
    pub checkpoint_bytes: u64,
    pub checkpoint_includes_optimizer: bool,
    pub state_fingerprint: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AbVariantResult {
    pub label: String,
    pub config: Config,
    pub normalization: NormalizationKind,
    pub position: PositionKind,
    pub optimizer: OptimizerId,
    pub measurements: Vec<AbMeasurement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionHeadVariantResult {
    pub label: String,
    pub config: Config,
    pub n_kv_head: usize,
    pub normalization: NormalizationKind,
    pub position: PositionKind,
    pub optimizer: OptimizerId,
    pub measurements: Vec<AbMeasurement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionHeadExperimentResult {
    pub schema_version: u32,
    pub code_revision: String,
    pub protocol: AbProtocol,
    pub token_fingerprint: u64,
    pub a: AttentionHeadVariantResult,
    pub b: AttentionHeadVariantResult,
}

impl AttentionHeadExperimentResult {
    pub fn human(&self) -> String {
        let mut out = format!(
            "attention_head_ab | schema={} revision={} seed={} steps={} repeats={} tokens={} token_fingerprint={:016x}\n",
            self.schema_version,
            self.code_revision,
            self.protocol.seed,
            self.protocol.steps,
            self.protocol.repeats,
            self.protocol.token_count,
            self.token_fingerprint,
        );
        for variant in [&self.a, &self.b] {
            for m in &variant.measurements {
                out.push_str(&format!(
                    "attention_head_ab_run | variant={} repetition={} n_head={} n_kv_head={} normalization={} position={} optimizer={} train_loss={:.6} eval_loss={:.6} ppl={:.6} tok_per_s={:.3} params={} parameter_bytes={} checkpoint_bytes={} state_fingerprint={:016x}\n",
                    variant.label,
                    m.repetition,
                    variant.config.n_head,
                    variant.n_kv_head,
                    variant.normalization.as_str(),
                    variant.position.as_str(),
                    variant.optimizer.as_str(),
                    m.final_train_loss,
                    m.eval_loss,
                    m.eval_perplexity,
                    m.tokens_per_second,
                    m.parameter_count,
                    m.parameter_bytes,
                    m.checkpoint_bytes,
                    m.state_fingerprint,
                ));
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionWindowVariantResult {
    pub label: String,
    pub config: Config,
    pub n_kv_head: usize,
    pub attention_window: usize,
    pub normalization: NormalizationKind,
    pub position: PositionKind,
    pub optimizer: OptimizerId,
    pub measurements: Vec<AbMeasurement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionWindowExperimentResult {
    pub schema_version: u32,
    pub code_revision: String,
    pub protocol: AbProtocol,
    pub token_fingerprint: u64,
    pub a: AttentionWindowVariantResult,
    pub b: AttentionWindowVariantResult,
}

impl AttentionWindowExperimentResult {
    pub fn human(&self) -> String {
        let mut out = format!(
            "attention_window_ab | schema={} revision={} seed={} steps={} repeats={} tokens={} token_fingerprint={:016x}\n",
            self.schema_version,
            self.code_revision,
            self.protocol.seed,
            self.protocol.steps,
            self.protocol.repeats,
            self.protocol.token_count,
            self.token_fingerprint,
        );
        for variant in [&self.a, &self.b] {
            for m in &variant.measurements {
                out.push_str(&format!(
                    "attention_window_ab_run | variant={} repetition={} n_head={} n_kv_head={} attention_window={} normalization={} position={} optimizer={} train_loss={:.6} eval_loss={:.6} ppl={:.6} tok_per_s={:.3} params={} parameter_bytes={} checkpoint_bytes={} state_fingerprint={:016x}\n",
                    variant.label,
                    m.repetition,
                    variant.config.n_head,
                    variant.n_kv_head,
                    variant.attention_window,
                    variant.normalization.as_str(),
                    variant.position.as_str(),
                    variant.optimizer.as_str(),
                    m.final_train_loss,
                    m.eval_loss,
                    m.eval_perplexity,
                    m.tokens_per_second,
                    m.parameter_count,
                    m.parameter_bytes,
                    m.checkpoint_bytes,
                    m.state_fingerprint,
                ));
            }
        }
        out
    }
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
        if self.a.config != self.b.config
            || self.a.normalization != self.b.normalization
            || self.a.position != self.b.position
            || self.a.optimizer != self.b.optimizer
        {
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
                    && a.optimizer_state_serialized_bytes == b.optimizer_state_serialized_bytes
                    && a.checkpoint_bytes == b.checkpoint_bytes
                    && a.checkpoint_includes_optimizer == b.checkpoint_includes_optimizer
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
                    "brain_ab_run | variant={} repetition={} normalization={} position={} optimizer={} n_embd={} n_head={} n_layer={} block={} n_ff={} train_loss={:.6} eval_loss={:.6} ppl={:.6} tok_per_s={:.3} params={} parameter_bytes={} optimizer_state_bytes={} optimizer_state_serialized_bytes={} checkpoint_bytes={} checkpoint_includes_optimizer={} state_fingerprint={:016x}\n",
                    variant.label,
                    m.repetition,
                    variant.normalization.as_str(),
                    variant.position.as_str(),
                    variant.optimizer.as_str(),
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
                    m.optimizer_state_serialized_bytes,
                    m.checkpoint_bytes,
                    m.checkpoint_includes_optimizer,
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
                    "\"optimizer_state_bytes\":{},\"optimizer_state_serialized_bytes\":{},",
                    "\"checkpoint_bytes\":{},\"checkpoint_includes_optimizer\":{},",
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
                m.optimizer_state_serialized_bytes,
                m.checkpoint_bytes,
                m.checkpoint_includes_optimizer,
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
                "{{\"label\":\"{}\",\"normalization\":\"{}\",\"position\":\"{}\",\"optimizer\":\"{}\",\"config\":{},\"measurements\":[{}]}}",
                escape_json(&v.label),
                v.normalization.as_str(),
                v.position.as_str(),
                v.optimizer.as_str(),
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

pub fn run_attention_window_experiment(
    protocol: AbProtocol,
    a: AbVariant,
    a_window: usize,
    b: AbVariant,
    b_window: usize,
    n_kv_head: usize,
) -> Result<AttentionWindowExperimentResult, String> {
    let protocol = protocol.validate()?;
    validate_variants(&a, &b, protocol.token_count)?;
    if a.config != b.config
        || a.normalization != b.normalization
        || a.position != b.position
        || a.optimizer != b.optimizer
    {
        return Err(
            "attention-window A/B must differ only in label/window; model and training policy must match"
                .into(),
        );
    }
    if n_kv_head == 0
        || n_kv_head > a.config.n_head
        || a.config.n_head % n_kv_head != 0
    {
        return Err(format!(
            "invalid attention-window A/B KV-head count: n_head={} n_kv_head={n_kv_head}",
            a.config.n_head
        ));
    }
    for (label, window) in [(&a.label, a_window), (&b.label, b_window)] {
        if window > a.config.block {
            return Err(format!(
                "invalid attention window for {label}: window={window} block={}",
                a.config.block
            ));
        }
    }
    if a_window == b_window {
        return Err("attention-window A/B requires distinct window policies".into());
    }

    let tokens = token_stream(a.config.vocab, protocol.token_count);
    let token_fingerprint = fingerprint_tokens(&tokens);
    let mut a_measurements = Vec::with_capacity(protocol.repeats);
    let mut b_measurements = Vec::with_capacity(protocol.repeats);

    for repetition in 0..protocol.repeats {
        if repetition % 2 == 0 {
            a_measurements.push(run_variant_with_attention_policy(
                &a,
                n_kv_head,
                a_window,
                protocol,
                &tokens,
                repetition + 1,
            )?);
            b_measurements.push(run_variant_with_attention_policy(
                &b,
                n_kv_head,
                b_window,
                protocol,
                &tokens,
                repetition + 1,
            )?);
        } else {
            b_measurements.push(run_variant_with_attention_policy(
                &b,
                n_kv_head,
                b_window,
                protocol,
                &tokens,
                repetition + 1,
            )?);
            a_measurements.push(run_variant_with_attention_policy(
                &a,
                n_kv_head,
                a_window,
                protocol,
                &tokens,
                repetition + 1,
            )?);
        }
    }

    Ok(AttentionWindowExperimentResult {
        schema_version: ATTENTION_WINDOW_AB_SCHEMA_VERSION,
        code_revision: build_revision().to_string(),
        protocol,
        token_fingerprint,
        a: AttentionWindowVariantResult {
            label: a.label,
            config: a.config,
            n_kv_head,
            attention_window: a_window,
            normalization: a.normalization,
            position: a.position,
            optimizer: a.optimizer,
            measurements: a_measurements,
        },
        b: AttentionWindowVariantResult {
            label: b.label,
            config: b.config,
            n_kv_head,
            attention_window: b_window,
            normalization: b.normalization,
            position: b.position,
            optimizer: b.optimizer,
            measurements: b_measurements,
        },
    })
}

pub fn run_attention_head_experiment(
    protocol: AbProtocol,
    a: AbVariant,
    a_n_kv_head: usize,
    b: AbVariant,
    b_n_kv_head: usize,
) -> Result<AttentionHeadExperimentResult, String> {
    let protocol = protocol.validate()?;
    validate_variants(&a, &b, protocol.token_count)?;
    for (label, cfg, n_kv_head) in [
        (&a.label, a.config, a_n_kv_head),
        (&b.label, b.config, b_n_kv_head),
    ] {
        if n_kv_head == 0 || n_kv_head > cfg.n_head || cfg.n_head % n_kv_head != 0 {
            return Err(format!(
                "invalid A/B KV-head count for {label}: n_head={} n_kv_head={n_kv_head}",
                cfg.n_head
            ));
        }
    }

    let tokens = token_stream(a.config.vocab, protocol.token_count);
    let token_fingerprint = fingerprint_tokens(&tokens);
    let mut a_measurements = Vec::with_capacity(protocol.repeats);
    let mut b_measurements = Vec::with_capacity(protocol.repeats);
    for repetition in 0..protocol.repeats {
        if repetition % 2 == 0 {
            a_measurements.push(run_variant_with_kv_heads(
                &a,
                a_n_kv_head,
                protocol,
                &tokens,
                repetition + 1,
            )?);
            b_measurements.push(run_variant_with_kv_heads(
                &b,
                b_n_kv_head,
                protocol,
                &tokens,
                repetition + 1,
            )?);
        } else {
            b_measurements.push(run_variant_with_kv_heads(
                &b,
                b_n_kv_head,
                protocol,
                &tokens,
                repetition + 1,
            )?);
            a_measurements.push(run_variant_with_kv_heads(
                &a,
                a_n_kv_head,
                protocol,
                &tokens,
                repetition + 1,
            )?);
        }
    }

    Ok(AttentionHeadExperimentResult {
        schema_version: ATTENTION_HEAD_AB_SCHEMA_VERSION,
        code_revision: build_revision().to_string(),
        protocol,
        token_fingerprint,
        a: AttentionHeadVariantResult {
            label: a.label,
            config: a.config,
            n_kv_head: a_n_kv_head,
            normalization: a.normalization,
            position: a.position,
            optimizer: a.optimizer,
            measurements: a_measurements,
        },
        b: AttentionHeadVariantResult {
            label: b.label,
            config: b.config,
            n_kv_head: b_n_kv_head,
            normalization: b.normalization,
            position: b.position,
            optimizer: b.optimizer,
            measurements: b_measurements,
        },
    })
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
            normalization: a.normalization,
            position: a.position,
            optimizer: a.optimizer,
            measurements: a_measurements,
        },
        b: AbVariantResult {
            label: b.label,
            config: b.config,
            normalization: b.normalization,
            position: b.position,
            optimizer: b.optimizer,
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
    for (label, cfg, position) in [
        (&a.label, a.config, a.position),
        (&b.label, b.config, b.position),
    ] {
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
        if position == PositionKind::Rope && (cfg.n_embd / cfg.n_head) % 2 != 0 {
            return Err(format!("A/B RoPE requires even head width for {label}"));
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
    run_variant_with_kv_heads(
        variant,
        variant.config.n_head,
        protocol,
        tokens,
        repetition,
    )
}

fn run_variant_with_kv_heads(
    variant: &AbVariant,
    n_kv_head: usize,
    protocol: AbProtocol,
    tokens: &[usize],
    repetition: usize,
) -> Result<AbMeasurement, String> {
    run_variant_with_attention_policy(
        variant,
        n_kv_head,
        0,
        protocol,
        tokens,
        repetition,
    )
}

fn run_variant_with_attention_policy(
    variant: &AbVariant,
    n_kv_head: usize,
    attention_window: usize,
    protocol: AbProtocol,
    tokens: &[usize],
    repetition: usize,
) -> Result<AbMeasurement, String> {
    let mut rng = StdRng::seed_from_u64(protocol.seed);
    let mut gpt = Gpt::new_with_attention_policy(
        variant.config,
        variant.normalization,
        variant.position,
        n_kv_head,
        attention_window,
        &mut rng,
    );
    let parameter_count = gpt.collect_params().len();
    let mut optimizer = make_optimizer(variant.optimizer, parameter_count, protocol.learning_rate);
    let mut grads = vec![0.0f32; parameter_count];
    let mut workspace = TrainWorkspace::new(&gpt);
    let cfg = protocol.train_config();
    let started = Instant::now();
    let mut processed_tokens = 0u64;
    let mut final_train_loss = f32::NAN;

    for _ in 0..protocol.steps {
        let global_step = optimizer.global_step();
        let metrics = train_step_reuse(
            &mut gpt,
            optimizer.as_mut(),
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
    let state_fingerprint = training_state_fingerprint(
        &params,
        optimizer.as_ref(),
        variant.normalization,
        variant.position,
    );
    let optimizer_state_bytes = optimizer.state_vector_bytes();
    let optimizer_state_serialized_bytes = optimizer
        .canonical_state_text()
        .map_err(|e| format!("A/B optimizer state serialization failed for {}: {e}", variant.label))?
        .len();
    let checkpoint_includes_optimizer = optimizer.legacy_adam().is_some();

    let tok = synthetic_tokenizer(variant.config.vocab)?;
    let path = temporary_checkpoint_path(&variant.label, repetition);
    checkpoint::save_full(&path, &gpt, &tok, optimizer.legacy_adam())
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
        optimizer_state_bytes,
        optimizer_state_serialized_bytes,
        checkpoint_bytes,
        checkpoint_includes_optimizer,
        state_fingerprint,
    })
}

fn make_optimizer(
    kind: OptimizerId,
    parameter_count: usize,
    learning_rate: f32,
) -> Box<dyn Optimizer> {
    match kind {
        OptimizerId::Adam => Box::new(Adam::new(parameter_count, learning_rate)),
        OptimizerId::AdamW => Box::new(AdamW::new(parameter_count, learning_rate, 0.01)),
        OptimizerId::Lion => Box::new(Lion::new(parameter_count, learning_rate, 0.01)),
    }
}

fn training_state_fingerprint(
    params: &[f32],
    optimizer: &dyn Optimizer,
    normalization: NormalizationKind,
    position: PositionKind,
) -> u64 {
    let mut h = fingerprint_params(params);
    for byte in normalization
        .as_str()
        .bytes()
        .chain(position.as_str().bytes())
        .chain(optimizer.id().as_str().bytes())
    {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for byte in optimizer.state_fingerprint().to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
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
    use crate::optim::{AdamWState, LionState};

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
            AbVariant { label: "A".into(), config: tiny(), normalization: NormalizationKind::LayerNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
            AbVariant { label: "B".into(), config: tiny(), normalization: NormalizationKind::LayerNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
        ).unwrap();
        assert!(result.a_vs_a_reproducible());
        assert_eq!(result.a.measurements[0].state_fingerprint, result.a.measurements[1].state_fingerprint);
        assert!(result.json().contains("\"a_vs_a_reproducible\":true"));
        assert!(result.human().contains("comparable_budget=true"));
    }

    #[test]
    fn normalization_variant_is_part_of_experiment_identity() {
        let protocol = AbProtocol {
            steps: 2,
            repeats: 1,
            token_count: 128,
            ..AbProtocol::default()
        };
        let result = run_experiment(
            protocol,
            AbVariant {
                label: "layernorm".into(),
                config: tiny(),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Adam,
            },
            AbVariant {
                label: "rmsnorm".into(),
                config: tiny(),
                normalization: NormalizationKind::RmsNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Adam,
            },
        )
        .unwrap();
        assert!(!result.a_vs_a_reproducible());
        assert_eq!(result.a.config, result.b.config);
        assert_eq!(result.a.measurements[0].parameter_count, result.b.measurements[0].parameter_count);
        assert_eq!(result.a.measurements[0].checkpoint_bytes, result.b.measurements[0].checkpoint_bytes);
        assert_ne!(result.a.measurements[0].state_fingerprint, result.b.measurements[0].state_fingerprint);
        assert!(result.human().contains("normalization=rmsnorm"));
        assert!(result.json().contains("\"normalization\":\"rmsnorm\""));
    }

    #[test]
    fn rope_variant_is_reproducible_and_distinct_from_learned_absolute() {
        let result = run_experiment(
            AbProtocol {
                steps: 2,
                repeats: 2,
                token_count: 128,
                ..AbProtocol::default()
            },
            AbVariant {
                label: "learned".into(),
                config: tiny(),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Adam,
            },
            AbVariant {
                label: "rope".into(),
                config: tiny(),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::Rope,
                optimizer: OptimizerId::Adam,
            },
        )
        .unwrap();
        assert_eq!(result.a.position, PositionKind::LearnedAbsolute);
        assert_eq!(result.b.position, PositionKind::Rope);
        assert!(!result.a_vs_a_reproducible());
        assert_ne!(
            result.a.measurements[0].state_fingerprint,
            result.b.measurements[0].state_fingerprint
        );
        assert!(result.json().contains("\"position\":\"rope\""));
    }

    #[test]
    fn alibi_variant_is_reproducible_and_distinct_from_learned_absolute() {
        let result = run_experiment(
            AbProtocol {
                steps: 2,
                repeats: 2,
                token_count: 128,
                ..AbProtocol::default()
            },
            AbVariant {
                label: "learned".into(),
                config: tiny(),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Adam,
            },
            AbVariant {
                label: "alibi".into(),
                config: tiny(),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::Alibi,
                optimizer: OptimizerId::Adam,
            },
        )
        .unwrap();
        assert_eq!(result.a.position, PositionKind::LearnedAbsolute);
        assert_eq!(result.b.position, PositionKind::Alibi);
        assert!(!result.a_vs_a_reproducible());
        assert_eq!(
            result.a.measurements[0].parameter_count,
            result.b.measurements[0].parameter_count
        );
        assert_eq!(
            result.a.measurements[0].checkpoint_bytes,
            result.b.measurements[0].checkpoint_bytes
        );
        assert_ne!(
            result.a.measurements[0].state_fingerprint,
            result.b.measurements[0].state_fingerprint
        );
        assert!(result.json().contains("\"position\":\"alibi\""));
    }

    #[test]
    fn optimizer_variant_is_part_of_experiment_identity() {
        let result = run_experiment(
            AbProtocol {
                steps: 2,
                repeats: 2,
                token_count: 128,
                ..AbProtocol::default()
            },
            AbVariant {
                label: "adam".into(),
                config: tiny(),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Adam,
            },
            AbVariant {
                label: "lion".into(),
                config: tiny(),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Lion,
            },
        )
        .unwrap();
        assert_eq!(result.a.optimizer, OptimizerId::Adam);
        assert_eq!(result.b.optimizer, OptimizerId::Lion);
        assert!(!result.a_vs_a_reproducible());
        assert_eq!(
            result.b.measurements[0].optimizer_state_bytes * 2,
            result.a.measurements[0].optimizer_state_bytes
        );
        assert!(result.a.measurements[0].checkpoint_includes_optimizer);
        assert!(!result.b.measurements[0].checkpoint_includes_optimizer);
        assert!(result.json().contains("\"optimizer\":\"lion\""));
    }

    fn run_optimizer_steps(
        gpt: &mut Gpt,
        optimizer: &mut dyn Optimizer,
        tokens: &[usize],
        steps: usize,
    ) {
        let n = gpt.collect_params().len();
        let mut grads = vec![0.0f32; n];
        let mut workspace = TrainWorkspace::new(gpt);
        let cfg = TrainConfig {
            seed: 659_918,
            batch_size: 2,
            gradient_accumulation_steps: 2,
            grad_clip_norm: 1.0,
        };
        for _ in 0..steps {
            let global_step = optimizer.global_step();
            train_step_reuse(
                gpt,
                optimizer,
                tokens,
                cfg,
                global_step,
                &mut grads,
                &mut workspace,
            )
            .unwrap();
        }
    }

    fn fresh_resume_model(seed: u64) -> Gpt {
        let mut rng = StdRng::seed_from_u64(seed);
        Gpt::new_with_policies(
            tiny(),
            NormalizationKind::LayerNorm,
            PositionKind::LearnedAbsolute,
            &mut rng,
        )
    }

    #[test]
    fn adamw_training_resume_is_exact() {
        let seed = 7_141;
        let tokens = token_stream(8, 256);

        let mut continuous = fresh_resume_model(seed);
        let n = continuous.collect_params().len();
        let mut continuous_opt = AdamW::new(n, 3e-3, 0.01);
        run_optimizer_steps(&mut continuous, &mut continuous_opt, &tokens, 4);

        let mut split = fresh_resume_model(seed);
        let mut split_opt = AdamW::new(n, 3e-3, 0.01);
        run_optimizer_steps(&mut split, &mut split_opt, &tokens, 2);
        let split_params = split.collect_params();
        let encoded = split_opt.state().encode().unwrap();

        let mut resumed = fresh_resume_model(seed);
        resumed.write_params(&split_params);
        let state = AdamWState::decode(&encoded).unwrap();
        let mut resumed_opt = AdamW::try_from_state(state).unwrap();
        run_optimizer_steps(&mut resumed, &mut resumed_opt, &tokens, 2);

        assert_eq!(resumed.collect_params(), continuous.collect_params());
        assert_eq!(
            resumed_opt.state().encode().unwrap(),
            continuous_opt.state().encode().unwrap()
        );
    }

    #[test]
    fn lion_training_resume_is_exact() {
        let seed = 7_142;
        let tokens = token_stream(8, 256);

        let mut continuous = fresh_resume_model(seed);
        let n = continuous.collect_params().len();
        let mut continuous_opt = Lion::new(n, 3e-3, 0.01);
        run_optimizer_steps(&mut continuous, &mut continuous_opt, &tokens, 4);

        let mut split = fresh_resume_model(seed);
        let mut split_opt = Lion::new(n, 3e-3, 0.01);
        run_optimizer_steps(&mut split, &mut split_opt, &tokens, 2);
        let split_params = split.collect_params();
        let encoded = split_opt.state().encode().unwrap();

        let mut resumed = fresh_resume_model(seed);
        resumed.write_params(&split_params);
        let state = LionState::decode(&encoded).unwrap();
        let mut resumed_opt = Lion::try_from_state(state).unwrap();
        run_optimizer_steps(&mut resumed, &mut resumed_opt, &tokens, 2);

        assert_eq!(resumed.collect_params(), continuous.collect_params());
        assert_eq!(
            resumed_opt.state().encode().unwrap(),
            continuous_opt.state().encode().unwrap()
        );
    }

    #[test]
    fn attention_window_ab_tracks_dense_vs_local_under_identical_budget() {
        let dense = AbVariant {
            label: "dense".into(),
            config: tiny(),
            normalization: NormalizationKind::LayerNorm,
            position: PositionKind::LearnedAbsolute,
            optimizer: OptimizerId::Adam,
        };
        let mut local = dense.clone();
        local.label = "local-w2".into();
        let heads = dense.config.n_head;
        let result = run_attention_window_experiment(
            AbProtocol {
                steps: 2,
                repeats: 2,
                token_count: 128,
                ..AbProtocol::default()
            },
            dense,
            0,
            local,
            2,
            heads,
        )
        .unwrap();

        assert_eq!(result.schema_version, ATTENTION_WINDOW_AB_SCHEMA_VERSION);
        assert_eq!(result.a.attention_window, 0);
        assert_eq!(result.b.attention_window, 2);
        assert_eq!(
            result.a.measurements[0].parameter_count,
            result.b.measurements[0].parameter_count
        );
        assert!(result.a.measurements[0].eval_loss.is_finite());
        assert!(result.b.measurements[0].eval_loss.is_finite());
        assert!(result.human().contains("attention_window=2"));
    }

    #[test]
    fn attention_window_ab_rejects_invalid_or_identical_windows() {
        let dense = AbVariant {
            label: "dense".into(),
            config: tiny(),
            normalization: NormalizationKind::LayerNorm,
            position: PositionKind::LearnedAbsolute,
            optimizer: OptimizerId::Adam,
        };
        let mut local = dense.clone();
        local.label = "local".into();
        let heads = dense.config.n_head;
        let protocol = AbProtocol {
            steps: 1,
            repeats: 1,
            token_count: 64,
            ..AbProtocol::default()
        };
        assert!(run_attention_window_experiment(
            protocol,
            dense.clone(),
            0,
            local.clone(),
            dense.config.block + 1,
            heads,
        )
        .is_err());
        assert!(run_attention_window_experiment(
            protocol,
            dense,
            0,
            local,
            0,
            heads,
        )
        .is_err());
    }

    #[test]
    fn attention_head_ab_tracks_mha_vs_mqa_parameter_and_quality_measurements() {
        let variant = AbVariant {
            label: "mha".into(),
            config: tiny(),
            normalization: NormalizationKind::LayerNorm,
            position: PositionKind::LearnedAbsolute,
            optimizer: OptimizerId::Adam,
        };
        let mut mqa = variant.clone();
        mqa.label = "mqa".into();
        let mha_heads = variant.config.n_head;
        let result = run_attention_head_experiment(
            AbProtocol {
                steps: 2,
                repeats: 2,
                token_count: 128,
                ..AbProtocol::default()
            },
            variant,
            mha_heads,
            mqa,
            1,
        )
        .unwrap();
        assert_eq!(result.schema_version, ATTENTION_HEAD_AB_SCHEMA_VERSION);
        assert_eq!(result.a.n_kv_head, mha_heads);
        assert_eq!(result.b.n_kv_head, 1);
        assert!(
            result.b.measurements[0].parameter_count
                < result.a.measurements[0].parameter_count
        );
        assert!(result.a.measurements[0].eval_loss.is_finite());
        assert!(result.b.measurements[0].eval_loss.is_finite());
        assert!(result.human().contains("n_kv_head=1"));
    }

    #[test]
    fn attention_head_ab_rejects_non_divisible_kv_heads() {
        let variant = AbVariant {
            label: "bad".into(),
            config: tiny(),
            normalization: NormalizationKind::LayerNorm,
            position: PositionKind::LearnedAbsolute,
            optimizer: OptimizerId::Adam,
        };
        let err = run_attention_head_experiment(
            AbProtocol {
                steps: 1,
                repeats: 1,
                token_count: 64,
                ..AbProtocol::default()
            },
            variant.clone(),
            3,
            variant,
            4,
        )
        .unwrap_err();
        assert!(err.contains("KV-head"));
    }

    #[test]
    fn mismatched_vocab_is_rejected() {
        let mut other = tiny();
        other.vocab = 9;
        assert!(run_experiment(
            AbProtocol { steps: 1, repeats: 1, token_count: 64, ..AbProtocol::default() },
            AbVariant { label: "A".into(), config: tiny(), normalization: NormalizationKind::LayerNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
            AbVariant { label: "B".into(), config: other, normalization: NormalizationKind::LayerNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
        ).is_err());
    }
}
