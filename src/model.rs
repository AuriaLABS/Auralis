//! Decoder-only Transformer from first principles.
//!
//! No deep-learning framework is used here: forward pass, causal multi-head
//! attention, layer normalization, GELU, cross-entropy and backward pass are
//! implemented explicitly over `Vec<f32>`.

use crate::arena::Arena;
use crate::attention::{
    local_probability_len, Attention, AttentionShape, GroupedAttentionShape, RowSlicesAttention,
};
use crate::backend::{
    Backend, BackendId, MatrixMut, MatrixRef, OptimizedCpuBackend, ScalarCpuBackend,
};
use crate::kv_cache::KvCache;
use crate::layer_diagnostics::{
    cosine_similarity, summarize_tensor, AdjacentLayerCosine, GradientLayerSummary,
    LayerDiagnosticsReport, LayerHooks, LayerTensorSummary,
};
use crate::memory::ExternalMemory;
use crate::memory_integration::{
    fuse_last_hidden, retrieve_hidden_residual, MemoryInferenceMode, MemoryTrace,
};
use crate::numeric::{explain, scan_f32, Scan, Stage};
use crate::recurrent::RecurrentConfig;
use crate::position::{
    alibi_slopes, Alibi, LearnedAbsolute, PositionKind, PositionalEncoding, Rotary,
    TrainablePositionalEncoding,
};
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block: usize,
    pub n_ff: usize,
}

impl Config {
    pub fn tiny(vocab: usize) -> Self {
        Self {
            vocab,
            n_embd: 32,
            n_head: 4,
            n_layer: 2,
            block: 32,
            n_ff: 96,
        }
    }

    fn validate(&self) {
        assert!(self.vocab > 1, "vocab must contain at least two tokens");
        assert!(self.n_embd > 0 && self.n_head > 0 && self.n_layer > 0);
        assert!(self.block > 0 && self.n_ff > 0);
        assert_eq!(
            self.n_embd % self.n_head,
            0,
            "embedding width must be divisible by number of heads"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalizationKind {
    LayerNorm,
    RmsNorm,
}

impl NormalizationKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LayerNorm => "layernorm",
            Self::RmsNorm => "rmsnorm",
        }
    }
}

impl std::str::FromStr for NormalizationKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "layernorm" => Ok(Self::LayerNorm),
            "rmsnorm" => Ok(Self::RmsNorm),
            other => Err(format!("unknown normalization kind {other}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuBackend {
    Optimized,
    Scalar,
}

static OPTIMIZED_CPU_BACKEND: OptimizedCpuBackend = OptimizedCpuBackend;
static SCALAR_CPU_BACKEND: ScalarCpuBackend = ScalarCpuBackend;
static OPTIMIZED_ATTENTION: RowSlicesAttention = RowSlicesAttention;

impl CpuBackend {
    pub fn id(self) -> BackendId {
        match self {
            Self::Optimized => BackendId::OptimizedCpu,
            Self::Scalar => BackendId::ScalarCpu,
        }
    }
}

#[derive(Clone)]
struct Block {
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

#[derive(Clone)]
pub struct Gpt {
    pub cfg: Config,
    normalization: NormalizationKind,
    position: PositionKind,
    n_kv_head: usize,
    attention_window: usize,
    tok_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    blocks: Vec<Block>,
    ln_f_g: Vec<f32>,
    ln_f_b: Vec<f32>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
}

#[derive(Clone)]
struct LnCache {
    xhat: Vec<f32>,
    inv_std: Vec<f32>,
    rows: usize,
    cols: usize,
}

struct LayerCache {
    ln1: LnCache,
    h1: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    probs: Vec<f32>,
    att: Vec<f32>,
    ln2: LnCache,
    h2: Vec<f32>,
    ff_pre: Vec<f32>,
    ff_act: Vec<f32>,
}

struct ForwardCache {
    layers: Vec<LayerCache>,
    ln_f: LnCache,
    h_final: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForwardTensorSummary {
    pub layer: Option<usize>,
    pub name: &'static str,
    pub stage: Stage,
    pub scan: Scan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForwardDiagnosticsReport {
    pub tensors: Vec<ForwardTensorSummary>,
}

impl ForwardDiagnosticsReport {
    pub fn first_fault(&self) -> Option<&ForwardTensorSummary> {
        self.tensors.iter().find(|summary| !summary.scan.is_finite())
    }

    pub fn first_fault_context(&self) -> Option<String> {
        let summary = self.first_fault()?;
        let scope = match summary.layer {
            Some(layer) => format!("layer[{layer}].{}", summary.name),
            None => summary.name.to_string(),
        };
        Some(explain(summary.stage, &scope, &summary.scan))
    }
}

#[derive(Debug)]
struct BlockGrad {
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

#[derive(Debug)]
struct GptGrad {
    tok_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    blocks: Vec<BlockGrad>,
    ln_f_g: Vec<f32>,
    ln_f_b: Vec<f32>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
}

#[derive(Debug)]
pub(crate) struct BackwardWorkspace {
    cfg: Config,
    n_kv_head: usize,
    attention_window: usize,
    grads: GptGrad,
    scratch: Arena,
    dx: Vec<f32>,
    residual: Vec<f32>,
}

fn init_vec(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

fn init_compact_projection(
    rng: &mut impl Rng,
    input_width: usize,
    full_output_width: usize,
    compact_output_width: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(compact_output_width > 0 && compact_output_width <= full_output_width);
    let full = init_vec(
        rng,
        input_width
            .checked_mul(full_output_width)
            .expect("projection initialization size overflow"),
        scale,
    );
    if compact_output_width == full_output_width {
        return full;
    }
    let mut compact = Vec::with_capacity(
        input_width
            .checked_mul(compact_output_width)
            .expect("compact projection initialization size overflow"),
    );
    for row in 0..input_width {
        let start = row * full_output_width;
        compact.extend_from_slice(&full[start..start + compact_output_width]);
    }
    compact
}

fn zeros_block_grad(cfg: Config, n_kv_head: usize) -> BlockGrad {
    let d = cfg.n_embd;
    let head_width = d / cfg.n_head;
    let kv_width = head_width * n_kv_head;
    BlockGrad {
        ln1_g: vec![0.0; d],
        ln1_b: vec![0.0; d],
        wq: vec![0.0; d * d],
        wk: vec![0.0; d * kv_width],
        wv: vec![0.0; d * kv_width],
        wo: vec![0.0; d * d],
        ln2_g: vec![0.0; d],
        ln2_b: vec![0.0; d],
        w1: vec![0.0; d * cfg.n_ff],
        b1: vec![0.0; cfg.n_ff],
        w2: vec![0.0; cfg.n_ff * d],
        b2: vec![0.0; d],
    }
}

impl BlockGrad {
    fn clear(&mut self) {
        self.ln1_g.fill(0.0);
        self.ln1_b.fill(0.0);
        self.wq.fill(0.0);
        self.wk.fill(0.0);
        self.wv.fill(0.0);
        self.wo.fill(0.0);
        self.ln2_g.fill(0.0);
        self.ln2_b.fill(0.0);
        self.w1.fill(0.0);
        self.b1.fill(0.0);
        self.w2.fill(0.0);
        self.b2.fill(0.0);
    }
}

impl GptGrad {
    fn clear(&mut self) {
        self.tok_emb.fill(0.0);
        self.pos_emb.fill(0.0);
        for block in &mut self.blocks {
            block.clear();
        }
        self.ln_f_g.fill(0.0);
        self.ln_f_b.fill(0.0);
        self.w_out.fill(0.0);
        self.b_out.fill(0.0);
    }
}

impl BackwardWorkspace {
    pub(crate) fn new(gpt: &Gpt) -> Self {
        let td = gpt
            .cfg
            .block
            .checked_mul(gpt.cfg.n_embd)
            .expect("backward workspace size overflow");
        let attention_scratch = td
            .checked_mul(3)
            .and_then(|n| n.checked_add(gpt.cfg.block))
            .expect("attention scratch size overflow");
        let ffn_scratch = gpt
            .cfg
            .block
            .checked_mul(gpt.cfg.n_ff)
            .expect("ffn scratch size overflow");
        let layernorm_scratch = td
            .checked_add(
                gpt.cfg
                    .n_embd
                    .checked_mul(2)
                    .expect("layernorm scratch size overflow"),
            )
            .expect("layernorm scratch size overflow");
        let scratch_capacity = attention_scratch.max(ffn_scratch).max(layernorm_scratch);

        Self {
            cfg: gpt.cfg,
            n_kv_head: gpt.n_kv_head,
            attention_window: gpt.attention_window,
            grads: gpt.zero_grads(),
            scratch: Arena::with_capacity(scratch_capacity),
            dx: vec![0.0; td],
            residual: vec![0.0; td],
        }
    }

    pub(crate) fn matches(&self, gpt: &Gpt) -> bool {
        self.cfg == gpt.cfg
            && self.n_kv_head == gpt.n_kv_head
            && self.attention_window == gpt.attention_window
    }

    fn clear(&mut self) {
        self.grads.clear();
        self.scratch.reset();
    }
}

impl Gpt {
    pub fn new(cfg: Config, rng: &mut impl Rng) -> Self {
        Self::new_with_attention_heads(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::LearnedAbsolute,
            cfg.n_head,
            rng,
        )
    }

    pub fn new_with_normalization(
        cfg: Config,
        normalization: NormalizationKind,
        rng: &mut impl Rng,
    ) -> Self {
        Self::new_with_attention_heads(
            cfg,
            normalization,
            PositionKind::LearnedAbsolute,
            cfg.n_head,
            rng,
        )
    }

    pub fn new_with_policies(
        cfg: Config,
        normalization: NormalizationKind,
        position: PositionKind,
        rng: &mut impl Rng,
    ) -> Self {
        Self::new_with_attention_heads(cfg, normalization, position, cfg.n_head, rng)
    }

    pub fn new_with_attention_heads(
        cfg: Config,
        normalization: NormalizationKind,
        position: PositionKind,
        n_kv_head: usize,
        rng: &mut impl Rng,
    ) -> Self {
        Self::new_with_attention_policy(
            cfg,
            normalization,
            position,
            n_kv_head,
            0,
            rng,
        )
    }

    pub fn new_with_attention_policy(
        cfg: Config,
        normalization: NormalizationKind,
        position: PositionKind,
        n_kv_head: usize,
        attention_window: usize,
        rng: &mut impl Rng,
    ) -> Self {
        cfg.validate();
        assert!(
            n_kv_head > 0 && n_kv_head <= cfg.n_head && cfg.n_head % n_kv_head == 0,
            "n_kv_head must divide n_head"
        );
        assert!(
            attention_window <= cfg.block,
            "attention_window must be 0 (dense) or <= block"
        );
        if position == PositionKind::Rope {
            Rotary::new(cfg.block, cfg.n_embd, cfg.n_head)
                .expect("RoPE requires even per-head width");
        }
        let d = cfg.n_embd;
        let kv_width = (d / cfg.n_head) * n_kv_head;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            blocks.push(Block {
                ln1_g: vec![1.0; d],
                ln1_b: vec![0.0; d],
                wq: init_vec(rng, d * d, 0.02),
                wk: init_compact_projection(rng, d, d, kv_width, 0.02),
                wv: init_compact_projection(rng, d, d, kv_width, 0.02),
                wo: init_vec(rng, d * d, 0.02),
                ln2_g: vec![1.0; d],
                ln2_b: vec![0.0; d],
                w1: init_vec(rng, d * cfg.n_ff, 0.02),
                b1: vec![0.0; cfg.n_ff],
                w2: init_vec(rng, cfg.n_ff * d, 0.02),
                b2: vec![0.0; d],
            });
        }
        Self {
            cfg,
            normalization,
            position,
            n_kv_head,
            attention_window,
            tok_emb: init_vec(rng, cfg.vocab * d, 0.02),
            pos_emb: init_vec(rng, cfg.block * d, 0.02),
            blocks,
            ln_f_g: vec![1.0; d],
            ln_f_b: vec![0.0; d],
            w_out: init_vec(rng, d * cfg.vocab, 0.02),
            b_out: vec![0.0; cfg.vocab],
        }
    }

    pub fn normalization(&self) -> NormalizationKind {
        self.normalization
    }

    pub fn set_normalization(&mut self, normalization: NormalizationKind) {
        self.normalization = normalization;
    }

    pub fn position_kind(&self) -> PositionKind {
        self.position
    }

    pub fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    pub fn attention_window(&self) -> usize {
        self.attention_window
    }

    pub fn set_attention_window(&mut self, attention_window: usize) -> Result<(), String> {
        if attention_window > self.cfg.block {
            return Err(format!(
                "attention_window {attention_window} exceeds block {}",
                self.cfg.block
            ));
        }
        self.attention_window = attention_window;
        Ok(())
    }

    pub fn kv_width(&self) -> usize {
        (self.cfg.n_embd / self.cfg.n_head) * self.n_kv_head
    }

    pub fn expected_parameter_count(
        cfg: Config,
        n_kv_head: usize,
    ) -> Result<usize, String> {
        if cfg.vocab <= 1
            || cfg.n_embd == 0
            || cfg.n_head == 0
            || cfg.n_layer == 0
            || cfg.block == 0
            || cfg.n_ff == 0
            || cfg.n_embd % cfg.n_head != 0
            || n_kv_head == 0
            || n_kv_head > cfg.n_head
            || cfg.n_head % n_kv_head != 0
        {
            return Err("invalid model/KV-head shape for parameter count".into());
        }
        let d = cfg.n_embd;
        let kv_width = (d / cfg.n_head)
            .checked_mul(n_kv_head)
            .ok_or_else(|| "KV width overflow".to_string())?;
        let per_block = d
            .checked_mul(d)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(d.checked_mul(kv_width)?.checked_mul(2)?))
            .and_then(|n| n.checked_add(d.checked_mul(5)?))
            .and_then(|n| n.checked_add(d.checked_mul(cfg.n_ff)?.checked_mul(2)?))
            .and_then(|n| n.checked_add(cfg.n_ff))
            .ok_or_else(|| "block parameter count overflow".to_string())?;

        cfg.vocab
            .checked_mul(d)
            .and_then(|n| n.checked_add(cfg.block.checked_mul(d)?))
            .and_then(|n| n.checked_add(cfg.n_layer.checked_mul(per_block)?))
            .and_then(|n| n.checked_add(d.checked_mul(2)?))
            .and_then(|n| n.checked_add(d.checked_mul(cfg.vocab)?))
            .and_then(|n| n.checked_add(cfg.vocab))
            .ok_or_else(|| "model parameter count overflow".to_string())
    }


    pub fn set_position_kind(&mut self, position: PositionKind) {
        if position == PositionKind::Rope {
            Rotary::new(self.cfg.block, self.cfg.n_embd, self.cfg.n_head)
                .expect("RoPE requires even per-head width");
        }
        self.position = position;
    }

    fn embed_tokens_with_positions(&self, tokens: &[usize]) -> Vec<f32> {
        assert!(!tokens.is_empty() && tokens.len() <= self.cfg.block);
        let t = tokens.len();
        let d = self.cfg.n_embd;
        let mut x = vec![0.0; t * d];

        for i in 0..t {
            let tok = tokens[i];
            assert!(tok < self.cfg.vocab);
            for j in 0..d {
                x[i * d + j] = self.tok_emb[tok * d + j];
            }
        }

        match self.position {
            PositionKind::LearnedAbsolute => {
                let positional = LearnedAbsolute::new(&self.pos_emb, self.cfg.block, d)
                    .expect("model positional storage matches config");
                positional
                    .add_forward(&mut x, t)
                    .expect("token length already validated against positional capacity");
            }
            PositionKind::Rope => {
                let positional = Rotary::new(self.cfg.block, d, self.cfg.n_head)
                    .expect("model RoPE shape matches config");
                positional
                    .add_forward(&mut x, t)
                    .expect("token length already validated against positional capacity");
            }
            PositionKind::Alibi => {
                let positional = Alibi::new(self.cfg.block, d, self.cfg.n_head)
                    .expect("model ALiBi shape matches config");
                positional
                    .add_forward(&mut x, t)
                    .expect("token length already validated against positional capacity");
            }
        }
        x
    }

    fn apply_position_to_qk(&self, q: &mut [f32], k: &mut [f32], positions: usize) {
        if self.n_kv_head == self.cfg.n_head {
            match self.position {
                PositionKind::LearnedAbsolute => {
                    let positional = LearnedAbsolute::new(
                        &self.pos_emb,
                        self.cfg.block,
                        self.cfg.n_embd,
                    )
                    .expect("model positional storage matches config");
                    positional
                        .apply_qk(q, k, positions, self.cfg.n_head)
                        .expect("model Q/K shapes match positional contract");
                }
                PositionKind::Rope => {
                    let positional =
                        Rotary::new(self.cfg.block, self.cfg.n_embd, self.cfg.n_head)
                            .expect("model RoPE shape matches config");
                    positional
                        .apply_qk(q, k, positions, self.cfg.n_head)
                        .expect("model Q/K shapes match RoPE contract");
                }
                PositionKind::Alibi => {
                    let positional =
                        Alibi::new(self.cfg.block, self.cfg.n_embd, self.cfg.n_head)
                            .expect("model ALiBi shape matches config");
                    positional
                        .apply_qk(q, k, positions, self.cfg.n_head)
                        .expect("model Q/K shapes match ALiBi contract");
                }
            }
            return;
        }

        assert_eq!(q.len(), positions * self.cfg.n_embd);
        assert_eq!(k.len(), positions * self.kv_width());
        if self.position == PositionKind::Rope {
            Rotary::new(self.cfg.block, self.cfg.n_embd, self.cfg.n_head)
                .expect("model RoPE shape matches config")
                .apply_grouped_qk(q, k, positions, self.cfg.n_head, self.n_kv_head)
                .expect("grouped model Q/K shapes match RoPE contract");
        }
    }

    fn backward_position_qk(
        &self,
        dq: &mut [f32],
        dk: &mut [f32],
        positions: usize,
    ) {
        if self.n_kv_head == self.cfg.n_head {
            match self.position {
                PositionKind::LearnedAbsolute => {
                    let positional = LearnedAbsolute::new(
                        &self.pos_emb,
                        self.cfg.block,
                        self.cfg.n_embd,
                    )
                    .expect("model positional storage matches config");
                    positional
                        .backward_qk(dq, dk, positions, self.cfg.n_head)
                        .expect("model Q/K gradient shapes match positional contract");
                }
                PositionKind::Rope => {
                    let positional =
                        Rotary::new(self.cfg.block, self.cfg.n_embd, self.cfg.n_head)
                            .expect("model RoPE shape matches config");
                    positional
                        .backward_qk(dq, dk, positions, self.cfg.n_head)
                        .expect("model Q/K gradient shapes match RoPE contract");
                }
                PositionKind::Alibi => {
                    let positional =
                        Alibi::new(self.cfg.block, self.cfg.n_embd, self.cfg.n_head)
                            .expect("model ALiBi shape matches config");
                    positional
                        .backward_qk(dq, dk, positions, self.cfg.n_head)
                        .expect("model Q/K gradient shapes match ALiBi contract");
                }
            }
            return;
        }

        assert_eq!(dq.len(), positions * self.cfg.n_embd);
        assert_eq!(dk.len(), positions * self.kv_width());
        if self.position == PositionKind::Rope {
            Rotary::new(self.cfg.block, self.cfg.n_embd, self.cfg.n_head)
                .expect("model RoPE shape matches config")
                .backward_grouped_qk(
                    dq,
                    dk,
                    positions,
                    self.cfg.n_head,
                    self.n_kv_head,
                )
                .expect("grouped model Q/K gradient shapes match RoPE contract");
        }
    }

    fn attention_eval_current(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        tokens: usize,
    ) -> Vec<f32> {
        if self.attention_window == 0 && self.n_kv_head == self.cfg.n_head {
            return attention_eval(
                q,
                k,
                v,
                tokens,
                self.cfg.n_embd,
                self.cfg.n_head,
                self.position,
            );
        }
        let shape = GroupedAttentionShape {
            tokens,
            width: self.cfg.n_embd,
            heads: self.cfg.n_head,
            kv_heads: self.n_kv_head,
        };
        let mut out = vec![0.0; tokens * self.cfg.n_embd];
        let mut scores = vec![0.0; tokens];
        let slopes = if self.position == PositionKind::Alibi {
            Some(alibi_slopes(self.cfg.n_head).expect("validated ALiBi heads"))
        } else {
            None
        };
        if self.attention_window == 0 {
            OPTIMIZED_ATTENTION
                .forward_eval_grouped(
                    q,
                    k,
                    v,
                    shape,
                    slopes.as_deref(),
                    &mut out,
                    &mut scores,
                )
                .expect("grouped attention eval shapes match model");
        } else {
            OPTIMIZED_ATTENTION
                .forward_eval_grouped_local(
                    q,
                    k,
                    v,
                    shape,
                    self.attention_window,
                    slopes.as_deref(),
                    &mut out,
                    &mut scores,
                )
                .expect("local grouped attention eval shapes match model");
        }
        out
    }

    fn attention_forward_current(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        tokens: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        if self.attention_window == 0 && self.n_kv_head == self.cfg.n_head {
            return attention_forward(
                q,
                k,
                v,
                tokens,
                self.cfg.n_embd,
                self.cfg.n_head,
                self.position,
            );
        }
        let shape = GroupedAttentionShape {
            tokens,
            width: self.cfg.n_embd,
            heads: self.cfg.n_head,
            kv_heads: self.n_kv_head,
        };
        let mut out = vec![0.0; tokens * self.cfg.n_embd];
        let probs_len = if self.attention_window == 0 {
            shape.probs_len().expect("validated grouped attention shape")
        } else {
            local_probability_len(shape, self.attention_window)
                .expect("validated local attention shape")
        };
        let mut probs = vec![0.0; probs_len];
        let slopes = if self.position == PositionKind::Alibi {
            Some(alibi_slopes(self.cfg.n_head).expect("validated ALiBi heads"))
        } else {
            None
        };
        if self.attention_window == 0 {
            OPTIMIZED_ATTENTION
                .forward_cached_grouped(
                    q,
                    k,
                    v,
                    shape,
                    slopes.as_deref(),
                    &mut out,
                    &mut probs,
                )
                .expect("grouped attention forward shapes match model");
        } else {
            OPTIMIZED_ATTENTION
                .forward_cached_grouped_local(
                    q,
                    k,
                    v,
                    shape,
                    self.attention_window,
                    slopes.as_deref(),
                    &mut out,
                    &mut probs,
                )
                .expect("local grouped attention forward shapes match model");
        }
        (out, probs)
    }

    fn attention_backward_current(
        &self,
        dout: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        probs: &[f32],
        tokens: usize,
        dq: &mut [f32],
        dk: &mut [f32],
        dv: &mut [f32],
        dp: &mut [f32],
    ) {
        if self.attention_window == 0 && self.n_kv_head == self.cfg.n_head {
            attention_backward_into(
                dout,
                q,
                k,
                v,
                probs,
                tokens,
                self.cfg.n_embd,
                self.cfg.n_head,
                dq,
                dk,
                dv,
                dp,
            );
            return;
        }
        let shape = GroupedAttentionShape {
            tokens,
            width: self.cfg.n_embd,
            heads: self.cfg.n_head,
            kv_heads: self.n_kv_head,
        };
        if self.attention_window == 0 {
            OPTIMIZED_ATTENTION
                .backward_grouped(
                    dout, q, k, v, probs, shape, dq, dk, dv, dp,
                )
                .expect("grouped attention backward shapes match model");
        } else {
            OPTIMIZED_ATTENTION
                .backward_grouped_local(
                    dout,
                    q,
                    k,
                    v,
                    probs,
                    shape,
                    self.attention_window,
                    dq,
                    dk,
                    dv,
                    dp,
                )
                .expect("local grouped attention backward shapes match model");
        }
    }

    pub fn collect_params(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.param_count());
        out.extend_from_slice(&self.tok_emb);
        out.extend_from_slice(&self.pos_emb);
        for b in &self.blocks {
            out.extend_from_slice(&b.ln1_g);
            out.extend_from_slice(&b.ln1_b);
            out.extend_from_slice(&b.wq);
            out.extend_from_slice(&b.wk);
            out.extend_from_slice(&b.wv);
            out.extend_from_slice(&b.wo);
            out.extend_from_slice(&b.ln2_g);
            out.extend_from_slice(&b.ln2_b);
            out.extend_from_slice(&b.w1);
            out.extend_from_slice(&b.b1);
            out.extend_from_slice(&b.w2);
            out.extend_from_slice(&b.b2);
        }
        out.extend_from_slice(&self.ln_f_g);
        out.extend_from_slice(&self.ln_f_b);
        out.extend_from_slice(&self.w_out);
        out.extend_from_slice(&self.b_out);
        out
    }

    pub fn write_params(&mut self, params: &[f32]) {
        assert_eq!(params.len(), self.param_count(), "parameter count mismatch");
        let mut p = 0usize;
        copy_param(&mut self.tok_emb, params, &mut p);
        copy_param(&mut self.pos_emb, params, &mut p);
        for b in &mut self.blocks {
            copy_param(&mut b.ln1_g, params, &mut p);
            copy_param(&mut b.ln1_b, params, &mut p);
            copy_param(&mut b.wq, params, &mut p);
            copy_param(&mut b.wk, params, &mut p);
            copy_param(&mut b.wv, params, &mut p);
            copy_param(&mut b.wo, params, &mut p);
            copy_param(&mut b.ln2_g, params, &mut p);
            copy_param(&mut b.ln2_b, params, &mut p);
            copy_param(&mut b.w1, params, &mut p);
            copy_param(&mut b.b1, params, &mut p);
            copy_param(&mut b.w2, params, &mut p);
            copy_param(&mut b.b2, params, &mut p);
        }
        copy_param(&mut self.ln_f_g, params, &mut p);
        copy_param(&mut self.ln_f_b, params, &mut p);
        copy_param(&mut self.w_out, params, &mut p);
        copy_param(&mut self.b_out, params, &mut p);
        debug_assert_eq!(p, params.len());
    }

    /// Compute vocabulary logits for every input position without constructing
    /// backward-only caches. The returned tensor is row-major `[tokens, vocab]`.
    pub fn logits(&self, tokens: &[usize]) -> Vec<f32> {
        self.forward_eval_with_backend(tokens, &OPTIMIZED_CPU_BACKEND)
    }

    /// Create an empty per-session KV cache matching this model.
    pub fn new_kv_cache(&self) -> KvCache {
        KvCache::new_with_heads(
            self.cfg.n_layer,
            self.kv_width(),
            self.cfg.n_head,
            self.n_kv_head,
            self.cfg.block,
            self.position,
        )
        .expect("validated model config produces a valid KV cache")
    }

    /// Fill an empty KV cache from a causal prefix and return logits for every
    /// prefix position in the same row-major layout as `logits()`.
    pub fn prefill_kv_cache(
        &self,
        tokens: &[usize],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, String> {
        cache.validate_for_heads(
            self.cfg.n_layer,
            self.kv_width(),
            self.cfg.n_head,
            self.n_kv_head,
            self.cfg.block,
            self.position,
        )?;
        if !cache.is_empty() {
            return Err("kv cache prefill requires an empty cache".into());
        }
        if tokens.is_empty() {
            return Err("kv cache prefill requires at least one token".into());
        }
        if tokens.len() > self.cfg.block {
            return Err(format!(
                "kv cache prefill length {} exceeds model block {}",
                tokens.len(),
                self.cfg.block
            ));
        }
        if tokens.iter().any(|&token| token >= self.cfg.vocab) {
            return Err("kv cache prefill token id outside vocabulary".into());
        }

        let mut logits = Vec::with_capacity(tokens.len().saturating_mul(self.cfg.vocab));
        for &token in tokens {
            logits.extend(self.decode_kv_cached(token, cache)?);
        }
        Ok(logits)
    }

    /// Append one token to an existing KV-cache session and return only that
    /// token's vocabulary-logit row.
    pub fn decode_kv_cached(
        &self,
        token: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, String> {
        cache.validate_for_heads(
            self.cfg.n_layer,
            self.kv_width(),
            self.cfg.n_head,
            self.n_kv_head,
            self.cfg.block,
            self.position,
        )?;
        if token >= self.cfg.vocab {
            return Err(format!(
                "kv cache token id {token} outside vocabulary {}",
                self.cfg.vocab
            ));
        }
        if cache.len() >= self.cfg.block {
            return Err(format!(
                "kv cache context full at {} tokens (block={})",
                cache.len(),
                self.cfg.block
            ));
        }
        self.forward_decode_one_with_backend(token, cache, &OPTIMIZED_CPU_BACKEND)
    }

    /// Explicit CPU backend selection for verification/benchmarking.
    ///
    /// This choice is ephemeral and is not part of RunConfig/checkpoints.
    pub fn logits_with_cpu_backend(&self, tokens: &[usize], backend: CpuBackend) -> Vec<f32> {
        match backend {
            CpuBackend::Optimized => {
                self.forward_eval_with_backend(tokens, &OPTIMIZED_CPU_BACKEND)
            }
            CpuBackend::Scalar => self.forward_eval_with_backend(tokens, &SCALAR_CPU_BACKEND),
        }
    }

    /// Optional external-memory inference path.
    ///
    /// Memory-off delegates directly to logits(). Memory-on never writes to
    /// the memory backend: it retrieves a hidden-space residual and fuses it
    /// only into the final hidden row before the output projection.
    pub fn logits_with_memory(
        &self,
        tokens: &[usize],
        memory: Option<&dyn ExternalMemory>,
        mode: &MemoryInferenceMode,
    ) -> Result<(Vec<f32>, MemoryTrace), String> {
        let retrieval = retrieve_hidden_residual(memory, mode, self.cfg.n_embd)?;

        if retrieval.residual.is_none() {
            return Ok((self.logits(tokens), retrieval.trace));
        }

        let fusion = match mode {
            MemoryInferenceMode::On { fusion, .. } => *fusion,
            MemoryInferenceMode::Off => {
                return Ok((self.logits(tokens), retrieval.trace));
            }
        };

        let mut h_final =
            self.forward_eval_hidden_with_backend(tokens, &OPTIMIZED_CPU_BACKEND);
        fuse_last_hidden(
            &mut h_final,
            tokens.len(),
            self.cfg.n_embd,
            retrieval.residual.as_deref().expect("checked above"),
            fusion,
        )?;
        let logits =
            self.project_logits_with_backend(&h_final, tokens.len(), &OPTIMIZED_CPU_BACKEND);
        Ok((logits, retrieval.trace))
    }

    /// Debug-only forward pass with compact numerical summaries for cached
    /// activations. The normal `logits()` / `forward_eval()` path remains
    /// untouched and pays no diagnostics branch or scan cost.
    pub fn forward_diagnostics(
        &self,
        tokens: &[usize],
    ) -> Result<(Vec<f32>, ForwardDiagnosticsReport), String> {
        let (logits, cache) = self.forward_internal(tokens);
        let mut tensors = Vec::with_capacity(cache.layers.len() * 9 + 2);

        fn push(
            tensors: &mut Vec<ForwardTensorSummary>,
            layer: Option<usize>,
            name: &'static str,
            stage: Stage,
            values: &[f32],
        ) {
            tensors.push(ForwardTensorSummary {
                layer,
                name,
                stage,
                scan: scan_f32(values),
            });
        }

        for (layer, cache) in cache.layers.iter().enumerate() {
            push(&mut tensors, Some(layer), "h1", Stage::Activation, &cache.h1);
            push(&mut tensors, Some(layer), "q", Stage::Activation, &cache.q);
            push(&mut tensors, Some(layer), "k", Stage::Activation, &cache.k);
            push(&mut tensors, Some(layer), "v", Stage::Activation, &cache.v);
            push(
                &mut tensors,
                Some(layer),
                "probs",
                Stage::Activation,
                &cache.probs,
            );
            push(&mut tensors, Some(layer), "att", Stage::Activation, &cache.att);
            push(&mut tensors, Some(layer), "h2", Stage::Activation, &cache.h2);
            push(
                &mut tensors,
                Some(layer),
                "ff_pre",
                Stage::Activation,
                &cache.ff_pre,
            );
            push(
                &mut tensors,
                Some(layer),
                "ff_act",
                Stage::Activation,
                &cache.ff_act,
            );
        }
        push(
            &mut tensors,
            None,
            "h_final",
            Stage::Activation,
            &cache.h_final,
        );
        push(&mut tensors, None, "logits", Stage::Logits, &logits);

        let report = ForwardDiagnosticsReport { tensors };
        if let Some(context) = report.first_fault_context() {
            return Err(context);
        }
        Ok((logits, report))
    }

    /// Mean next-token cross-entropy without constructing or propagating
    /// parameter gradients. This is the canonical evaluation path.
    pub fn loss(&self, x: &[usize], y: &[usize]) -> f32 {
        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty() && x.len() <= self.cfg.block);
        assert!(x.iter().all(|&t| t < self.cfg.vocab));
        assert!(y.iter().all(|&t| t < self.cfg.vocab));

        let logits = self.logits(x);
        let t = x.len();
        let v = self.cfg.vocab;
        let mut loss = 0.0f32;

        for i in 0..t {
            let row = &logits[i * v..(i + 1) * v];
            let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &logit in row {
                sum += (logit - maxv).exp();
            }
            let p = ((row[y[i]] - maxv).exp() / sum.max(1e-20)).max(1e-20);
            loss -= p.ln();
        }

        loss / t as f32
    }

    /// Evaluation loss through an explicit CPU backend, used for equivalence
    /// checks without changing the persistent experiment configuration.
    pub fn loss_with_cpu_backend(
        &self,
        x: &[usize],
        y: &[usize],
        backend: CpuBackend,
    ) -> f32 {
        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty() && x.len() <= self.cfg.block);
        assert!(x.iter().all(|&t| t < self.cfg.vocab));
        assert!(y.iter().all(|&t| t < self.cfg.vocab));

        let logits = self.logits_with_cpu_backend(x, backend);
        let t = x.len();
        let v = self.cfg.vocab;
        let mut loss = 0.0f32;
        for i in 0..t {
            let row = &logits[i * v..(i + 1) * v];
            let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &logit in row {
                sum += (logit - maxv).exp();
            }
            let p = ((row[y[i]] - maxv).exp() / sum.max(1e-20)).max(1e-20);
            loss -= p.ln();
        }
        loss / t as f32
    }

    /// Experimental recurrent inference with shared physical block weights.
    ///
    /// reasoning_steps=1 delegates to the exact baseline logits path.
    pub fn logits_recurrent(
        &self,
        tokens: &[usize],
        config: RecurrentConfig,
    ) -> Result<Vec<f32>, String> {
        let config = config.validate()?;
        if config.reasoning_steps == 1 {
            return Ok(self.logits(tokens));
        }
        Ok(self.forward_eval_recurrent_with_backend(
            tokens,
            &OPTIMIZED_CPU_BACKEND,
            config.reasoning_steps,
        ))
    }

    pub fn loss_recurrent(
        &self,
        x: &[usize],
        y: &[usize],
        config: RecurrentConfig,
    ) -> Result<f32, String> {
        let config = config.validate()?;
        if config.reasoning_steps == 1 {
            return Ok(self.loss(x, y));
        }
        if x.len() != y.len() {
            return Err("recurrent loss input/target length mismatch".into());
        }
        if x.is_empty() || x.len() > self.cfg.block {
            return Err("recurrent loss token length outside model block".into());
        }
        if x.iter().any(|&token| token >= self.cfg.vocab)
            || y.iter().any(|&token| token >= self.cfg.vocab)
        {
            return Err("recurrent loss token id outside vocabulary".into());
        }

        let logits = self.forward_eval_recurrent_with_backend(
            x,
            &OPTIMIZED_CPU_BACKEND,
            config.reasoning_steps,
        );
        Ok(mean_cross_entropy(&logits, y, self.cfg.vocab))
    }

    /// Experimental recurrent backward pass.
    ///
    /// Shared physical block parameters accumulate gradient contributions from
    /// every recurrent application. reasoning_steps=1 delegates to baseline.
    pub fn backward_recurrent_into(
        &self,
        x: &[usize],
        y: &[usize],
        grads: &mut [f32],
        config: RecurrentConfig,
    ) -> Result<f32, String> {
        let config = config.validate()?;
        if config.reasoning_steps == 1 {
            return Ok(self.backward_into(x, y, grads));
        }
        let mut workspace = BackwardWorkspace::new(self);
        self.backward_recurrent_into_reuse(
            x,
            y,
            grads,
            &mut workspace,
            config.reasoning_steps,
        )
    }

    pub fn backward_into(&self, x: &[usize], y: &[usize], grads: &mut [f32]) -> f32 {
        let mut workspace = BackwardWorkspace::new(self);
        self.backward_into_reuse(x, y, grads, &mut workspace)
    }

    pub fn backward_with_layer_diagnostics(
        &self,
        x: &[usize],
        y: &[usize],
        grads: &mut [f32],
        hooks: LayerHooks,
    ) -> (f32, Option<LayerDiagnosticsReport>) {
        if !hooks.enabled {
            return (self.backward_into(x, y, grads), None);
        }

        let (_logits, cache) = self.forward_internal(x);
        let mut activations = Vec::with_capacity(cache.layers.len() * 9);

        fn push(
            activations: &mut Vec<LayerTensorSummary>,
            layer: usize,
            name: &'static str,
            values: &[f32],
            histogram: bool,
        ) {
            activations.push(LayerTensorSummary {
                layer,
                name,
                stage: Stage::Activation,
                scan: scan_f32(values),
                stats: summarize_tensor(values, histogram),
            });
        }

        for (layer, cache) in cache.layers.iter().enumerate() {
            push(&mut activations, layer, "h1", &cache.h1, hooks.histogram);
            push(&mut activations, layer, "q", &cache.q, hooks.histogram);
            push(&mut activations, layer, "k", &cache.k, hooks.histogram);
            push(&mut activations, layer, "v", &cache.v, hooks.histogram);
            push(&mut activations, layer, "probs", &cache.probs, hooks.histogram);
            push(&mut activations, layer, "att", &cache.att, hooks.histogram);
            push(&mut activations, layer, "h2", &cache.h2, hooks.histogram);
            push(&mut activations, layer, "ff_pre", &cache.ff_pre, hooks.histogram);
            push(&mut activations, layer, "ff_act", &cache.ff_act, hooks.histogram);
        }

        let loss = self.backward_into(x, y, grads);
        let block_len = self.block_param_count();
        let block_base = self.tok_emb.len() + self.pos_emb.len();
        let mut gradients = Vec::with_capacity(self.cfg.n_layer);

        for layer in 0..self.cfg.n_layer {
            let start = block_base + layer * block_len;
            let end = start + block_len;
            gradients.push(GradientLayerSummary {
                layer,
                stats: summarize_tensor(&grads[start..end], hooks.histogram),
            });
        }

        let mut adjacent_gradient_cosine = Vec::new();
        if hooks.adjacent_cosine {
            adjacent_gradient_cosine.reserve(self.cfg.n_layer.saturating_sub(1));
            for layer in 0..self.cfg.n_layer.saturating_sub(1) {
                let left_start = block_base + layer * block_len;
                let right_start = left_start + block_len;
                let left = &grads[left_start..left_start + block_len];
                let right = &grads[right_start..right_start + block_len];
                adjacent_gradient_cosine.push(AdjacentLayerCosine {
                    left_layer: layer,
                    right_layer: layer + 1,
                    cosine: cosine_similarity(left, right),
                });
            }
        }

        (
            loss,
            Some(LayerDiagnosticsReport {
                activations,
                gradients,
                adjacent_gradient_cosine,
            }),
        )
    }

    pub(crate) fn backward_into_reuse(
        &self,
        x: &[usize],
        y: &[usize],
        grads: &mut [f32],
        workspace: &mut BackwardWorkspace,
    ) -> f32 {
        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty() && x.len() <= self.cfg.block);
        assert_eq!(grads.len(), self.param_count());
        assert!(x.iter().all(|&t| t < self.cfg.vocab));
        assert!(y.iter().all(|&t| t < self.cfg.vocab));
        assert!(workspace.matches(self), "backward workspace config mismatch");

        let (mut dlogits, cache) = self.forward_internal(x);
        let t = x.len();
        let v = self.cfg.vocab;
        let mut loss = 0.0f32;

        for i in 0..t {
            let start = i * v;
            let end = start + v;
            let maxv = dlogits[start..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..v {
                let idx = start + j;
                let e = (dlogits[idx] - maxv).exp();
                dlogits[idx] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-20);
            for j in 0..v {
                dlogits[start + j] *= inv;
            }
            let target = start + y[i];
            let p = dlogits[target].max(1e-20);
            loss -= p.ln();
            dlogits[target] -= 1.0;
        }
        let inv_t = 1.0 / t as f32;
        loss *= inv_t;
        for g in &mut dlogits {
            *g *= inv_t;
        }

        workspace.clear();
        let BackwardWorkspace {
            grads: gg,
            scratch,
            dx,
            residual,
            ..
        } = workspace;
        let d = self.cfg.n_embd;
        let td = t * d;
        let dx = &mut dx[..td];
        let residual = &mut residual[..td];

        matmul_grad_b(&cache.h_final, t, d, &dlogits, v, &mut gg.w_out);
        sum_rows_into(&dlogits, t, v, &mut gg.b_out);
        matmul_b_t_into(&dlogits, t, v, &self.w_out, d, dx);

        scratch.reset();
        let dx_ln_slot = scratch.alloc(td);
        let dgamma_slot = scratch.alloc(d);
        let dbeta_slot = scratch.alloc(d);
        {
            let (dx_ln, dgamma, dbeta) =
                scratch.get3_mut(dx_ln_slot, dgamma_slot, dbeta_slot);
            normalization_backward_into(self.normalization, dx, &cache.ln_f, &self.ln_f_g, dx_ln, dgamma, dbeta);
            add_inplace(&mut gg.ln_f_g, dgamma);
            add_inplace(&mut gg.ln_f_b, dbeta);
            dx.copy_from_slice(dx_ln);
        }

        for li in (0..self.blocks.len()).rev() {
            let b = &self.blocks[li];
            let c = &cache.layers[li];
            let bg = &mut gg.blocks[li];

            let dff_out = &dx[..];
            matmul_grad_b(&c.ff_act, t, self.cfg.n_ff, dff_out, d, &mut bg.w2);
            sum_rows_into(dff_out, t, d, &mut bg.b2);

            scratch.reset();
            let dff_pre_slot = scratch.alloc(t * self.cfg.n_ff);
            {
                let dff_pre = scratch.get_mut(dff_pre_slot);
                matmul_b_t_into(dff_out, t, d, &b.w2, self.cfg.n_ff, dff_pre);
                for i in 0..dff_pre.len() {
                    dff_pre[i] *= gelu_deriv(c.ff_pre[i]);
                }

                matmul_grad_b(&c.h2, t, d, dff_pre, self.cfg.n_ff, &mut bg.w1);
                sum_rows_into(dff_pre, t, self.cfg.n_ff, &mut bg.b1);
                matmul_b_t_into(dff_pre, t, self.cfg.n_ff, &b.w1, d, residual);
            }

            scratch.reset();
            let dln2_slot = scratch.alloc(td);
            let dg2_slot = scratch.alloc(d);
            let db2_slot = scratch.alloc(d);
            {
                let (dln2, dg2, db2) = scratch.get3_mut(dln2_slot, dg2_slot, db2_slot);
                normalization_backward_into(self.normalization, residual, &c.ln2, &b.ln2_g, dln2, dg2, db2);
                add_inplace(&mut bg.ln2_g, dg2);
                add_inplace(&mut bg.ln2_b, db2);
                for i in 0..td {
                    residual[i] = dx[i] + dln2[i];
                }
            }

            let dproj = &residual[..];
            matmul_grad_b(&c.att, t, d, dproj, d, &mut bg.wo);
            matmul_b_t_into(dproj, t, d, &b.wo, d, dx);

            scratch.reset();
            let kv_width = self.kv_width();
            let dq_slot = scratch.alloc(td);
            let dk_slot = scratch.alloc(t * kv_width);
            let dv_slot = scratch.alloc(t * kv_width);
            let dp_slot = scratch.alloc(t);
            {
                let (dq, dk, dv, dp) = scratch.get4_mut(dq_slot, dk_slot, dv_slot, dp_slot);
                self.attention_backward_current(
                    dx,
                    &c.q,
                    &c.k,
                    &c.v,
                    &c.probs,
                    t,
                    dq,
                    dk,
                    dv,
                    dp,
                );
                self.backward_position_qk(dq, dk, t);

                matmul_grad_b(&c.h1, t, d, dq, d, &mut bg.wq);
                matmul_grad_b(&c.h1, t, d, dk, kv_width, &mut bg.wk);
                matmul_grad_b(&c.h1, t, d, dv, kv_width, &mut bg.wv);

                matmul_b_t_into(dq, t, d, &b.wq, d, dx);
                matmul_b_t_add_into(dk, t, kv_width, &b.wk, d, dx);
                matmul_b_t_add_into(dv, t, kv_width, &b.wv, d, dx);
            }

            scratch.reset();
            let dln1_slot = scratch.alloc(td);
            let dg1_slot = scratch.alloc(d);
            let db1_slot = scratch.alloc(d);
            {
                let (dln1, dg1, db1) = scratch.get3_mut(dln1_slot, dg1_slot, db1_slot);
                normalization_backward_into(self.normalization, dx, &c.ln1, &b.ln1_g, dln1, dg1, db1);
                add_inplace(&mut bg.ln1_g, dg1);
                add_inplace(&mut bg.ln1_b, db1);
                for i in 0..td {
                    dx[i] = residual[i] + dln1[i];
                }
            }
        }

        for i in 0..t {
            let tok = x[i];
            for j in 0..d {
                let g = dx[i * d + j];
                gg.tok_emb[tok * d + j] += g;
            }
        }

        match self.position {
            PositionKind::LearnedAbsolute => {
                let positional = LearnedAbsolute::new(&self.pos_emb, self.cfg.block, d)
                    .expect("model positional storage matches config");
                positional
                    .accumulate_backward(dx, t, &mut gg.pos_emb)
                    .expect("backward positional shapes match validated forward");
            }
            PositionKind::Rope => {
                let positional = Rotary::new(self.cfg.block, d, self.cfg.n_head)
                    .expect("model RoPE shape matches config");
                positional
                    .accumulate_backward(dx, t, &mut gg.pos_emb)
                    .expect("reserved positional gradient layout matches model");
            }
            PositionKind::Alibi => {
                let positional = Alibi::new(self.cfg.block, d, self.cfg.n_head)
                    .expect("model ALiBi shape matches config");
                positional
                    .accumulate_backward(dx, t, &mut gg.pos_emb)
                    .expect("reserved positional gradient layout matches model");
            }
        }

        copy_grads_into(gg, grads);
        loss
    }

    fn backward_recurrent_into_reuse(
        &self,
        x: &[usize],
        y: &[usize],
        grads: &mut [f32],
        workspace: &mut BackwardWorkspace,
        reasoning_steps: usize,
    ) -> Result<f32, String> {
        if x.len() != y.len() {
            return Err("recurrent backward input/target length mismatch".into());
        }
        if x.is_empty() || x.len() > self.cfg.block {
            return Err("recurrent backward token length outside model block".into());
        }
        if grads.len() != self.param_count() {
            return Err("recurrent backward gradient size mismatch".into());
        }
        if x.iter().any(|&token| token >= self.cfg.vocab)
            || y.iter().any(|&token| token >= self.cfg.vocab)
        {
            return Err("recurrent backward token id outside vocabulary".into());
        }
        if !workspace.matches(self) {
            return Err("recurrent backward workspace config mismatch".into());
        }

        let (mut dlogits, cache) = self.forward_internal_recurrent_with_backend(
            x,
            &OPTIMIZED_CPU_BACKEND,
            reasoning_steps,
        );
        let expected_caches = self
            .blocks
            .len()
            .checked_mul(reasoning_steps)
            .ok_or_else(|| "recurrent cache count overflow".to_string())?;
        if cache.layers.len() != expected_caches {
            return Err("recurrent cache count mismatch".into());
        }

        let t = x.len();
        let v = self.cfg.vocab;
        let mut loss = 0.0f32;
        for i in 0..t {
            let start = i * v;
            let end = start + v;
            let maxv = dlogits[start..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..v {
                let idx = start + j;
                let e = (dlogits[idx] - maxv).exp();
                dlogits[idx] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-20);
            for j in 0..v {
                dlogits[start + j] *= inv;
            }
            let target = start + y[i];
            let p = dlogits[target].max(1e-20);
            loss -= p.ln();
            dlogits[target] -= 1.0;
        }
        let inv_t = 1.0 / t as f32;
        loss *= inv_t;
        for value in &mut dlogits {
            *value *= inv_t;
        }

        workspace.clear();
        let BackwardWorkspace {
            grads: gg,
            scratch,
            dx,
            residual,
            ..
        } = workspace;
        let d = self.cfg.n_embd;
        let td = t * d;
        let dx = &mut dx[..td];
        let residual = &mut residual[..td];

        // Output projection and final normalization execute once after all
        // recurrent applications, so their gradients keep baseline semantics.
        matmul_grad_b(&cache.h_final, t, d, &dlogits, v, &mut gg.w_out);
        sum_rows_into(&dlogits, t, v, &mut gg.b_out);
        matmul_b_t_into(&dlogits, t, v, &self.w_out, d, dx);

        scratch.reset();
        let dx_ln_slot = scratch.alloc(td);
        let dgamma_slot = scratch.alloc(d);
        let dbeta_slot = scratch.alloc(d);
        {
            let (dx_ln, dgamma, dbeta) =
                scratch.get3_mut(dx_ln_slot, dgamma_slot, dbeta_slot);
            normalization_backward_into(
                self.normalization,
                dx,
                &cache.ln_f,
                &self.ln_f_g,
                dx_ln,
                dgamma,
                dbeta,
            );
            add_inplace(&mut gg.ln_f_g, dgamma);
            add_inplace(&mut gg.ln_f_b, dbeta);
            dx.copy_from_slice(dx_ln);
        }

        for cache_index in (0..cache.layers.len()).rev() {
            let li = cache_index % self.blocks.len();
            let b = &self.blocks[li];
            let c = &cache.layers[cache_index];
            let bg = &mut gg.blocks[li];

            let dff_out = &dx[..];
            matmul_grad_b_add_reference(
                &c.ff_act,
                t,
                self.cfg.n_ff,
                dff_out,
                d,
                &mut bg.w2,
            );
            sum_rows_into(dff_out, t, d, &mut bg.b2);

            scratch.reset();
            let dff_pre_slot = scratch.alloc(t * self.cfg.n_ff);
            {
                let dff_pre = scratch.get_mut(dff_pre_slot);
                matmul_b_t_into(dff_out, t, d, &b.w2, self.cfg.n_ff, dff_pre);
                for i in 0..dff_pre.len() {
                    dff_pre[i] *= gelu_deriv(c.ff_pre[i]);
                }

                matmul_grad_b_add_reference(
                    &c.h2,
                    t,
                    d,
                    dff_pre,
                    self.cfg.n_ff,
                    &mut bg.w1,
                );
                sum_rows_into(dff_pre, t, self.cfg.n_ff, &mut bg.b1);
                matmul_b_t_into(dff_pre, t, self.cfg.n_ff, &b.w1, d, residual);
            }

            scratch.reset();
            let dln2_slot = scratch.alloc(td);
            let dg2_slot = scratch.alloc(d);
            let db2_slot = scratch.alloc(d);
            {
                let (dln2, dg2, db2) =
                    scratch.get3_mut(dln2_slot, dg2_slot, db2_slot);
                normalization_backward_into(
                    self.normalization,
                    residual,
                    &c.ln2,
                    &b.ln2_g,
                    dln2,
                    dg2,
                    db2,
                );
                add_inplace(&mut bg.ln2_g, dg2);
                add_inplace(&mut bg.ln2_b, db2);
                for i in 0..td {
                    residual[i] = dx[i] + dln2[i];
                }
            }

            let dproj = &residual[..];
            matmul_grad_b_add_reference(&c.att, t, d, dproj, d, &mut bg.wo);
            matmul_b_t_into(dproj, t, d, &b.wo, d, dx);

            scratch.reset();
            let kv_width = self.kv_width();
            let dq_slot = scratch.alloc(td);
            let dk_slot = scratch.alloc(t * kv_width);
            let dv_slot = scratch.alloc(t * kv_width);
            let dp_slot = scratch.alloc(t);
            {
                let (dq, dk, dv, dp) =
                    scratch.get4_mut(dq_slot, dk_slot, dv_slot, dp_slot);
                self.attention_backward_current(
                    dx,
                    &c.q,
                    &c.k,
                    &c.v,
                    &c.probs,
                    t,
                    dq,
                    dk,
                    dv,
                    dp,
                );
                self.backward_position_qk(dq, dk, t);

                matmul_grad_b_add_reference(&c.h1, t, d, dq, d, &mut bg.wq);
                matmul_grad_b_add_reference(&c.h1, t, d, dk, kv_width, &mut bg.wk);
                matmul_grad_b_add_reference(&c.h1, t, d, dv, kv_width, &mut bg.wv);

                matmul_b_t_into(dq, t, d, &b.wq, d, dx);
                matmul_b_t_add_into(dk, t, kv_width, &b.wk, d, dx);
                matmul_b_t_add_into(dv, t, kv_width, &b.wv, d, dx);
            }

            scratch.reset();
            let dln1_slot = scratch.alloc(td);
            let dg1_slot = scratch.alloc(d);
            let db1_slot = scratch.alloc(d);
            {
                let (dln1, dg1, db1) =
                    scratch.get3_mut(dln1_slot, dg1_slot, db1_slot);
                normalization_backward_into(
                    self.normalization,
                    dx,
                    &c.ln1,
                    &b.ln1_g,
                    dln1,
                    dg1,
                    db1,
                );
                add_inplace(&mut bg.ln1_g, dg1);
                add_inplace(&mut bg.ln1_b, db1);
                for i in 0..td {
                    dx[i] = residual[i] + dln1[i];
                }
            }
        }

        for i in 0..t {
            let token = x[i];
            for j in 0..d {
                gg.tok_emb[token * d + j] += dx[i * d + j];
            }
        }

        match self.position {
            PositionKind::LearnedAbsolute => {
                let positional = LearnedAbsolute::new(&self.pos_emb, self.cfg.block, d)
                    .expect("model positional storage matches config");
                positional
                    .accumulate_backward(dx, t, &mut gg.pos_emb)
                    .expect("recurrent positional gradients match model");
            }
            PositionKind::Rope => {
                let positional = Rotary::new(self.cfg.block, d, self.cfg.n_head)
                    .expect("model RoPE shape matches config");
                positional
                    .accumulate_backward(dx, t, &mut gg.pos_emb)
                    .expect("reserved recurrent positional gradient matches model");
            }
            PositionKind::Alibi => {
                let positional = Alibi::new(self.cfg.block, d, self.cfg.n_head)
                    .expect("model ALiBi shape matches config");
                positional
                    .accumulate_backward(dx, t, &mut gg.pos_emb)
                    .expect("reserved recurrent positional gradient matches model");
            }
        }

        copy_grads_into(gg, grads);
        Ok(loss)
    }

    pub fn generate(
        &self,
        ids: &mut Vec<usize>,
        n_tokens: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) {
        if ids.is_empty() {
            return;
        }
        for _ in 0..n_tokens {
            let start = ids.len().saturating_sub(self.cfg.block);
            let ctx = &ids[start..];
            let logits = self.logits(ctx);
            let row = &logits[(ctx.len() - 1) * self.cfg.vocab..ctx.len() * self.cfg.vocab];
            let next = sample_logits(row, temperature, rng);
            ids.push(next);
        }
    }

    fn forward_decode_one_with_backend<B: Backend>(
        &self,
        token: usize,
        cache: &mut KvCache,
        backend: &B,
    ) -> Result<Vec<f32>, String> {
        let position = cache.len();
        let d = self.cfg.n_embd;
        let mut x = self.tok_emb[token * d..(token + 1) * d].to_vec();

        match self.position {
            PositionKind::LearnedAbsolute => {
                let start = position
                    .checked_mul(d)
                    .ok_or_else(|| "kv cache positional offset overflow".to_string())?;
                for j in 0..d {
                    x[j] += self.pos_emb[start + j];
                }
            }
            PositionKind::Rope | PositionKind::Alibi => {}
        }

        let head_slopes = if self.position == PositionKind::Alibi {
            Some(alibi_slopes(self.cfg.n_head).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let mut staged = Vec::with_capacity(self.blocks.len());

        let kv_width = self.kv_width();
        for (layer_index, b) in self.blocks.iter().enumerate() {
            let h1 = normalization_eval(self.normalization, &x, 1, d, &b.ln1_g, &b.ln1_b);
            let mut q = matmul_with_backend(backend, &h1, 1, d, &b.wq, d);
            let mut k = matmul_with_backend(backend, &h1, 1, d, &b.wk, kv_width);
            let v = matmul_with_backend(backend, &h1, 1, d, &b.wv, kv_width);

            if self.position == PositionKind::Rope {
                let rotary = Rotary::new(self.cfg.block, d, self.cfg.n_head)
                    .map_err(|e| e.to_string())?;
                if self.n_kv_head == self.cfg.n_head {
                    rotary
                        .apply_qk_at_position(&mut q, &mut k, position, self.cfg.n_head)
                        .map_err(|e| e.to_string())?;
                } else {
                    rotary
                        .apply_grouped_qk_at_position(
                            &mut q,
                            &mut k,
                            position,
                            self.cfg.n_head,
                            self.n_kv_head,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }

            let (history_k, history_v) = cache.history(layer_index)?;
            let mut att = vec![0.0; d];
            let mut scores = vec![0.0; position + 1];
            if self.n_kv_head == self.cfg.n_head {
                OPTIMIZED_ATTENTION
                    .forward_decode(
                        &q,
                        history_k,
                        history_v,
                        &k,
                        &v,
                        position,
                        d,
                        self.cfg.n_head,
                        head_slopes.as_deref(),
                        &mut att,
                        &mut scores,
                    )
                    .map_err(|e| e.to_string())?;
            } else {
                OPTIMIZED_ATTENTION
                    .forward_decode_grouped(
                        &q,
                        history_k,
                        history_v,
                        &k,
                        &v,
                        position,
                        d,
                        self.cfg.n_head,
                        self.n_kv_head,
                        head_slopes.as_deref(),
                        &mut att,
                        &mut scores,
                    )
                    .map_err(|e| e.to_string())?;
            }

            let mut proj = matmul_with_backend(backend, &att, 1, d, &b.wo, d);
            let mut r1 = x;
            add_inplace(&mut r1, &proj);

            let h2 = normalization_eval(self.normalization, &r1, 1, d, &b.ln2_g, &b.ln2_b);
            let mut ff_pre = matmul_with_backend(backend, &h2, 1, d, &b.w1, self.cfg.n_ff);
            add_bias_inplace(&mut ff_pre, 1, self.cfg.n_ff, &b.b1);
            let ff_act: Vec<f32> = ff_pre.iter().copied().map(gelu).collect();
            matmul_into_with_backend(
                backend,
                &ff_act,
                1,
                self.cfg.n_ff,
                &b.w2,
                d,
                &mut proj,
            );
            add_bias_inplace(&mut proj, 1, d, &b.b2);
            let mut out = r1;
            add_inplace(&mut out, &proj);
            x = out;
            staged.push((k, v));
        }

        let h_final =
            normalization_eval(self.normalization, &x, 1, d, &self.ln_f_g, &self.ln_f_b);
        let logits = self.project_logits_with_backend(&h_final, 1, backend);

        // Commit only after every layer and the output projection succeeded.
        cache.commit(&staged)?;
        Ok(logits)
    }

    /// Forward path for evaluation/inference. It preserves the training-forward
    /// arithmetic order but does not materialize caches used only by backward.
    fn forward_eval(&self, tokens: &[usize]) -> Vec<f32> {
        self.forward_eval_with_backend(tokens, &OPTIMIZED_CPU_BACKEND)
    }

    fn forward_eval_hidden_with_backend<B: Backend>(
        &self,
        tokens: &[usize],
        backend: &B,
    ) -> Vec<f32> {
        let t = tokens.len();
        let d = self.cfg.n_embd;
        let mut x = self.embed_tokens_with_positions(tokens);

        for b in &self.blocks {
            let h1 = normalization_eval(self.normalization, &x, t, d, &b.ln1_g, &b.ln1_b);
            let mut q = matmul_with_backend(backend, &h1, t, d, &b.wq, d);
            let mut k = matmul_with_backend(backend, &h1, t, d, &b.wk, self.kv_width());
            let v = matmul_with_backend(backend, &h1, t, d, &b.wv, self.kv_width());
            self.apply_position_to_qk(&mut q, &mut k, t);
            let att = self.attention_eval_current(&q, &k, &v, t);
            let mut proj = matmul_with_backend(backend, &att, t, d, &b.wo, d);
            let mut r1 = x;
            add_inplace(&mut r1, &proj);

            let h2 = normalization_eval(self.normalization, &r1, t, d, &b.ln2_g, &b.ln2_b);
            let mut ff_pre = matmul_with_backend(backend, &h2, t, d, &b.w1, self.cfg.n_ff);
            add_bias_inplace(&mut ff_pre, t, self.cfg.n_ff, &b.b1);
            let ff_act: Vec<f32> = ff_pre.iter().copied().map(gelu).collect();
            matmul_into_with_backend(backend, &ff_act, t, self.cfg.n_ff, &b.w2, d, &mut proj);
            add_bias_inplace(&mut proj, t, d, &b.b2);
            let mut out = r1;
            add_inplace(&mut out, &proj);
            x = out;
        }

        normalization_eval(self.normalization, &x, t, d, &self.ln_f_g, &self.ln_f_b)
    }

    fn project_logits_with_backend<B: Backend>(
        &self,
        h_final: &[f32],
        tokens: usize,
        backend: &B,
    ) -> Vec<f32> {
        let mut logits = matmul_with_backend(
            backend,
            h_final,
            tokens,
            self.cfg.n_embd,
            &self.w_out,
            self.cfg.vocab,
        );
        add_bias_inplace(&mut logits, tokens, self.cfg.vocab, &self.b_out);
        logits
    }

    fn forward_eval_with_backend<B: Backend>(
        &self,
        tokens: &[usize],
        backend: &B,
    ) -> Vec<f32> {
        let h_final = self.forward_eval_hidden_with_backend(tokens, backend);
        self.project_logits_with_backend(&h_final, tokens.len(), backend)
    }

    fn forward_eval_recurrent_with_backend<B: Backend>(
        &self,
        tokens: &[usize],
        backend: &B,
        reasoning_steps: usize,
    ) -> Vec<f32> {
        debug_assert!(reasoning_steps > 1);
        let t = tokens.len();
        let d = self.cfg.n_embd;
        let mut x = self.embed_tokens_with_positions(tokens);

        for _ in 0..reasoning_steps {
            for b in &self.blocks {
                let h1 =
                    normalization_eval(self.normalization, &x, t, d, &b.ln1_g, &b.ln1_b);
                let mut q = matmul_with_backend(backend, &h1, t, d, &b.wq, d);
                let mut k = matmul_with_backend(backend, &h1, t, d, &b.wk, self.kv_width());
                let v = matmul_with_backend(backend, &h1, t, d, &b.wv, self.kv_width());
                self.apply_position_to_qk(&mut q, &mut k, t);
                let att = self.attention_eval_current(&q, &k, &v, t);
                let mut proj = matmul_with_backend(backend, &att, t, d, &b.wo, d);
                let mut r1 = x;
                add_inplace(&mut r1, &proj);

                let h2 =
                    normalization_eval(self.normalization, &r1, t, d, &b.ln2_g, &b.ln2_b);
                let mut ff_pre =
                    matmul_with_backend(backend, &h2, t, d, &b.w1, self.cfg.n_ff);
                add_bias_inplace(&mut ff_pre, t, self.cfg.n_ff, &b.b1);
                let ff_act: Vec<f32> = ff_pre.iter().copied().map(gelu).collect();
                matmul_into_with_backend(
                    backend,
                    &ff_act,
                    t,
                    self.cfg.n_ff,
                    &b.w2,
                    d,
                    &mut proj,
                );
                add_bias_inplace(&mut proj, t, d, &b.b2);
                let mut out = r1;
                add_inplace(&mut out, &proj);
                x = out;
            }
        }

        let h_final =
            normalization_eval(self.normalization, &x, t, d, &self.ln_f_g, &self.ln_f_b);
        self.project_logits_with_backend(&h_final, t, backend)
    }

    fn forward_internal_recurrent_with_backend<B: Backend>(
        &self,
        tokens: &[usize],
        backend: &B,
        reasoning_steps: usize,
    ) -> (Vec<f32>, ForwardCache) {
        debug_assert!(reasoning_steps > 1);
        let t = tokens.len();
        let d = self.cfg.n_embd;
        let mut x = self.embed_tokens_with_positions(tokens);
        let mut layer_caches =
            Vec::with_capacity(self.blocks.len().saturating_mul(reasoning_steps));

        for _ in 0..reasoning_steps {
            for b in &self.blocks {
                let (h1, ln1) =
                    normalization_forward(self.normalization, &x, t, d, &b.ln1_g, &b.ln1_b);
                let mut q = matmul_with_backend(backend, &h1, t, d, &b.wq, d);
                let mut k = matmul_with_backend(backend, &h1, t, d, &b.wk, self.kv_width());
                let v = matmul_with_backend(backend, &h1, t, d, &b.wv, self.kv_width());
                self.apply_position_to_qk(&mut q, &mut k, t);
                let (att, probs) = self.attention_forward_current(&q, &k, &v, t);
                let mut proj = matmul_with_backend(backend, &att, t, d, &b.wo, d);
                let mut r1 = x;
                add_inplace(&mut r1, &proj);

                let (h2, ln2) =
                    normalization_forward(self.normalization, &r1, t, d, &b.ln2_g, &b.ln2_b);
                let mut ff_pre =
                    matmul_with_backend(backend, &h2, t, d, &b.w1, self.cfg.n_ff);
                add_bias_inplace(&mut ff_pre, t, self.cfg.n_ff, &b.b1);
                let ff_act: Vec<f32> = ff_pre.iter().copied().map(gelu).collect();
                matmul_into_with_backend(
                    backend,
                    &ff_act,
                    t,
                    self.cfg.n_ff,
                    &b.w2,
                    d,
                    &mut proj,
                );
                add_bias_inplace(&mut proj, t, d, &b.b2);
                let mut out = r1;
                add_inplace(&mut out, &proj);

                layer_caches.push(LayerCache {
                    ln1,
                    h1,
                    q,
                    k,
                    v,
                    probs,
                    att,
                    ln2,
                    h2,
                    ff_pre,
                    ff_act,
                });
                x = out;
            }
        }

        let (h_final, ln_f) =
            normalization_forward(self.normalization, &x, t, d, &self.ln_f_g, &self.ln_f_b);
        let logits = self.project_logits_with_backend(&h_final, t, backend);
        (
            logits,
            ForwardCache {
                layers: layer_caches,
                ln_f,
                h_final,
            },
        )
    }

    fn forward_internal(&self, tokens: &[usize]) -> (Vec<f32>, ForwardCache) {
        self.forward_internal_with_backend(tokens, &OPTIMIZED_CPU_BACKEND)
    }

    fn forward_internal_with_backend<B: Backend>(
        &self,
        tokens: &[usize],
        backend: &B,
    ) -> (Vec<f32>, ForwardCache) {
        let t = tokens.len();
        let d = self.cfg.n_embd;
        let mut x = self.embed_tokens_with_positions(tokens);

        let mut layer_caches = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let (h1, ln1) = normalization_forward(self.normalization, &x, t, d, &b.ln1_g, &b.ln1_b);
            let mut q = matmul_with_backend(backend, &h1, t, d, &b.wq, d);
            let mut k = matmul_with_backend(backend, &h1, t, d, &b.wk, self.kv_width());
            let v = matmul_with_backend(backend, &h1, t, d, &b.wv, self.kv_width());
            self.apply_position_to_qk(&mut q, &mut k, t);
            let (att, probs) = self.attention_forward_current(&q, &k, &v, t);
            let mut proj = matmul_with_backend(backend, &att, t, d, &b.wo, d);
            let mut r1 = x;
            add_inplace(&mut r1, &proj);

            let (h2, ln2) = normalization_forward(self.normalization, &r1, t, d, &b.ln2_g, &b.ln2_b);
            let mut ff_pre = matmul_with_backend(backend, &h2, t, d, &b.w1, self.cfg.n_ff);
            add_bias_inplace(&mut ff_pre, t, self.cfg.n_ff, &b.b1);
            let ff_act: Vec<f32> = ff_pre.iter().copied().map(gelu).collect();
            matmul_into_with_backend(backend, &ff_act, t, self.cfg.n_ff, &b.w2, d, &mut proj);
            add_bias_inplace(&mut proj, t, d, &b.b2);
            let mut out = r1;
            add_inplace(&mut out, &proj);

            layer_caches.push(LayerCache {
                ln1,
                h1,
                q,
                k,
                v,
                probs,
                att,
                ln2,
                h2,
                ff_pre,
                ff_act,
            });
            x = out;
        }

        let (h_final, ln_f) = normalization_forward(self.normalization, &x, t, d, &self.ln_f_g, &self.ln_f_b);
        let mut logits = matmul_with_backend(backend, &h_final, t, d, &self.w_out, self.cfg.vocab);
        add_bias_inplace(&mut logits, t, self.cfg.vocab, &self.b_out);

        (
            logits,
            ForwardCache {
                layers: layer_caches,
                ln_f,
                h_final,
            },
        )
    }

    fn zero_grads(&self) -> GptGrad {
        GptGrad {
            tok_emb: vec![0.0; self.tok_emb.len()],
            pos_emb: vec![0.0; self.pos_emb.len()],
            blocks: (0..self.cfg.n_layer)
                .map(|_| zeros_block_grad(self.cfg, self.n_kv_head))
                .collect(),
            ln_f_g: vec![0.0; self.ln_f_g.len()],
            ln_f_b: vec![0.0; self.ln_f_b.len()],
            w_out: vec![0.0; self.w_out.len()],
            b_out: vec![0.0; self.b_out.len()],
        }
    }

    fn block_param_count(&self) -> usize {
        let d = self.cfg.n_embd;
        let kv_width = self.kv_width();
        2 * d * d
            + 2 * d * kv_width
            + 2 * d
            + 2 * d
            + d * self.cfg.n_ff
            + self.cfg.n_ff
            + self.cfg.n_ff * d
            + d
    }

    fn param_count(&self) -> usize {
        Self::expected_parameter_count(self.cfg, self.n_kv_head)
            .expect("validated model shape has finite parameter count")
    }
}

fn copy_param(dst: &mut [f32], src: &[f32], p: &mut usize) {
    let end = *p + dst.len();
    dst.copy_from_slice(&src[*p..end]);
    *p = end;
}

fn copy_grad(src: &[f32], dst: &mut [f32], p: &mut usize) {
    let end = *p + src.len();
    dst[*p..end].copy_from_slice(src);
    *p = end;
}

fn copy_grads_into(g: &GptGrad, out: &mut [f32]) {
    let mut p = 0usize;
    copy_grad(&g.tok_emb, out, &mut p);
    copy_grad(&g.pos_emb, out, &mut p);
    for b in &g.blocks {
        copy_grad(&b.ln1_g, out, &mut p);
        copy_grad(&b.ln1_b, out, &mut p);
        copy_grad(&b.wq, out, &mut p);
        copy_grad(&b.wk, out, &mut p);
        copy_grad(&b.wv, out, &mut p);
        copy_grad(&b.wo, out, &mut p);
        copy_grad(&b.ln2_g, out, &mut p);
        copy_grad(&b.ln2_b, out, &mut p);
        copy_grad(&b.w1, out, &mut p);
        copy_grad(&b.b1, out, &mut p);
        copy_grad(&b.w2, out, &mut p);
        copy_grad(&b.b2, out, &mut p);
    }
    copy_grad(&g.ln_f_g, out, &mut p);
    copy_grad(&g.ln_f_b, out, &mut p);
    copy_grad(&g.w_out, out, &mut p);
    copy_grad(&g.b_out, out, &mut p);
    debug_assert_eq!(p, out.len());
}

fn mean_cross_entropy(logits: &[f32], targets: &[usize], vocab: usize) -> f32 {
    assert_eq!(logits.len(), targets.len() * vocab);
    let mut loss = 0.0f32;
    for (i, &target) in targets.iter().enumerate() {
        let row = &logits[i * vocab..(i + 1) * vocab];
        let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &logit in row {
            sum += (logit - maxv).exp();
        }
        let p = ((row[target] - maxv).exp() / sum.max(1e-20)).max(1e-20);
        loss -= p.ln();
    }
    loss / targets.len() as f32
}

fn add_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

fn add_bias_inplace(x: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    assert_eq!(bias.len(), cols);
    for i in 0..rows {
        for j in 0..cols {
            x[i * cols + j] += bias[j];
        }
    }
}

fn sum_rows_into(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    assert_eq!(out.len(), cols);
    for i in 0..rows {
        for j in 0..cols {
            out[j] += x[i * cols + j];
        }
    }
}

fn matmul_with_backend<B: Backend>(
    backend: &B,
    a: &[f32],
    rows: usize,
    inner: usize,
    b: &[f32],
    cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    matmul_into_with_backend(backend, a, rows, inner, b, cols, &mut out);
    out
}

fn matmul_into_with_backend<B: Backend>(
    backend: &B,
    a: &[f32],
    rows: usize,
    inner: usize,
    b: &[f32],
    cols: usize,
    out: &mut [f32],
) {
    backend
        .matmul(
            MatrixRef::new(a, rows, inner, backend.device())
                .expect("model A shape/device invariant"),
            MatrixRef::new(b, inner, cols, backend.device())
                .expect("model B shape/device invariant"),
            MatrixMut::new(out, rows, cols, backend.device())
                .expect("model output shape/device invariant"),
        )
        .expect("model backend matmul invariant");
}

fn matmul_grad_b(
    a: &[f32],
    rows: usize,
    inner: usize,
    dy: &[f32],
    cols: usize,
    db: &mut [f32],
) {
    crate::kernels::matmul_grad_b_rowwise_zeroed(a, rows, inner, dy, cols, db);
}

fn matmul_grad_b_add_reference(
    a: &[f32],
    rows: usize,
    inner: usize,
    dy: &[f32],
    cols: usize,
    db: &mut [f32],
) {
    crate::kernels::matmul_grad_b_reference(a, rows, inner, dy, cols, db);
}

fn matmul_b_t_into(
    dy: &[f32],
    rows: usize,
    out_cols: usize,
    b: &[f32],
    result_cols: usize,
    out: &mut [f32],
) {
    assert_eq!(dy.len(), rows * out_cols);
    assert_eq!(b.len(), result_cols * out_cols);
    assert_eq!(out.len(), rows * result_cols);
    crate::kernels::matmul_b_t_row_slices_into(dy, rows, out_cols, b, result_cols, out);
}

fn matmul_b_t_add_into(
    dy: &[f32],
    rows: usize,
    out_cols: usize,
    b: &[f32],
    result_cols: usize,
    out: &mut [f32],
) {
    crate::kernels::matmul_b_t_row_slices_add_into(dy, rows, out_cols, b, result_cols, out);
}

fn normalization_eval(
    kind: NormalizationKind,
    x: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    match kind {
        NormalizationKind::LayerNorm => layernorm_eval(x, rows, cols, gamma, beta),
        NormalizationKind::RmsNorm => rmsnorm_eval(x, rows, cols, gamma),
    }
}

fn normalization_forward(
    kind: NormalizationKind,
    x: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
) -> (Vec<f32>, LnCache) {
    match kind {
        NormalizationKind::LayerNorm => layernorm_forward(x, rows, cols, gamma, beta),
        NormalizationKind::RmsNorm => rmsnorm_forward(x, rows, cols, gamma),
    }
}

fn normalization_backward_into(
    kind: NormalizationKind,
    dy: &[f32],
    cache: &LnCache,
    gamma: &[f32],
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) {
    match kind {
        NormalizationKind::LayerNorm => {
            layernorm_backward_into(dy, cache, gamma, dx, dgamma, dbeta)
        }
        NormalizationKind::RmsNorm => {
            rmsnorm_backward_into(dy, cache, gamma, dx, dgamma, dbeta)
        }
    }
}

fn rmsnorm_eval(
    x: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
) -> Vec<f32> {
    const EPS: f32 = 1e-5;
    assert_eq!(x.len(), rows * cols);
    assert_eq!(gamma.len(), cols);
    let mut y = vec![0.0; x.len()];
    for i in 0..rows {
        let row = &x[i * cols..(i + 1) * cols];
        let mean_sq = row.iter().map(|v| *v * *v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        for j in 0..cols {
            y[i * cols + j] = x[i * cols + j] * inv * gamma[j];
        }
    }
    y
}

fn rmsnorm_forward(
    x: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
) -> (Vec<f32>, LnCache) {
    const EPS: f32 = 1e-5;
    assert_eq!(x.len(), rows * cols);
    assert_eq!(gamma.len(), cols);
    let mut y = vec![0.0; x.len()];
    let mut xhat = vec![0.0; x.len()];
    let mut inv_std = vec![0.0; rows];
    for i in 0..rows {
        let row = &x[i * cols..(i + 1) * cols];
        let mean_sq = row.iter().map(|v| *v * *v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        inv_std[i] = inv;
        for j in 0..cols {
            let idx = i * cols + j;
            let h = x[idx] * inv;
            xhat[idx] = h;
            y[idx] = h * gamma[j];
        }
    }
    (
        y,
        LnCache {
            xhat,
            inv_std,
            rows,
            cols,
        },
    )
}

fn rmsnorm_backward_into(
    dy: &[f32],
    cache: &LnCache,
    gamma: &[f32],
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) {
    let rows = cache.rows;
    let cols = cache.cols;
    assert_eq!(dy.len(), rows * cols);
    assert_eq!(gamma.len(), cols);
    assert_eq!(dx.len(), dy.len());
    assert_eq!(dgamma.len(), cols);
    assert_eq!(dbeta.len(), cols);
    dgamma.fill(0.0);
    // beta slots are retained only to preserve the historical parameter layout.
    dbeta.fill(0.0);

    for i in 0..rows {
        let mut sum_gxh = 0.0f32;
        for j in 0..cols {
            let idx = i * cols + j;
            dgamma[j] += dy[idx] * cache.xhat[idx];
            sum_gxh += dy[idx] * gamma[j] * cache.xhat[idx];
        }
        let mean_gxh = sum_gxh / cols as f32;
        for j in 0..cols {
            let idx = i * cols + j;
            let z = dy[idx] * gamma[j];
            dx[idx] = cache.inv_std[i] * (z - cache.xhat[idx] * mean_gxh);
        }
    }
}

fn layernorm_eval(
    x: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    const EPS: f32 = 1e-5;
    let mut y = vec![0.0; x.len()];
    for i in 0..rows {
        let row = &x[i * cols..(i + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row
            .iter()
            .map(|v| {
                let z = *v - mean;
                z * z
            })
            .sum::<f32>()
            / cols as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for j in 0..cols {
            let h = (x[i * cols + j] - mean) * inv;
            y[i * cols + j] = h * gamma[j] + beta[j];
        }
    }
    y
}

fn layernorm_forward(
    x: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
) -> (Vec<f32>, LnCache) {
    const EPS: f32 = 1e-5;
    let mut y = vec![0.0; x.len()];
    let mut xhat = vec![0.0; x.len()];
    let mut inv_std = vec![0.0; rows];
    for i in 0..rows {
        let row = &x[i * cols..(i + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row
            .iter()
            .map(|v| {
                let z = *v - mean;
                z * z
            })
            .sum::<f32>()
            / cols as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        inv_std[i] = inv;
        for j in 0..cols {
            let h = (x[i * cols + j] - mean) * inv;
            xhat[i * cols + j] = h;
            y[i * cols + j] = h * gamma[j] + beta[j];
        }
    }
    (
        y,
        LnCache {
            xhat,
            inv_std,
            rows,
            cols,
        },
    )
}

fn layernorm_backward_into(
    dy: &[f32],
    cache: &LnCache,
    gamma: &[f32],
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) {
    let rows = cache.rows;
    let cols = cache.cols;
    assert_eq!(dy.len(), rows * cols);
    assert_eq!(gamma.len(), cols);
    assert_eq!(dx.len(), dy.len());
    assert_eq!(dgamma.len(), cols);
    assert_eq!(dbeta.len(), cols);
    dgamma.fill(0.0);
    dbeta.fill(0.0);

    for i in 0..rows {
        let mut sum_g = 0.0;
        let mut sum_gxh = 0.0;
        for j in 0..cols {
            let idx = i * cols + j;
            dgamma[j] += dy[idx] * cache.xhat[idx];
            dbeta[j] += dy[idx];
            let z = dy[idx] * gamma[j];
            sum_g += z;
            sum_gxh += z * cache.xhat[idx];
        }
        let scale = cache.inv_std[i] / cols as f32;
        for j in 0..cols {
            let idx = i * cols + j;
            let z = dy[idx] * gamma[j];
            dx[idx] = scale * (cols as f32 * z - sum_g - cache.xhat[idx] * sum_gxh);
        }
    }
}

fn attention_eval(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
    position: PositionKind,
) -> Vec<f32> {
    let shape = AttentionShape {
        tokens: t,
        width: d,
        heads: n_head,
    };
    let mut out = vec![0.0; t * d];
    let mut scores = vec![0.0; t];
    if position == PositionKind::Alibi {
        let slopes = alibi_slopes(n_head).expect("model ALiBi head count is positive");
        OPTIMIZED_ATTENTION
            .forward_eval_alibi(q, k, v, shape, &slopes, &mut out, &mut scores)
            .expect("model ALiBi attention shapes are validated by Config");
    } else {
        OPTIMIZED_ATTENTION
            .forward_eval(q, k, v, shape, &mut out, &mut scores)
            .expect("model attention shapes are validated by Config");
    }
    out
}

fn attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
    position: PositionKind,
) -> (Vec<f32>, Vec<f32>) {
    let shape = AttentionShape {
        tokens: t,
        width: d,
        heads: n_head,
    };
    let mut probs = vec![0.0; n_head * t * t];
    let mut out = vec![0.0; t * d];
    if position == PositionKind::Alibi {
        let slopes = alibi_slopes(n_head).expect("model ALiBi head count is positive");
        OPTIMIZED_ATTENTION
            .forward_cached_alibi(q, k, v, shape, &slopes, &mut out, &mut probs)
            .expect("model ALiBi attention shapes are validated by Config");
    } else {
        OPTIMIZED_ATTENTION
            .forward_cached(q, k, v, shape, &mut out, &mut probs)
            .expect("model attention shapes are validated by Config");
    }
    (out, probs)
}

fn attention_backward_into(
    dout: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    dp: &mut [f32],
) {
    let shape = AttentionShape {
        tokens: t,
        width: d,
        heads: n_head,
    };
    OPTIMIZED_ATTENTION
        .backward(dout, q, k, v, probs, shape, dq, dk, dv, dp)
        .expect("model attention workspace shapes match Config");
}

fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

fn gelu_deriv(x: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    let u = C * (x + 0.044_715 * x * x * x);
    let th = u.tanh();
    let du = C * (1.0 + 3.0 * 0.044_715 * x * x);
    0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du
}

fn sample_logits(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> usize {
    if temperature <= 1e-6 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
    let inv_t = 1.0 / temperature.max(1e-4);
    let maxv = logits
        .iter()
        .map(|x| *x * inv_t)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for &x in logits {
        let p = (x * inv_t - maxv).exp();
        probs.push(p);
        sum += p;
    }
    let mut r = rng.gen::<f32>() * sum.max(1e-20);
    for (i, p) in probs.into_iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    logits.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::{
        rmsnorm_backward_into, rmsnorm_eval, rmsnorm_forward, BackendId, BackwardWorkspace, Config,
        CpuBackend, Gpt, NormalizationKind, PositionKind, RecurrentConfig,
    };
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn explicit_layernorm_policy_is_bit_exact_with_historical_default() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng_a = StdRng::seed_from_u64(0xA11CE_9301);
        let mut rng_b = StdRng::seed_from_u64(0xA11CE_9301);
        let a = Gpt::new(cfg, &mut rng_a);
        let b = Gpt::new_with_normalization(cfg, NormalizationKind::LayerNorm, &mut rng_b);
        assert_eq!(a.collect_params(), b.collect_params());
        assert_eq!(a.normalization(), NormalizationKind::LayerNorm);
        assert_eq!(b.normalization(), NormalizationKind::LayerNorm);

        let x = [0, 1, 2, 3];
        let y = [1, 2, 3, 4];
        assert_eq!(a.logits(&x), b.logits(&x));
        assert_eq!(a.loss(&x, &y).to_bits(), b.loss(&x, &y).to_bits());

        let mut ga = vec![0.0; a.collect_params().len()];
        let mut gb = vec![0.0; b.collect_params().len()];
        let la = a.backward_into(&x, &y, &mut ga);
        let lb = b.backward_into(&x, &y, &mut gb);
        assert_eq!(la.to_bits(), lb.to_bits());
        assert_eq!(ga, gb);
    }

    #[test]
    fn explicit_learned_absolute_policy_is_bit_exact_with_default() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng_a = StdRng::seed_from_u64(0xA11CE_3801);
        let mut rng_b = StdRng::seed_from_u64(0xA11CE_3801);
        let a = Gpt::new(cfg, &mut rng_a);
        let b = Gpt::new_with_policies(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::LearnedAbsolute,
            &mut rng_b,
        );
        assert_eq!(a.collect_params(), b.collect_params());
        assert_eq!(b.position_kind(), PositionKind::LearnedAbsolute);

        let x = [0, 1, 2, 3];
        let y = [1, 2, 3, 4];
        assert_eq!(a.logits(&x), b.logits(&x));
        let mut ga = vec![0.0; a.collect_params().len()];
        let mut gb = vec![0.0; b.collect_params().len()];
        let la = a.backward_into(&x, &y, &mut ga);
        let lb = b.backward_into(&x, &y, &mut gb);
        assert_eq!(la.to_bits(), lb.to_bits());
        assert_eq!(ga, gb);
    }

    #[test]
    fn rope_keeps_parameter_layout_but_reserved_position_gradient_zero() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng_a = StdRng::seed_from_u64(0xA11CE_3802);
        let mut rng_b = StdRng::seed_from_u64(0xA11CE_3802);
        let learned = Gpt::new(cfg, &mut rng_a);
        let rope = Gpt::new_with_policies(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::Rope,
            &mut rng_b,
        );
        assert_eq!(learned.collect_params(), rope.collect_params());
        assert_eq!(rope.position_kind(), PositionKind::Rope);

        let x = [0, 1, 2, 3];
        let y = [1, 2, 3, 4];
        assert_ne!(learned.logits(&x), rope.logits(&x));

        let mut grads = vec![0.0; rope.collect_params().len()];
        let loss = rope.backward_into(&x, &y, &mut grads);
        assert!(loss.is_finite());
        assert!(grads.iter().all(|g| g.is_finite()));

        let pos_start = cfg.vocab * cfg.n_embd;
        let pos_end = pos_start + cfg.block * cfg.n_embd;
        assert!(
            grads[pos_start..pos_end].iter().all(|&g| g == 0.0),
            "reserved learned-absolute position slots must remain inert under RoPE"
        );
        assert!(
            grads[..pos_start].iter().any(|g| g.abs() > 0.0),
            "token embeddings should still receive gradient"
        );
    }

    #[test]
    fn alibi_keeps_parameter_layout_and_reserved_position_gradient_inert() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng_a = StdRng::seed_from_u64(0xA11CE_3901);
        let mut rng_b = StdRng::seed_from_u64(0xA11CE_3901);
        let learned = Gpt::new(cfg, &mut rng_a);
        let alibi = Gpt::new_with_policies(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::Alibi,
            &mut rng_b,
        );
        assert_eq!(learned.collect_params(), alibi.collect_params());
        assert_eq!(alibi.position_kind(), PositionKind::Alibi);

        let x = [0, 1, 2, 3];
        let y = [1, 2, 3, 4];
        assert_ne!(learned.logits(&x), alibi.logits(&x));

        let mut grads = vec![0.0; alibi.collect_params().len()];
        let loss = alibi.backward_into(&x, &y, &mut grads);
        assert!(loss.is_finite());
        assert!(grads.iter().all(|g| g.is_finite()));

        let pos_start = cfg.vocab * cfg.n_embd;
        let pos_end = pos_start + cfg.block * cfg.n_embd;
        assert!(
            grads[pos_start..pos_end].iter().all(|&g| g == 0.0),
            "reserved learned-absolute position slots must remain inert under ALiBi"
        );
        assert!(
            grads[..pos_start].iter().any(|g| g.abs() > 0.0),
            "token embeddings should still receive gradient"
        );
    }

    #[test]
    fn rmsnorm_forward_and_backward_match_finite_differences() {
        let x = vec![0.4f32, -0.7, 1.2, -0.3];
        let gamma = vec![1.1f32, 0.8, 1.3, 0.9];
        let dy = vec![0.2f32, -0.4, 0.7, 0.1];
        let (forward, cache) = rmsnorm_forward(&x, 1, 4, &gamma);
        assert_eq!(forward, rmsnorm_eval(&x, 1, 4, &gamma));

        let mut dx = vec![0.0; 4];
        let mut dgamma = vec![0.0; 4];
        let mut dbeta = vec![f32::NAN; 4];
        rmsnorm_backward_into(&dy, &cache, &gamma, &mut dx, &mut dgamma, &mut dbeta);
        assert_eq!(dbeta, vec![0.0; 4]);

        let objective = |xx: &[f32], gg: &[f32]| -> f32 {
            rmsnorm_eval(xx, 1, 4, gg)
                .iter()
                .zip(&dy)
                .map(|(a, b)| a * b)
                .sum()
        };
        let h = 1e-3f32;
        for i in 0..4 {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[i] += h;
            minus[i] -= h;
            let numeric = (objective(&plus, &gamma) - objective(&minus, &gamma)) / (2.0 * h);
            assert!((dx[i] - numeric).abs() < 2e-3, "dx[{i}] analytic={} numeric={numeric}", dx[i]);
        }
        for i in 0..4 {
            let mut plus = gamma.clone();
            let mut minus = gamma.clone();
            plus[i] += h;
            minus[i] -= h;
            let numeric = (objective(&x, &plus) - objective(&x, &minus)) / (2.0 * h);
            assert!((dgamma[i] - numeric).abs() < 2e-3, "dgamma[{i}] analytic={} numeric={numeric}", dgamma[i]);
        }
    }

    #[test]
    fn rmsnorm_model_backward_is_finite_and_keeps_parameter_layout() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng_a = StdRng::seed_from_u64(0xA11CE_9302);
        let mut rng_b = StdRng::seed_from_u64(0xA11CE_9302);
        let layer = Gpt::new(cfg, &mut rng_a);
        let rms = Gpt::new_with_normalization(cfg, NormalizationKind::RmsNorm, &mut rng_b);
        assert_eq!(layer.collect_params(), rms.collect_params());

        let mut grads = vec![0.0; rms.collect_params().len()];
        let loss = rms.backward_into(&[0, 1, 2, 3], &[1, 2, 3, 4], &mut grads);
        assert!(loss.is_finite());
        assert!(grads.iter().all(|g| g.is_finite()));
        assert_eq!(rms.normalization(), NormalizationKind::RmsNorm);
    }

    #[test]
    fn params_roundtrip() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng = rand::thread_rng();
        let mut a = Gpt::new(cfg, &mut rng);
        let params = a.collect_params();
        for x in &mut a.tok_emb {
            *x = 0.0;
        }
        a.write_params(&params);
        assert_eq!(a.collect_params(), params);
    }

    #[test]
    fn backward_is_finite_and_sized() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let mut rng = rand::thread_rng();
        let gpt = Gpt::new(cfg, &mut rng);
        let mut grads = vec![0.0; gpt.collect_params().len()];
        let loss = gpt.backward_into(&[0, 1, 2, 3], &[1, 2, 3, 4], &mut grads);
        assert!(loss.is_finite() && loss > 0.0);
        assert!(grads.iter().all(|x| x.is_finite()));
        assert!(grads.iter().any(|x| x.abs() > 0.0));
    }

    #[test]
    fn forward_diagnostics_matches_eval_logits_and_orders_summaries() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(44);
        let gpt = Gpt::new(cfg, &mut rng);
        let tokens = [0, 1, 2, 3];

        let expected = gpt.logits(&tokens);
        let (observed, report) = gpt.forward_diagnostics(&tokens).unwrap();

        assert_eq!(observed, expected);
        assert_eq!(report.tensors.len(), cfg.n_layer * 9 + 2);
        assert!(report.first_fault().is_none());

        let names: Vec<(Option<usize>, &'static str)> = report
            .tensors
            .iter()
            .map(|summary| (summary.layer, summary.name))
            .collect();
        assert_eq!(
            names,
            vec![
                (Some(0), "h1"),
                (Some(0), "q"),
                (Some(0), "k"),
                (Some(0), "v"),
                (Some(0), "probs"),
                (Some(0), "att"),
                (Some(0), "h2"),
                (Some(0), "ff_pre"),
                (Some(0), "ff_act"),
                (Some(1), "h1"),
                (Some(1), "q"),
                (Some(1), "k"),
                (Some(1), "v"),
                (Some(1), "probs"),
                (Some(1), "att"),
                (Some(1), "h2"),
                (Some(1), "ff_pre"),
                (Some(1), "ff_act"),
                (None, "h_final"),
                (None, "logits"),
            ]
        );
    }

    #[test]
    fn forward_diagnostics_reports_first_corrupt_layer_tensor() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(45);
        let mut gpt = Gpt::new(cfg, &mut rng);
        gpt.tok_emb[0] = f32::NAN;

        let error = gpt.forward_diagnostics(&[0, 1, 2, 3]).unwrap_err();
        assert!(error.contains("layer[0].h1"));
        assert!(error.contains("NaN"));
        assert!(error.contains("index"));
    }

    #[test]
    fn forward_diagnostics_identifies_logits_when_activations_are_clean() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(46);
        let mut gpt = Gpt::new(cfg, &mut rng);
        gpt.b_out[0] = f32::NAN;

        let error = gpt.forward_diagnostics(&[1, 2, 3, 4]).unwrap_err();
        assert!(error.contains("logits"));
        assert!(error.contains("Logits"));
        assert!(error.contains("NaN"));
    }

    #[test]
    fn forward_loss_matches_backward_loss() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let mut rng = rand::thread_rng();
        let gpt = Gpt::new(cfg, &mut rng);
        let x = [0, 1, 2, 3];
        let y = [1, 2, 3, 4];
        let forward_loss = gpt.loss(&x, &y);
        let mut grads = vec![0.0; gpt.collect_params().len()];
        let backward_loss = gpt.backward_into(&x, &y, &mut grads);
        assert!((forward_loss - backward_loss).abs() < 1e-6);
    }

    #[test]
    fn reused_backward_workspace_resets_exactly() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let mut rng = rand::thread_rng();
        let gpt = Gpt::new(cfg, &mut rng);
        let n = gpt.collect_params().len();
        let mut workspace = BackwardWorkspace::new(&gpt);
        let mut reference = vec![0.0; n];
        let mut reused = vec![0.0; n];

        for (x, y) in [
            ([0, 1, 2, 3], [1, 2, 3, 4]),
            ([3, 2, 1, 0], [2, 1, 0, 6]),
        ] {
            let reference_loss = gpt.backward_into(&x, &y, &mut reference);
            reused.fill(f32::NAN);
            let reused_loss = gpt.backward_into_reuse(&x, &y, &mut reused, &mut workspace);
            assert_eq!(reused_loss, reference_loss);
            assert_eq!(reused, reference);
        }
    }

    #[test]
    fn reused_backward_workspace_survives_poison_and_variable_contexts() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(0xA11CE_211B);
        let gpt = Gpt::new(cfg, &mut rng);
        let n = gpt.collect_params().len();
        let mut workspace = BackwardWorkspace::new(&gpt);
        let scratch_capacity = workspace.scratch.capacity();
        let mut reference = vec![0.0; n];
        let mut reused = vec![0.0; n];

        let cases: [(&[usize], &[usize]); 3] = [
            (&[0, 1, 2, 3], &[1, 2, 3, 4]),
            (&[3, 2], &[2, 1]),
            (&[5, 4, 3], &[4, 3, 2]),
        ];

        for (x, y) in cases {
            let reference_loss = gpt.backward_into(x, y, &mut reference);
            workspace.scratch.reset_poison();
            reused.fill(f32::NAN);
            let reused_loss = gpt.backward_into_reuse(x, y, &mut reused, &mut workspace);
            assert_eq!(reused_loss, reference_loss);
            assert_eq!(reused, reference);
        }

        assert_eq!(workspace.scratch.capacity(), scratch_capacity);
        assert_eq!(workspace.scratch.growths(), 0);
        assert!(workspace.scratch.resets() > 0);
    }

    #[test]
    fn eval_forward_matches_training_forward_logits_exactly() {
        let cases = [
            (
                Config {
                    vocab: 7,
                    n_embd: 4,
                    n_head: 1,
                    n_layer: 1,
                    block: 3,
                    n_ff: 8,
                },
                0xA11CE_2701,
            ),
            (
                Config {
                    vocab: 11,
                    n_embd: 8,
                    n_head: 2,
                    n_layer: 2,
                    block: 4,
                    n_ff: 16,
                },
                0xA11CE_2702,
            ),
            (
                Config {
                    vocab: 13,
                    n_embd: 12,
                    n_head: 3,
                    n_layer: 3,
                    block: 5,
                    n_ff: 24,
                },
                0xA11CE_2703,
            ),
            (
                Config {
                    vocab: 17,
                    n_embd: 16,
                    n_head: 4,
                    n_layer: 2,
                    block: 6,
                    n_ff: 32,
                },
                0xA11CE_2704,
            ),
        ];

        for (cfg, seed) in cases {
            let mut rng = StdRng::seed_from_u64(seed);
            let gpt = Gpt::new(cfg, &mut rng);
            let tokens: Vec<usize> = (0..cfg.block).map(|i| (i * 3 + 1) % cfg.vocab).collect();

            for len in 1..=cfg.block {
                let input = &tokens[..len];
                let eval_logits = gpt.logits(input);
                let (training_logits, _) = gpt.forward_internal(input);
                assert_eq!(
                    eval_logits, training_logits,
                    "eval/training logits diverged for seed={seed:#x}, heads={}, layers={}, context={len}",
                    cfg.n_head, cfg.n_layer
                );
            }
        }
    }

    #[test]
    fn cpu_backends_are_exact_for_logits_and_loss_across_contexts() {
        for (seed, cfg) in [
            (
                0x2381,
                Config {
                    vocab: 11,
                    n_embd: 8,
                    n_head: 2,
                    n_layer: 1,
                    block: 5,
                    n_ff: 16,
                },
            ),
            (
                0x2382,
                Config {
                    vocab: 13,
                    n_embd: 12,
                    n_head: 3,
                    n_layer: 2,
                    block: 7,
                    n_ff: 20,
                },
            ),
        ] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let gpt = Gpt::new(cfg, &mut rng);
            let tokens: Vec<usize> = (0..cfg.block)
                .map(|i| (i * 5 + 2) % cfg.vocab)
                .collect();
            let targets: Vec<usize> = (0..cfg.block)
                .map(|i| (i * 7 + 3) % cfg.vocab)
                .collect();

            for len in 1..=cfg.block {
                let x = &tokens[..len];
                let y = &targets[..len];
                let optimized = gpt.logits_with_cpu_backend(x, CpuBackend::Optimized);
                let scalar = gpt.logits_with_cpu_backend(x, CpuBackend::Scalar);
                assert_eq!(optimized, scalar, "logits backend mismatch len={len}");
                assert_eq!(gpt.logits(x), optimized, "default backend is not optimized");
                assert_eq!(
                    gpt.loss_with_cpu_backend(x, y, CpuBackend::Optimized),
                    gpt.loss_with_cpu_backend(x, y, CpuBackend::Scalar),
                    "loss backend mismatch len={len}"
                );
            }
        }
        assert_eq!(CpuBackend::Optimized.id(), BackendId::OptimizedCpu);
        assert_eq!(CpuBackend::Scalar.id(), BackendId::ScalarCpu);
    }

    #[test]
    fn grouped_kv_constructor_preserves_mha_default_and_compacts_parameters() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 4,
            n_layer: 2,
            block: 8,
            n_ff: 16,
        };
        let mut rng_a = StdRng::seed_from_u64(0xA11CE_1401);
        let mut rng_b = StdRng::seed_from_u64(0xA11CE_1401);
        let mha = Gpt::new(cfg, &mut rng_a);
        let explicit_mha = Gpt::new_with_attention_heads(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::LearnedAbsolute,
            cfg.n_head,
            &mut rng_b,
        );
        assert_eq!(mha.collect_params(), explicit_mha.collect_params());
        assert_eq!(mha.n_kv_head(), cfg.n_head);
        assert_eq!(mha.kv_width(), cfg.n_embd);

        let mut rng_gqa = StdRng::seed_from_u64(0xA11CE_1401);
        let gqa = Gpt::new_with_attention_heads(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::LearnedAbsolute,
            2,
            &mut rng_gqa,
        );
        assert_eq!(gqa.n_kv_head(), 2);
        assert_eq!(gqa.kv_width(), 4);
        assert!(gqa.collect_params().len() < mha.collect_params().len());

        let mut rng_mqa = StdRng::seed_from_u64(0xA11CE_1401);
        let mqa = Gpt::new_with_attention_heads(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::LearnedAbsolute,
            1,
            &mut rng_mqa,
        );
        assert_eq!(mqa.kv_width(), 2);
        assert!(mqa.collect_params().len() < gqa.collect_params().len());

        // Compact variants consume the same RNG budget as MHA, so everything
        // initialized after Wk/Wv remains directly comparable.
        let mut rng_mha_next = StdRng::seed_from_u64(0xA11CE_1402);
        let mha_next = Gpt::new_with_attention_heads(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::LearnedAbsolute,
            4,
            &mut rng_mha_next,
        );
        let after_mha: u64 = rng_mha_next.gen();
        let mut rng_mqa_next = StdRng::seed_from_u64(0xA11CE_1402);
        let _mqa_next = Gpt::new_with_attention_heads(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::LearnedAbsolute,
            1,
            &mut rng_mqa_next,
        );
        let after_mqa: u64 = rng_mqa_next.gen();
        assert_eq!(after_mqa, after_mha);
        assert_eq!(mha_next.n_kv_head(), 4);
    }

    #[test]
    fn generation_stays_inside_vocab() {
        let cfg = Config {
            vocab: 5,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let mut rng = rand::thread_rng();
        let gpt = Gpt::new(cfg, &mut rng);
        let mut ids = vec![0, 1];
        gpt.generate(&mut ids, 8, 0.8, &mut rng);
        assert_eq!(ids.len(), 10);
        assert!(ids.iter().all(|&x| x < cfg.vocab));
    }
    #[test]
    fn kv_cache_prefill_is_bit_exact_for_all_position_policies() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 8,
            n_ff: 16,
        };
        let tokens = [0, 1, 2, 3, 4, 5];

        for position in [
            PositionKind::LearnedAbsolute,
            PositionKind::Rope,
            PositionKind::Alibi,
        ] {
            let mut rng = StdRng::seed_from_u64(0xA11CE_4001);
            let mut gpt = Gpt::new(cfg, &mut rng);
            gpt.set_position_kind(position);
            let baseline = gpt.logits(&tokens);
            let params_before = gpt.collect_params();
            let mut cache = gpt.new_kv_cache();
            let cached = gpt.prefill_kv_cache(&tokens, &mut cache).unwrap();

            assert_eq!(cached, baseline, "position={}", position.as_str());
            assert_eq!(cache.len(), tokens.len());
            assert_eq!(cache.logical_bytes(), tokens.len() * cfg.n_layer * cfg.n_embd * 2 * 4);
            assert_eq!(gpt.collect_params(), params_before);
        }
    }

    #[test]
    fn kv_cache_incremental_decode_reset_clone_and_failures_are_explicit() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 6,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(0xA11CE_4002);
        let gpt = Gpt::new(cfg, &mut rng);
        let prefix = [0, 1, 2, 3];
        let mut cache = gpt.new_kv_cache();
        gpt.prefill_kv_cache(&prefix, &mut cache).unwrap();

        let mut cloned = cache.clone();
        let row = gpt.decode_kv_cached(4, &mut cloned).unwrap();
        let full = gpt.logits(&[0, 1, 2, 3, 4]);
        assert_eq!(row, full[4 * cfg.vocab..]);
        assert_eq!(cache.len(), 4);
        assert_eq!(cloned.len(), 5);

        cloned.reset();
        assert!(cloned.is_empty());
        assert_eq!(cache.len(), 4);

        let before = cache.len();
        assert!(gpt.decode_kv_cached(cfg.vocab, &mut cache).is_err());
        assert_eq!(cache.len(), before);
        assert!(gpt.prefill_kv_cache(&prefix, &mut cache).is_err());

        let mut full_cache = gpt.new_kv_cache();
        gpt.prefill_kv_cache(&[0, 1, 2, 3, 4, 5], &mut full_cache)
            .unwrap();
        assert!(gpt.decode_kv_cached(6, &mut full_cache).is_err());
        assert_eq!(full_cache.len(), cfg.block);
    }

    #[test]
    fn recurrent_one_step_is_bit_exact_with_baseline() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(0xA11CE_4101);
        let gpt = Gpt::new(cfg, &mut rng);
        let x = [0, 1, 2, 3];
        let y = [1, 2, 3, 4];
        let recurrent = RecurrentConfig::new(1).unwrap();

        assert_eq!(gpt.logits_recurrent(&x, recurrent).unwrap(), gpt.logits(&x));
        assert_eq!(
            gpt.loss_recurrent(&x, &y, recurrent).unwrap().to_bits(),
            gpt.loss(&x, &y).to_bits()
        );

        let mut baseline_grads = vec![0.0; gpt.collect_params().len()];
        let mut recurrent_grads = vec![0.0; baseline_grads.len()];
        let baseline_loss = gpt.backward_into(&x, &y, &mut baseline_grads);
        let recurrent_loss = gpt
            .backward_recurrent_into(&x, &y, &mut recurrent_grads, recurrent)
            .unwrap();
        assert_eq!(recurrent_loss.to_bits(), baseline_loss.to_bits());
        assert_eq!(recurrent_grads, baseline_grads);
    }

    #[test]
    fn recurrent_multi_step_is_finite_and_keeps_parameter_budget_fixed() {
        let cfg = Config {
            vocab: 11,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(0xA11CE_4102);
        let gpt = Gpt::new(cfg, &mut rng);
        let params_before = gpt.collect_params();
        let x = [0, 1, 2, 3];
        let y = [1, 2, 3, 4];

        let baseline_logits = gpt.logits(&x);
        for steps in [2usize, 3] {
            let recurrent = RecurrentConfig::new(steps).unwrap();
            let logits = gpt.logits_recurrent(&x, recurrent).unwrap();
            assert!(logits.iter().all(|value| value.is_finite()));
            assert_ne!(logits, baseline_logits);

            let eval_loss = gpt.loss_recurrent(&x, &y, recurrent).unwrap();
            let mut grads = vec![0.0; params_before.len()];
            let backward_loss = gpt
                .backward_recurrent_into(&x, &y, &mut grads, recurrent)
                .unwrap();
            assert!(eval_loss.is_finite() && backward_loss.is_finite());
            assert!(
                (eval_loss - backward_loss).abs() < 1e-6,
                "steps={steps} eval={eval_loss} backward={backward_loss}"
            );
            assert!(grads.iter().all(|value| value.is_finite()));
            assert!(grads.iter().any(|value| value.abs() > 0.0));
            assert_eq!(
                recurrent.block_applications(cfg.n_layer).unwrap(),
                steps * cfg.n_layer
            );
        }

        assert_eq!(gpt.collect_params(), params_before);
    }

    #[test]
    fn recurrent_two_step_gradient_matches_finite_difference() {
        let cfg = Config {
            vocab: 5,
            n_embd: 4,
            n_head: 1,
            n_layer: 1,
            block: 3,
            n_ff: 8,
        };
        let mut rng = StdRng::seed_from_u64(0xA11CE_4103);
        let gpt = Gpt::new(cfg, &mut rng);
        let x = [0, 1, 2];
        let y = [1, 2, 3];
        let recurrent = RecurrentConfig::new(2).unwrap();
        let params = gpt.collect_params();
        let mut analytic = vec![0.0; params.len()];
        gpt.backward_recurrent_into(&x, &y, &mut analytic, recurrent)
            .unwrap();

        let tok_index = 0usize;
        let pos_index = cfg.vocab * cfg.n_embd;
        let block_base = cfg.vocab * cfg.n_embd + cfg.block * cfg.n_embd;
        let wq_index = block_base + 2 * cfg.n_embd;
        let w_out_start = params.len() - cfg.vocab - cfg.n_embd * cfg.vocab;
        let probe_indices = [tok_index, pos_index, wq_index, w_out_start];

        let h = 1e-3f32;
        for &index in &probe_indices {
            let mut plus_params = params.clone();
            let mut minus_params = params.clone();
            plus_params[index] += h;
            minus_params[index] -= h;

            let mut plus = gpt.clone();
            plus.write_params(&plus_params);
            let mut minus = gpt.clone();
            minus.write_params(&minus_params);

            let plus_loss = plus.loss_recurrent(&x, &y, recurrent).unwrap();
            let minus_loss = minus.loss_recurrent(&x, &y, recurrent).unwrap();
            let numeric = (plus_loss - minus_loss) / (2.0 * h);
            let error = (analytic[index] - numeric).abs();
            assert!(
                error < 4e-3,
                "recurrent grad mismatch index={index} analytic={} numeric={numeric} error={error}",
                analytic[index]
            );
        }
    }


}
