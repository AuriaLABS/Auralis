//! Decoder-only Transformer from first principles.
//!
//! No deep-learning framework is used here: forward pass, causal multi-head
//! attention, layer normalization, GELU, cross-entropy and backward pass are
//! implemented explicitly over `Vec<f32>`.

use crate::arena::Arena;
use crate::attention::{Attention, AttentionShape, RowSlicesAttention};
use crate::backend::{
    Backend, BackendId, MatrixMut, MatrixRef, OptimizedCpuBackend, ScalarCpuBackend,
};
use crate::layer_diagnostics::{
    cosine_similarity, summarize_tensor, AdjacentLayerCosine, GradientLayerSummary,
    LayerDiagnosticsReport, LayerHooks, LayerTensorSummary,
};
use crate::numeric::{explain, scan_f32, Scan, Stage};
use crate::position::{LearnedAbsolute, PositionalEncoding, TrainablePositionalEncoding};
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

#[derive(Clone)]
struct RmsCache {
    xhat: Vec<f32>,
    inv_rms: Vec<f32>,
    rows: usize,
    cols: usize,
}

enum NormCache {
    LayerNorm(LnCache),
    RmsNorm(RmsCache),
}

struct LayerCache {
    ln1: NormCache,
    h1: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    probs: Vec<f32>,
    att: Vec<f32>,
    ln2: NormCache,
    h2: Vec<f32>,
    ff_pre: Vec<f32>,
    ff_act: Vec<f32>,
}

struct ForwardCache {
    layers: Vec<LayerCache>,
    ln_f: NormCache,
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
    grads: GptGrad,
    scratch: Arena,
    dx: Vec<f32>,
    residual: Vec<f32>,
}

fn init_vec(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

fn zeros_block_grad(cfg: Config) -> BlockGrad {
    let d = cfg.n_embd;
    BlockGrad {
        ln1_g: vec![0.0; d],
        ln1_b: vec![0.0; d],
        wq: vec![0.0; d * d],
        wk: vec![0.0; d * d],
        wv: vec![0.0; d * d],
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
            grads: gpt.zero_grads(),
            scratch: Arena::with_capacity(scratch_capacity),
            dx: vec![0.0; td],
            residual: vec![0.0; td],
        }
    }

    pub(crate) fn matches(&self, gpt: &Gpt) -> bool {
        self.cfg == gpt.cfg
    }

    fn clear(&mut self) {
        self.grads.clear();
        self.scratch.reset();
    }
}

impl Gpt {
    pub fn new(cfg: Config, rng: &mut impl Rng) -> Self {
        Self::new_with_normalization(cfg, NormalizationKind::LayerNorm, rng)
    }

    pub fn new_with_normalization(
        cfg: Config,
        normalization: NormalizationKind,
        rng: &mut impl Rng,
    ) -> Self {
        cfg.validate();
        let d = cfg.n_embd;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            blocks.push(Block {
                ln1_g: vec![1.0; d],
                ln1_b: vec![0.0; d],
                wq: init_vec(rng, d * d, 0.02),
                wk: init_vec(rng, d * d, 0.02),
                wv: init_vec(rng, d * d, 0.02),
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

        let positional = LearnedAbsolute::new(&self.pos_emb, self.cfg.block, d)
            .expect("model positional storage matches config");
        positional
            .add_forward(&mut x, t)
            .expect("token length already validated against positional capacity");
        x
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
            normalization_backward_into(dx, &cache.ln_f, &self.ln_f_g, dx_ln, dgamma, dbeta);
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
                normalization_backward_into(residual, &c.ln2, &b.ln2_g, dln2, dg2, db2);
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
            let dq_slot = scratch.alloc(td);
            let dk_slot = scratch.alloc(td);
            let dv_slot = scratch.alloc(td);
            let dp_slot = scratch.alloc(t);
            {
                let (dq, dk, dv, dp) = scratch.get4_mut(dq_slot, dk_slot, dv_slot, dp_slot);
                attention_backward_into(
                    dx,
                    &c.q,
                    &c.k,
                    &c.v,
                    &c.probs,
                    t,
                    d,
                    self.cfg.n_head,
                    dq,
                    dk,
                    dv,
                    dp,
                );

                matmul_grad_b(&c.h1, t, d, dq, d, &mut bg.wq);
                matmul_grad_b(&c.h1, t, d, dk, d, &mut bg.wk);
                matmul_grad_b(&c.h1, t, d, dv, d, &mut bg.wv);

                matmul_b_t_into(dq, t, d, &b.wq, d, dx);
                matmul_b_t_add_into(dk, t, d, &b.wk, d, dx);
                matmul_b_t_add_into(dv, t, d, &b.wv, d, dx);
            }

            scratch.reset();
            let dln1_slot = scratch.alloc(td);
            let dg1_slot = scratch.alloc(d);
            let db1_slot = scratch.alloc(d);
            {
                let (dln1, dg1, db1) = scratch.get3_mut(dln1_slot, dg1_slot, db1_slot);
                normalization_backward_into(dx, &c.ln1, &b.ln1_g, dln1, dg1, db1);
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

        let positional = LearnedAbsolute::new(&self.pos_emb, self.cfg.block, d)
            .expect("model positional storage matches config");
        positional
            .accumulate_backward(dx, t, &mut gg.pos_emb)
            .expect("backward positional shapes match validated forward");

        copy_grads_into(gg, grads);
        loss
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

    /// Forward path for evaluation/inference. It preserves the training-forward
    /// arithmetic order but does not materialize caches used only by backward.
    fn forward_eval(&self, tokens: &[usize]) -> Vec<f32> {
        self.forward_eval_with_backend(tokens, &OPTIMIZED_CPU_BACKEND)
    }

    fn forward_eval_with_backend<B: Backend>(
        &self,
        tokens: &[usize],
        backend: &B,
    ) -> Vec<f32> {
        let t = tokens.len();
        let d = self.cfg.n_embd;
        let mut x = self.embed_tokens_with_positions(tokens);

        for b in &self.blocks {
            let h1 = normalization_eval(self.normalization, &x, t, d, &b.ln1_g, &b.ln1_b);
            let q = matmul_with_backend(backend, &h1, t, d, &b.wq, d);
            let k = matmul_with_backend(backend, &h1, t, d, &b.wk, d);
            let v = matmul_with_backend(backend, &h1, t, d, &b.wv, d);
            let att = attention_eval(&q, &k, &v, t, d, self.cfg.n_head);
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

        let h_final = normalization_eval(self.normalization, &x, t, d, &self.ln_f_g, &self.ln_f_b);
        let mut logits = matmul_with_backend(backend, &h_final, t, d, &self.w_out, self.cfg.vocab);
        add_bias_inplace(&mut logits, t, self.cfg.vocab, &self.b_out);
        logits
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
            let q = matmul_with_backend(backend, &h1, t, d, &b.wq, d);
            let k = matmul_with_backend(backend, &h1, t, d, &b.wk, d);
            let v = matmul_with_backend(backend, &h1, t, d, &b.wv, d);
            let (att, probs) = attention_forward(&q, &k, &v, t, d, self.cfg.n_head);
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
                .map(|_| zeros_block_grad(self.cfg))
                .collect(),
            ln_f_g: vec![0.0; self.ln_f_g.len()],
            ln_f_b: vec![0.0; self.ln_f_b.len()],
            w_out: vec![0.0; self.w_out.len()],
            b_out: vec![0.0; self.b_out.len()],
        }
    }

    fn block_param_count(&self) -> usize {
        let d = self.cfg.n_embd;
        4 * d * d
            + 2 * d
            + 2 * d
            + d * self.cfg.n_ff
            + self.cfg.n_ff
            + self.cfg.n_ff * d
            + d
    }

    fn param_count(&self) -> usize {
        let per_block = self.block_param_count();
        self.tok_emb.len()
            + self.pos_emb.len()
            + self.cfg.n_layer * per_block
            + self.ln_f_g.len()
            + self.ln_f_b.len()
            + self.w_out.len()
            + self.b_out.len()
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
) -> (Vec<f32>, NormCache) {
    match kind {
        NormalizationKind::LayerNorm => {
            let (y, cache) = layernorm_forward(x, rows, cols, gamma, beta);
            (y, NormCache::LayerNorm(cache))
        }
        NormalizationKind::RmsNorm => {
            let (y, cache) = rmsnorm_forward(x, rows, cols, gamma);
            (y, NormCache::RmsNorm(cache))
        }
    }
}

fn normalization_backward_into(
    dy: &[f32],
    cache: &NormCache,
    gamma: &[f32],
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) {
    match cache {
        NormCache::LayerNorm(cache) => {
            layernorm_backward_into(dy, cache, gamma, dx, dgamma, dbeta)
        }
        NormCache::RmsNorm(cache) => {
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
) -> (Vec<f32>, RmsCache) {
    const EPS: f32 = 1e-5;
    assert_eq!(x.len(), rows * cols);
    assert_eq!(gamma.len(), cols);
    let mut y = vec![0.0; x.len()];
    let mut xhat = vec![0.0; x.len()];
    let mut inv_rms = vec![0.0; rows];
    for i in 0..rows {
        let row = &x[i * cols..(i + 1) * cols];
        let mean_sq = row.iter().map(|v| *v * *v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        inv_rms[i] = inv;
        for j in 0..cols {
            let idx = i * cols + j;
            let h = x[idx] * inv;
            xhat[idx] = h;
            y[idx] = h * gamma[j];
        }
    }
    (
        y,
        RmsCache {
            xhat,
            inv_rms,
            rows,
            cols,
        },
    )
}

fn rmsnorm_backward_into(
    dy: &[f32],
    cache: &RmsCache,
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
            dx[idx] = cache.inv_rms[i] * (z - cache.xhat[idx] * mean_gxh);
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
) -> Vec<f32> {
    let shape = AttentionShape {
        tokens: t,
        width: d,
        heads: n_head,
    };
    let mut out = vec![0.0; t * d];
    let mut scores = vec![0.0; t];
    OPTIMIZED_ATTENTION
        .forward_eval(q, k, v, shape, &mut out, &mut scores)
        .expect("model attention shapes are validated by Config");
    out
}

fn attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
) -> (Vec<f32>, Vec<f32>) {
    let shape = AttentionShape {
        tokens: t,
        width: d,
        heads: n_head,
    };
    let mut probs = vec![0.0; n_head * t * t];
    let mut out = vec![0.0; t * d];
    OPTIMIZED_ATTENTION
        .forward_cached(q, k, v, shape, &mut out, &mut probs)
        .expect("model attention shapes are validated by Config");
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
    use super::{BackendId, BackwardWorkspace, Config, CpuBackend, Gpt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
}
