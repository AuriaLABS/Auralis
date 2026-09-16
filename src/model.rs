//! Decoder-only Transformer from first principles.
//!
//! No deep-learning framework is used here: forward pass, causal multi-head
//! attention, layer normalization, GELU, cross-entropy and backward pass are
//! implemented explicitly over `Vec<f32>`.

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
    tokens: Vec<usize>,
    layers: Vec<LayerCache>,
    ln_f: LnCache,
    h_final: Vec<f32>,
}

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

struct GptGrad {
    tok_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    blocks: Vec<BlockGrad>,
    ln_f_g: Vec<f32>,
    ln_f_b: Vec<f32>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
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

impl Gpt {
    pub fn new(cfg: Config, rng: &mut impl Rng) -> Self {
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
            tok_emb: init_vec(rng, cfg.vocab * d, 0.02),
            pos_emb: init_vec(rng, cfg.block * d, 0.02),
            blocks,
            ln_f_g: vec![1.0; d],
            ln_f_b: vec![0.0; d],
            w_out: init_vec(rng, d * cfg.vocab, 0.02),
            b_out: vec![0.0; cfg.vocab],
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

    /// Mean next-token cross-entropy without constructing or propagating
    /// parameter gradients. This is the canonical evaluation path.
    pub fn loss(&self, x: &[usize], y: &[usize]) -> f32 {
        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty() && x.len() <= self.cfg.block);
        assert!(x.iter().all(|&t| t < self.cfg.vocab));
        assert!(y.iter().all(|&t| t < self.cfg.vocab));

        let (logits, _) = self.forward_internal(x);
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
        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty() && x.len() <= self.cfg.block);
        assert_eq!(grads.len(), self.param_count());
        assert!(x.iter().all(|&t| t < self.cfg.vocab));
        assert!(y.iter().all(|&t| t < self.cfg.vocab));

        let (logits, cache) = self.forward_internal(x);
        let t = x.len();
        let v = self.cfg.vocab;
        let mut dlogits = vec![0.0; t * v];
        let mut loss = 0.0f32;

        for i in 0..t {
            let row = &logits[i * v..(i + 1) * v];
            let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..v {
                let e = (row[j] - maxv).exp();
                dlogits[i * v + j] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-20);
            for j in 0..v {
                dlogits[i * v + j] *= inv;
            }
            let p = dlogits[i * v + y[i]].max(1e-20);
            loss -= p.ln();
            dlogits[i * v + y[i]] -= 1.0;
        }
        let inv_t = 1.0 / t as f32;
        loss *= inv_t;
        for g in &mut dlogits {
            *g *= inv_t;
        }

        let mut gg = self.zero_grads();
        let d = self.cfg.n_embd;

        matmul_grad_b(
            &cache.h_final,
            t,
            d,
            &dlogits,
            v,
            &mut gg.w_out,
        );
        sum_rows_into(&dlogits, t, v, &mut gg.b_out);
        let mut dx = matmul_b_t(&dlogits, t, v, &self.w_out, d);

        let (dx_ln, dgamma, dbeta) =
            layernorm_backward(&dx, &cache.ln_f, &self.ln_f_g);
        dx = dx_ln;
        add_inplace(&mut gg.ln_f_g, &dgamma);
        add_inplace(&mut gg.ln_f_b, &dbeta);

        for li in (0..self.blocks.len()).rev() {
            let b = &self.blocks[li];
            let c = &cache.layers[li];
            let bg = &mut gg.blocks[li];

            let mut dr1 = dx.clone();
            let dff_out = dx;

            matmul_grad_b(&c.ff_act, t, self.cfg.n_ff, &dff_out, d, &mut bg.w2);
            sum_rows_into(&dff_out, t, d, &mut bg.b2);
            let dff_act = matmul_b_t(&dff_out, t, d, &b.w2, self.cfg.n_ff);

            let mut dff_pre = dff_act;
            for i in 0..dff_pre.len() {
                dff_pre[i] *= gelu_deriv(c.ff_pre[i]);
            }

            matmul_grad_b(&c.h2, t, d, &dff_pre, self.cfg.n_ff, &mut bg.w1);
            sum_rows_into(&dff_pre, t, self.cfg.n_ff, &mut bg.b1);
            let dh2 = matmul_b_t(&dff_pre, t, self.cfg.n_ff, &b.w1, d);

            let (dln2, dg2, db2) = layernorm_backward(&dh2, &c.ln2, &b.ln2_g);
            add_inplace(&mut bg.ln2_g, &dg2);
            add_inplace(&mut bg.ln2_b, &db2);
            add_inplace(&mut dr1, &dln2);

            let dproj = dr1.clone();
            let mut dx_in = dr1;

            matmul_grad_b(&c.att, t, d, &dproj, d, &mut bg.wo);
            let datt = matmul_b_t(&dproj, t, d, &b.wo, d);

            let (dq, dk, dv) = attention_backward(
                &datt,
                &c.q,
                &c.k,
                &c.v,
                &c.probs,
                t,
                d,
                self.cfg.n_head,
            );

            matmul_grad_b(&c.h1, t, d, &dq, d, &mut bg.wq);
            matmul_grad_b(&c.h1, t, d, &dk, d, &mut bg.wk);
            matmul_grad_b(&c.h1, t, d, &dv, d, &mut bg.wv);

            let mut dh1 = matmul_b_t(&dq, t, d, &b.wq, d);
            add_inplace(&mut dh1, &matmul_b_t(&dk, t, d, &b.wk, d));
            add_inplace(&mut dh1, &matmul_b_t(&dv, t, d, &b.wv, d));

            let (dln1, dg1, db1) = layernorm_backward(&dh1, &c.ln1, &b.ln1_g);
            add_inplace(&mut bg.ln1_g, &dg1);
            add_inplace(&mut bg.ln1_b, &db1);
            add_inplace(&mut dx_in, &dln1);
            dx = dx_in;
        }

        for i in 0..t {
            let tok = cache.tokens[i];
            for j in 0..d {
                let g = dx[i * d + j];
                gg.tok_emb[tok * d + j] += g;
                gg.pos_emb[i * d + j] += g;
            }
        }

        copy_grads_into(&gg, grads);
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
            let (logits, _) = self.forward_internal(ctx);
            let row = &logits[(ctx.len() - 1) * self.cfg.vocab..ctx.len() * self.cfg.vocab];
            let next = sample_logits(row, temperature, rng);
            ids.push(next);
        }
    }

    fn forward_internal(&self, tokens: &[usize]) -> (Vec<f32>, ForwardCache) {
        assert!(!tokens.is_empty() && tokens.len() <= self.cfg.block);
        let t = tokens.len();
        let d = self.cfg.n_embd;
        let mut x = vec![0.0; t * d];
        for i in 0..t {
            let tok = tokens[i];
            assert!(tok < self.cfg.vocab);
            for j in 0..d {
                x[i * d + j] =
                    self.tok_emb[tok * d + j] + self.pos_emb[i * d + j];
            }
        }

        let mut layer_caches = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let (h1, ln1) = layernorm_forward(&x, t, d, &b.ln1_g, &b.ln1_b);
            let q = matmul(&h1, t, d, &b.wq, d);
            let k = matmul(&h1, t, d, &b.wk, d);
            let v = matmul(&h1, t, d, &b.wv, d);
            let (att, probs) =
                attention_forward(&q, &k, &v, t, d, self.cfg.n_head);
            let proj = matmul(&att, t, d, &b.wo, d);
            let mut r1 = x.clone();
            add_inplace(&mut r1, &proj);

            let (h2, ln2) = layernorm_forward(&r1, t, d, &b.ln2_g, &b.ln2_b);
            let mut ff_pre = matmul(&h2, t, d, &b.w1, self.cfg.n_ff);
            add_bias_inplace(&mut ff_pre, t, self.cfg.n_ff, &b.b1);
            let ff_act: Vec<f32> = ff_pre.iter().copied().map(gelu).collect();
            let mut ff_out = matmul(&ff_act, t, self.cfg.n_ff, &b.w2, d);
            add_bias_inplace(&mut ff_out, t, d, &b.b2);
            let mut out = r1;
            add_inplace(&mut out, &ff_out);

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

        let (h_final, ln_f) =
            layernorm_forward(&x, t, d, &self.ln_f_g, &self.ln_f_b);
        let mut logits = matmul(&h_final, t, d, &self.w_out, self.cfg.vocab);
        add_bias_inplace(&mut logits, t, self.cfg.vocab, &self.b_out);

        (
            logits,
            ForwardCache {
                tokens: tokens.to_vec(),
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

    fn param_count(&self) -> usize {
        let d = self.cfg.n_embd;
        let per_block =
            4 * d * d + 2 * d + 2 * d + d * self.cfg.n_ff
                + self.cfg.n_ff
                + self.cfg.n_ff * d
                + d;
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

fn matmul(a: &[f32], rows: usize, inner: usize, b: &[f32], cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(b.len(), inner * cols);
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for k in 0..inner {
            let av = a[i * inner + k];
            for j in 0..cols {
                out[i * cols + j] += av * b[k * cols + j];
            }
        }
    }
    out
}

fn matmul_grad_b(
    a: &[f32],
    rows: usize,
    inner: usize,
    dy: &[f32],
    cols: usize,
    db: &mut [f32],
) {
    assert_eq!(db.len(), inner * cols);
    for k in 0..inner {
        for j in 0..cols {
            let mut s = 0.0;
            for i in 0..rows {
                s += a[i * inner + k] * dy[i * cols + j];
            }
            db[k * cols + j] += s;
        }
    }
}

fn matmul_b_t(
    dy: &[f32],
    rows: usize,
    out_cols: usize,
    b: &[f32],
    result_cols: usize,
) -> Vec<f32> {
    assert_eq!(dy.len(), rows * out_cols);
    assert_eq!(b.len(), result_cols * out_cols);
    let mut out = vec![0.0; rows * result_cols];
    for i in 0..rows {
        for k in 0..result_cols {
            let mut s = 0.0;
            for j in 0..out_cols {
                s += dy[i * out_cols + j] * b[k * out_cols + j];
            }
            out[i * result_cols + k] = s;
        }
    }
    out
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

fn layernorm_backward(
    dy: &[f32],
    cache: &LnCache,
    gamma: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = cache.rows;
    let cols = cache.cols;
    let mut dx = vec![0.0; dy.len()];
    let mut dgamma = vec![0.0; cols];
    let mut dbeta = vec![0.0; cols];

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
            dx[idx] = scale
                * (cols as f32 * z - sum_g - cache.xhat[idx] * sum_gxh);
        }
    }
    (dx, dgamma, dbeta)
}

fn attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
) -> (Vec<f32>, Vec<f32>) {
    let hd = d / n_head;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut probs = vec![0.0; n_head * t * t];
    let mut out = vec![0.0; t * d];

    for h in 0..n_head {
        let hoff = h * hd;
        for i in 0..t {
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..=i {
                let mut s = 0.0;
                for z in 0..hd {
                    s += q[i * d + hoff + z] * k[j * d + hoff + z];
                }
                s *= scale;
                let idx = (h * t + i) * t + j;
                probs[idx] = s;
                max_score = max_score.max(s);
            }
            let mut sum = 0.0;
            for j in 0..=i {
                let idx = (h * t + i) * t + j;
                let e = (probs[idx] - max_score).exp();
                probs[idx] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-20);
            for j in 0..=i {
                let pidx = (h * t + i) * t + j;
                probs[pidx] *= inv;
                let p = probs[pidx];
                for z in 0..hd {
                    out[i * d + hoff + z] += p * v[j * d + hoff + z];
                }
            }
        }
    }
    (out, probs)
}

fn attention_backward(
    dout: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hd = d / n_head;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut dq = vec![0.0; t * d];
    let mut dk = vec![0.0; t * d];
    let mut dv = vec![0.0; t * d];
    let mut dp = vec![0.0; t];

    for h in 0..n_head {
        let hoff = h * hd;
        for i in 0..t {
            dp[..=i].fill(0.0);
            for j in 0..=i {
                let p = probs[(h * t + i) * t + j];
                for z in 0..hd {
                    let go = dout[i * d + hoff + z];
                    dp[j] += go * v[j * d + hoff + z];
                    dv[j * d + hoff + z] += p * go;
                }
            }
            let mut dot = 0.0;
            for j in 0..=i {
                dot += dp[j] * probs[(h * t + i) * t + j];
            }
            for j in 0..=i {
                let p = probs[(h * t + i) * t + j];
                let ds = p * (dp[j] - dot) * scale;
                for z in 0..hd {
                    let qi = q[i * d + hoff + z];
                    let kj = k[j * d + hoff + z];
                    dq[i * d + hoff + z] += ds * kj;
                    dk[j * d + hoff + z] += ds * qi;
                }
            }
        }
    }
    (dq, dk, dv)
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
    use super::{Config, Gpt};

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
