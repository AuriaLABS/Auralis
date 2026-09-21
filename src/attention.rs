//! Attention boundary for Brain architecture experiments.
//!
//! The supported semantic contract is causal multi-head scaled dot-product
//! attention. Eval keeps O(tokens) scratch, while training exposes the full
//! probability cache required by backward.

use crate::kernels::{
    attention_backward_reference_into, attention_backward_row_slices_into,
    attention_forward_reference_into, attention_forward_row_slices_alibi_into,
    attention_forward_row_slices_into,
};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionId {
    Reference,
    RowSlices,
}

impl AttentionId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::RowSlices => "row_slices",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionShape {
    pub tokens: usize,
    pub width: usize,
    pub heads: usize,
}

impl AttentionShape {
    pub fn validate(self) -> Result<Self, AttentionError> {
        if self.tokens == 0 || self.width == 0 || self.heads == 0 {
            return Err(AttentionError::ZeroDimension {
                tokens: self.tokens,
                width: self.width,
                heads: self.heads,
            });
        }
        if self.width % self.heads != 0 {
            return Err(AttentionError::HeadWidthMismatch {
                width: self.width,
                heads: self.heads,
            });
        }
        self.activation_len()?;
        self.probs_len()?;
        Ok(self)
    }

    pub fn head_width(self) -> Result<usize, AttentionError> {
        self.validate().map(|shape| shape.width / shape.heads)
    }

    pub fn activation_len(self) -> Result<usize, AttentionError> {
        self.tokens
            .checked_mul(self.width)
            .ok_or(AttentionError::SizeOverflow)
    }

    pub fn probs_len(self) -> Result<usize, AttentionError> {
        self.heads
            .checked_mul(self.tokens)
            .and_then(|n| n.checked_mul(self.tokens))
            .ok_or(AttentionError::SizeOverflow)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttentionError {
    ZeroDimension { tokens: usize, width: usize, heads: usize },
    HeadWidthMismatch { width: usize, heads: usize },
    SizeOverflow,
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { tokens, width, heads } => write!(
                f,
                "attention requires positive dimensions: tokens={tokens} width={width} heads={heads}"
            ),
            Self::HeadWidthMismatch { width, heads } => write!(
                f,
                "attention width {width} is not divisible by heads {heads}"
            ),
            Self::SizeOverflow => write!(f, "attention shape size overflow"),
            Self::LengthMismatch { tensor, expected, actual } => write!(
                f,
                "attention {tensor} length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for AttentionError {}

pub trait Attention {
    fn id(&self) -> AttentionId;

    fn forward_eval(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AttentionShape,
        out: &mut [f32],
        scores: &mut [f32],
    ) -> Result<(), AttentionError>;

    fn forward_cached(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AttentionShape,
        out: &mut [f32],
        probs: &mut [f32],
    ) -> Result<(), AttentionError>;

    fn backward(
        &self,
        dout: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        probs: &[f32],
        shape: AttentionShape,
        dq: &mut [f32],
        dk: &mut [f32],
        dv: &mut [f32],
        dp: &mut [f32],
    ) -> Result<(), AttentionError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceAttention;

#[derive(Clone, Copy, Debug, Default)]
pub struct RowSlicesAttention;

fn validate_qkv(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
) -> Result<usize, AttentionError> {
    let shape = shape.validate()?;
    let expected = shape.activation_len()?;
    for (tensor, actual) in [("q", q.len()), ("k", k.len()), ("v", v.len())] {
        if actual != expected {
            return Err(AttentionError::LengthMismatch {
                tensor,
                expected,
                actual,
            });
        }
    }
    Ok(expected)
}

fn validate_len(
    tensor: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), AttentionError> {
    if actual != expected {
        return Err(AttentionError::LengthMismatch {
            tensor,
            expected,
            actual,
        });
    }
    Ok(())
}

fn forward_eval_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    out: &mut [f32],
    scores: &mut [f32],
) -> Result<(), AttentionError> {
    let expected = validate_qkv(q, k, v, shape)?;
    validate_len("out", out.len(), expected)?;
    validate_len("scores", scores.len(), shape.tokens)?;
    out.fill(0.0);
    scores.fill(0.0);

    let t = shape.tokens;
    let d = shape.width;
    let n_head = shape.heads;
    let hd = d / n_head;
    let scale = 1.0 / (hd as f32).sqrt();

    for h in 0..n_head {
        let hoff = h * hd;
        for i in 0..t {
            let q_head = &q[i * d + hoff..i * d + hoff + hd];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..=i {
                let k_head = &k[j * d + hoff..j * d + hoff + hd];
                let mut score = 0.0f32;
                for (&qv, &kv) in q_head.iter().zip(k_head) {
                    score += qv * kv;
                }
                score *= scale;
                scores[j] = score;
                max_score = max_score.max(score);
            }

            let mut sum = 0.0f32;
            for score in &mut scores[..=i] {
                let e = (*score - max_score).exp();
                *score = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-20);
            let out_head = &mut out[i * d + hoff..i * d + hoff + hd];
            for j in 0..=i {
                scores[j] *= inv;
                let p = scores[j];
                let v_head = &v[j * d + hoff..j * d + hoff + hd];
                for (dst, &vv) in out_head.iter_mut().zip(v_head) {
                    *dst += p * vv;
                }
            }
        }
    }
    Ok(())
}

impl Attention for ReferenceAttention {
    fn id(&self) -> AttentionId { AttentionId::Reference }

    fn forward_eval(
        &self, q: &[f32], k: &[f32], v: &[f32], shape: AttentionShape,
        out: &mut [f32], scores: &mut [f32],
    ) -> Result<(), AttentionError> {
        forward_eval_reference(q, k, v, shape, out, scores)
    }

    fn forward_cached(
        &self, q: &[f32], k: &[f32], v: &[f32], shape: AttentionShape,
        out: &mut [f32], probs: &mut [f32],
    ) -> Result<(), AttentionError> {
        let expected = validate_qkv(q, k, v, shape)?;
        validate_len("out", out.len(), expected)?;
        validate_len("probs", probs.len(), shape.probs_len()?)?;
        attention_forward_reference_into(
            q, k, v, shape.tokens, shape.width, shape.heads, out, probs,
        );
        Ok(())
    }

    fn backward(
        &self, dout: &[f32], q: &[f32], k: &[f32], v: &[f32], probs: &[f32],
        shape: AttentionShape, dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
        dp: &mut [f32],
    ) -> Result<(), AttentionError> {
        validate_backward(dout, q, k, v, probs, shape, dq, dk, dv, dp)?;
        attention_backward_reference_into(
            dout, q, k, v, probs, shape.tokens, shape.width, shape.heads, dq, dk, dv, dp,
        );
        Ok(())
    }
}

impl Attention for RowSlicesAttention {
    fn id(&self) -> AttentionId { AttentionId::RowSlices }

    fn forward_eval(
        &self, q: &[f32], k: &[f32], v: &[f32], shape: AttentionShape,
        out: &mut [f32], scores: &mut [f32],
    ) -> Result<(), AttentionError> {
        forward_eval_reference(q, k, v, shape, out, scores)
    }

    fn forward_cached(
        &self, q: &[f32], k: &[f32], v: &[f32], shape: AttentionShape,
        out: &mut [f32], probs: &mut [f32],
    ) -> Result<(), AttentionError> {
        let expected = validate_qkv(q, k, v, shape)?;
        validate_len("out", out.len(), expected)?;
        validate_len("probs", probs.len(), shape.probs_len()?)?;
        attention_forward_row_slices_into(
            q, k, v, shape.tokens, shape.width, shape.heads, out, probs,
        );
        Ok(())
    }

    fn backward(
        &self, dout: &[f32], q: &[f32], k: &[f32], v: &[f32], probs: &[f32],
        shape: AttentionShape, dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
        dp: &mut [f32],
    ) -> Result<(), AttentionError> {
        validate_backward(dout, q, k, v, probs, shape, dq, dk, dv, dp)?;
        attention_backward_row_slices_into(
            dout, q, k, v, probs, shape.tokens, shape.width, shape.heads, dq, dk, dv, dp,
        );
        Ok(())
    }
}

impl RowSlicesAttention {
    pub fn forward_eval_alibi(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AttentionShape,
        head_slopes: &[f32],
        out: &mut [f32],
        scores: &mut [f32],
    ) -> Result<(), AttentionError> {
        let expected = validate_qkv(q, k, v, shape)?;
        validate_len("out", out.len(), expected)?;
        validate_len("scores", scores.len(), shape.tokens)?;
        validate_len("head_slopes", head_slopes.len(), shape.heads)?;
        out.fill(0.0);
        scores.fill(0.0);

        let t = shape.tokens;
        let d = shape.width;
        let hd = d / shape.heads;
        let scale = 1.0 / (hd as f32).sqrt();

        for h in 0..shape.heads {
            let hoff = h * hd;
            let slope = head_slopes[h];
            for i in 0..t {
                let q_head = &q[i * d + hoff..i * d + hoff + hd];
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=i {
                    let k_head = &k[j * d + hoff..j * d + hoff + hd];
                    let mut score = 0.0f32;
                    for (&qv, &kv) in q_head.iter().zip(k_head) {
                        score += qv * kv;
                    }
                    score *= scale;
                    score += slope * (j as f32 - i as f32);
                    scores[j] = score;
                    max_score = max_score.max(score);
                }

                let mut sum = 0.0f32;
                for score in &mut scores[..=i] {
                    let e = (*score - max_score).exp();
                    *score = e;
                    sum += e;
                }
                let inv = 1.0 / sum.max(1e-20);
                let out_head = &mut out[i * d + hoff..i * d + hoff + hd];
                for j in 0..=i {
                    scores[j] *= inv;
                    let p = scores[j];
                    let v_head = &v[j * d + hoff..j * d + hoff + hd];
                    for (dst, &vv) in out_head.iter_mut().zip(v_head) {
                        *dst += p * vv;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn forward_cached_alibi(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AttentionShape,
        head_slopes: &[f32],
        out: &mut [f32],
        probs: &mut [f32],
    ) -> Result<(), AttentionError> {
        let expected = validate_qkv(q, k, v, shape)?;
        validate_len("out", out.len(), expected)?;
        validate_len("probs", probs.len(), shape.probs_len()?)?;
        validate_len("head_slopes", head_slopes.len(), shape.heads)?;
        attention_forward_row_slices_alibi_into(
            q,
            k,
            v,
            shape.tokens,
            shape.width,
            shape.heads,
            head_slopes,
            out,
            probs,
        );
        Ok(())
    }
}

fn validate_backward(
    dout: &[f32], q: &[f32], k: &[f32], v: &[f32], probs: &[f32],
    shape: AttentionShape, dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    dp: &mut [f32],
) -> Result<(), AttentionError> {
    let expected = validate_qkv(q, k, v, shape)?;
    validate_len("dout", dout.len(), expected)?;
    validate_len("probs", probs.len(), shape.probs_len()?)?;
    validate_len("dq", dq.len(), expected)?;
    validate_len("dk", dk.len(), expected)?;
    validate_len("dv", dv.len(), expected)?;
    validate_len("dp", dp.len(), shape.tokens)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let raw = ((i * 37 + salt * 19 + i / 3) % 101) as f32;
                (raw - 50.0) / 31.0
            })
            .collect()
    }

    #[test]
    fn cached_forward_reference_and_rowslices_are_exact_and_causal() {
        for shape in [
            AttentionShape { tokens: 1, width: 4, heads: 1 },
            AttentionShape { tokens: 4, width: 8, heads: 2 },
            AttentionShape { tokens: 9, width: 12, heads: 3 },
        ] {
            let n = shape.activation_len().unwrap();
            let p = shape.probs_len().unwrap();
            let q = data(n, 3);
            let k = data(n, 7);
            let v = data(n, 11);
            let mut ref_out = vec![f32::NAN; n];
            let mut ref_probs = vec![f32::NAN; p];
            let mut opt_out = vec![f32::NAN; n];
            let mut opt_probs = vec![f32::NAN; p];

            ReferenceAttention
                .forward_cached(&q, &k, &v, shape, &mut ref_out, &mut ref_probs).unwrap();
            RowSlicesAttention
                .forward_cached(&q, &k, &v, shape, &mut opt_out, &mut opt_probs).unwrap();
            assert_eq!(opt_out, ref_out);
            assert_eq!(opt_probs, ref_probs);

            for h in 0..shape.heads {
                for i in 0..shape.tokens {
                    for j in i + 1..shape.tokens {
                        assert_eq!(ref_probs[(h * shape.tokens + i) * shape.tokens + j], 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn eval_reference_and_optimized_are_exact_without_full_probs() {
        let shape = AttentionShape { tokens: 5, width: 8, heads: 2 };
        let n = shape.activation_len().unwrap();
        let q = data(n, 2);
        let k = data(n, 5);
        let v = data(n, 9);
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut sa = vec![0.0; shape.tokens];
        let mut sb = vec![0.0; shape.tokens];
        ReferenceAttention.forward_eval(&q, &k, &v, shape, &mut a, &mut sa).unwrap();
        RowSlicesAttention.forward_eval(&q, &k, &v, shape, &mut b, &mut sb).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn alibi_forward_is_causal_and_prefers_recent_keys_for_zero_qk() {
        let shape = AttentionShape {
            tokens: 4,
            width: 8,
            heads: 2,
        };
        let n = shape.activation_len().unwrap();
        let q = vec![0.0; n];
        let k = vec![0.0; n];
        let v = data(n, 17);
        let slopes = [0.25f32, 0.0625];
        let mut out = vec![0.0; n];
        let mut probs = vec![f32::NAN; shape.probs_len().unwrap()];

        RowSlicesAttention
            .forward_cached_alibi(&q, &k, &v, shape, &slopes, &mut out, &mut probs)
            .unwrap();

        for h in 0..shape.heads {
            for i in 0..shape.tokens {
                let row = &probs[(h * shape.tokens + i) * shape.tokens
                    ..(h * shape.tokens + i + 1) * shape.tokens];
                for &future in &row[i + 1..] {
                    assert_eq!(future, 0.0);
                }
                if i >= 2 {
                    assert!(row[i] > row[i - 1]);
                    assert!(row[i - 1] > row[i - 2]);
                }
                let causal_sum: f32 = row[..=i].iter().sum();
                assert!((causal_sum - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn alibi_cached_and_eval_paths_match_and_validate_slope_shape() {
        let shape = AttentionShape {
            tokens: 5,
            width: 8,
            heads: 2,
        };
        let n = shape.activation_len().unwrap();
        let q = data(n, 2);
        let k = data(n, 5);
        let v = data(n, 9);
        let slopes = [0.25f32, 0.0625];
        let mut eval = vec![0.0; n];
        let mut cached = vec![0.0; n];
        let mut scores = vec![0.0; shape.tokens];
        let mut probs = vec![0.0; shape.probs_len().unwrap()];

        RowSlicesAttention
            .forward_eval_alibi(&q, &k, &v, shape, &slopes, &mut eval, &mut scores)
            .unwrap();
        RowSlicesAttention
            .forward_cached_alibi(&q, &k, &v, shape, &slopes, &mut cached, &mut probs)
            .unwrap();
        assert_eq!(eval, cached);

        let err = RowSlicesAttention
            .forward_cached_alibi(&q, &k, &v, shape, &[0.25], &mut cached, &mut probs)
            .unwrap_err();
        assert!(matches!(
            err,
            AttentionError::LengthMismatch {
                tensor: "head_slopes",
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn backward_reference_and_rowslices_are_exact() {
        let shape = AttentionShape { tokens: 5, width: 8, heads: 2 };
        let n = shape.activation_len().unwrap();
        let p = shape.probs_len().unwrap();
        let q = data(n, 2);
        let k = data(n, 5);
        let v = data(n, 9);
        let dout = data(n, 13);
        let mut out = vec![0.0; n];
        let mut probs = vec![0.0; p];
        ReferenceAttention.forward_cached(&q, &k, &v, shape, &mut out, &mut probs).unwrap();

        let mut rdq = vec![0.0; n];
        let mut rdk = vec![0.0; n];
        let mut rdv = vec![0.0; n];
        let mut rdp = vec![0.0; shape.tokens];
        let mut odq = vec![0.0; n];
        let mut odk = vec![0.0; n];
        let mut odv = vec![0.0; n];
        let mut odp = vec![0.0; shape.tokens];

        ReferenceAttention.backward(
            &dout, &q, &k, &v, &probs, shape,
            &mut rdq, &mut rdk, &mut rdv, &mut rdp,
        ).unwrap();
        RowSlicesAttention.backward(
            &dout, &q, &k, &v, &probs, shape,
            &mut odq, &mut odk, &mut odv, &mut odp,
        ).unwrap();

        assert_eq!(odq, rdq);
        assert_eq!(odk, rdk);
        assert_eq!(odv, rdv);
        assert_eq!(odp, rdp);
    }

    #[test]
    fn invalid_shapes_fail_before_kernel_execution() {
        let shape = AttentionShape { tokens: 4, width: 10, heads: 3 };
        assert!(matches!(shape.validate(), Err(AttentionError::HeadWidthMismatch { .. })));

        let valid = AttentionShape { tokens: 4, width: 8, heads: 2 };
        let q = vec![0.0; 31];
        let k = vec![0.0; 32];
        let v = vec![0.0; 32];
        let mut out = vec![0.0; 32];
        let mut probs = vec![0.0; 32];
        assert!(matches!(
            RowSlicesAttention.forward_cached(&q, &k, &v, valid, &mut out, &mut probs),
            Err(AttentionError::LengthMismatch { tensor: "q", .. })
        ));
    }
}
