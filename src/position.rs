//! Positional-encoding boundary for Brain experiments.
//!
//! Learned absolute, RoPE and ALiBi share one explicit policy boundary.
//! The model owns historical positional parameter storage; this module owns
//! positional semantics while keeping experimental policies layout-compatible.

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PositionKind {
    LearnedAbsolute,
    Rope,
    Alibi,
}

impl PositionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LearnedAbsolute => "learned_absolute",
            Self::Rope => "rope",
            Self::Alibi => "alibi",
        }
    }
}

impl FromStr for PositionKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "learned_absolute" => Ok(Self::LearnedAbsolute),
            "rope" => Ok(Self::Rope),
            "alibi" => Ok(Self::Alibi),
            other => Err(format!("unknown positional encoding {other}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositionError {
    ZeroShape,
    StorageMismatch { expected: usize, actual: usize },
    ContextExceeded { positions: usize, max_positions: usize },
    ActivationMismatch { expected: usize, actual: usize },
    GradientMismatch { expected: usize, actual: usize },
    HeadWidthMismatch { width: usize, heads: usize },
}

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroShape => write!(f, "positional encoding requires positive block and width"),
            Self::StorageMismatch { expected, actual } => write!(
                f,
                "positional parameter storage mismatch: expected {expected}, got {actual}"
            ),
            Self::ContextExceeded { positions, max_positions } => write!(
                f,
                "context length {positions} exceeds positional capacity {max_positions}"
            ),
            Self::ActivationMismatch { expected, actual } => write!(
                f,
                "positional activation size mismatch: expected {expected}, got {actual}"
            ),
            Self::GradientMismatch { expected, actual } => write!(
                f,
                "positional gradient size mismatch: expected {expected}, got {actual}"
            ),
            Self::HeadWidthMismatch { width, heads } => write!(
                f,
                "RoPE requires an even per-head width: width={width} heads={heads}"
            ),
        }
    }
}

impl std::error::Error for PositionError {}

pub trait PositionalEncoding {
    fn kind(&self) -> &'static str;
    fn max_positions(&self) -> usize;
    fn width(&self) -> usize;
    fn add_forward(&self, activations: &mut [f32], positions: usize) -> Result<(), PositionError>;

    fn apply_qk(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        positions: usize,
        heads: usize,
    ) -> Result<(), PositionError> {
        validate_qk(q, k, positions, self.width(), heads)
    }

    fn backward_qk(
        &self,
        dq: &mut [f32],
        dk: &mut [f32],
        positions: usize,
        heads: usize,
    ) -> Result<(), PositionError> {
        validate_qk(dq, dk, positions, self.width(), heads)
    }
}

pub trait TrainablePositionalEncoding: PositionalEncoding {
    fn accumulate_backward(
        &self,
        activation_grads: &[f32],
        positions: usize,
        parameter_grads: &mut [f32],
    ) -> Result<(), PositionError>;
}

fn validate_qk(
    q: &[f32],
    k: &[f32],
    positions: usize,
    width: usize,
    heads: usize,
) -> Result<(), PositionError> {
    if positions == 0 || width == 0 || heads == 0 {
        return Err(PositionError::ZeroShape);
    }
    if width % heads != 0 {
        return Err(PositionError::HeadWidthMismatch { width, heads });
    }
    let expected = positions
        .checked_mul(width)
        .ok_or(PositionError::ActivationMismatch {
            expected: usize::MAX,
            actual: q.len(),
        })?;
    if q.len() != expected {
        return Err(PositionError::ActivationMismatch {
            expected,
            actual: q.len(),
        });
    }
    if k.len() != expected {
        return Err(PositionError::ActivationMismatch {
            expected,
            actual: k.len(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct LearnedAbsolute<'a> {
    weights: &'a [f32],
    block: usize,
    width: usize,
}

impl<'a> LearnedAbsolute<'a> {
    pub fn new(weights: &'a [f32], block: usize, width: usize) -> Result<Self, PositionError> {
        if block == 0 || width == 0 {
            return Err(PositionError::ZeroShape);
        }
        let expected = block
            .checked_mul(width)
            .ok_or(PositionError::StorageMismatch {
                expected: usize::MAX,
                actual: weights.len(),
            })?;
        if weights.len() != expected {
            return Err(PositionError::StorageMismatch {
                expected,
                actual: weights.len(),
            });
        }
        Ok(Self { weights, block, width })
    }

    fn active_len(&self, positions: usize) -> Result<usize, PositionError> {
        if positions > self.block {
            return Err(PositionError::ContextExceeded {
                positions,
                max_positions: self.block,
            });
        }
        positions
            .checked_mul(self.width)
            .ok_or(PositionError::ActivationMismatch {
                expected: usize::MAX,
                actual: 0,
            })
    }
}

impl PositionalEncoding for LearnedAbsolute<'_> {
    fn kind(&self) -> &'static str {
        "learned_absolute"
    }

    fn max_positions(&self) -> usize {
        self.block
    }

    fn width(&self) -> usize {
        self.width
    }

    fn add_forward(&self, activations: &mut [f32], positions: usize) -> Result<(), PositionError> {
        let active = self.active_len(positions)?;
        if activations.len() != active {
            return Err(PositionError::ActivationMismatch {
                expected: active,
                actual: activations.len(),
            });
        }
        for i in 0..positions {
            for j in 0..self.width {
                let index = i * self.width + j;
                activations[index] += self.weights[index];
            }
        }
        Ok(())
    }
}

impl TrainablePositionalEncoding for LearnedAbsolute<'_> {
    fn accumulate_backward(
        &self,
        activation_grads: &[f32],
        positions: usize,
        parameter_grads: &mut [f32],
    ) -> Result<(), PositionError> {
        let active = self.active_len(positions)?;
        if activation_grads.len() != active {
            return Err(PositionError::GradientMismatch {
                expected: active,
                actual: activation_grads.len(),
            });
        }
        if parameter_grads.len() != self.weights.len() {
            return Err(PositionError::GradientMismatch {
                expected: self.weights.len(),
                actual: parameter_grads.len(),
            });
        }
        for i in 0..positions {
            for j in 0..self.width {
                let index = i * self.width + j;
                parameter_grads[index] += activation_grads[index];
            }
        }
        Ok(())
    }
}


pub fn alibi_slopes(heads: usize) -> Result<Vec<f32>, PositionError> {
    if heads == 0 {
        return Err(PositionError::ZeroShape);
    }

    fn power_of_two_slopes(heads: usize) -> Vec<f32> {
        debug_assert!(heads.is_power_of_two());
        let log2_heads = heads.ilog2() as f32;
        let start = 2.0f32.powf(-2.0f32.powf(-(log2_heads - 3.0)));
        (0..heads)
            .map(|index| start.powi((index + 1) as i32))
            .collect()
    }

    if heads.is_power_of_two() {
        return Ok(power_of_two_slopes(heads));
    }

    let lower = heads.next_power_of_two() / 2;
    let mut slopes = power_of_two_slopes(lower);
    let extended = power_of_two_slopes(lower * 2);
    slopes.extend(
        extended
            .into_iter()
            .step_by(2)
            .take(heads.saturating_sub(lower)),
    );
    Ok(slopes)
}

#[derive(Clone, Copy, Debug)]
pub struct Alibi {
    block: usize,
    width: usize,
    heads: usize,
}

impl Alibi {
    pub fn new(block: usize, width: usize, heads: usize) -> Result<Self, PositionError> {
        if block == 0 || width == 0 || heads == 0 {
            return Err(PositionError::ZeroShape);
        }
        if width % heads != 0 {
            return Err(PositionError::HeadWidthMismatch { width, heads });
        }
        Ok(Self {
            block,
            width,
            heads,
        })
    }

    fn active_len(&self, positions: usize) -> Result<usize, PositionError> {
        if positions > self.block {
            return Err(PositionError::ContextExceeded {
                positions,
                max_positions: self.block,
            });
        }
        positions
            .checked_mul(self.width)
            .ok_or(PositionError::ActivationMismatch {
                expected: usize::MAX,
                actual: 0,
            })
    }

    pub fn slopes(&self) -> Vec<f32> {
        alibi_slopes(self.heads).expect("validated ALiBi head count")
    }
}

impl PositionalEncoding for Alibi {
    fn kind(&self) -> &'static str {
        "alibi"
    }

    fn max_positions(&self) -> usize {
        self.block
    }

    fn width(&self) -> usize {
        self.width
    }

    fn add_forward(&self, activations: &mut [f32], positions: usize) -> Result<(), PositionError> {
        let expected = self.active_len(positions)?;
        if activations.len() != expected {
            return Err(PositionError::ActivationMismatch {
                expected,
                actual: activations.len(),
            });
        }
        Ok(())
    }

    fn apply_qk(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        positions: usize,
        heads: usize,
    ) -> Result<(), PositionError> {
        if heads != self.heads {
            return Err(PositionError::HeadWidthMismatch {
                width: self.width,
                heads,
            });
        }
        self.active_len(positions)?;
        validate_qk(q, k, positions, self.width, heads)
    }

    fn backward_qk(
        &self,
        dq: &mut [f32],
        dk: &mut [f32],
        positions: usize,
        heads: usize,
    ) -> Result<(), PositionError> {
        self.apply_qk(dq, dk, positions, heads)
    }
}

impl TrainablePositionalEncoding for Alibi {
    fn accumulate_backward(
        &self,
        activation_grads: &[f32],
        positions: usize,
        parameter_grads: &mut [f32],
    ) -> Result<(), PositionError> {
        let expected = self.active_len(positions)?;
        if activation_grads.len() != expected {
            return Err(PositionError::GradientMismatch {
                expected,
                actual: activation_grads.len(),
            });
        }
        let reserved = self
            .block
            .checked_mul(self.width)
            .ok_or(PositionError::GradientMismatch {
                expected: usize::MAX,
                actual: parameter_grads.len(),
            })?;
        if parameter_grads.len() != reserved {
            return Err(PositionError::GradientMismatch {
                expected: reserved,
                actual: parameter_grads.len(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rotary {
    block: usize,
    width: usize,
    heads: usize,
}

impl Rotary {
    pub fn new(block: usize, width: usize, heads: usize) -> Result<Self, PositionError> {
        if block == 0 || width == 0 || heads == 0 {
            return Err(PositionError::ZeroShape);
        }
        if width % heads != 0 || (width / heads) % 2 != 0 {
            return Err(PositionError::HeadWidthMismatch { width, heads });
        }
        Ok(Self { block, width, heads })
    }

    fn active_len(&self, positions: usize) -> Result<usize, PositionError> {
        if positions > self.block {
            return Err(PositionError::ContextExceeded {
                positions,
                max_positions: self.block,
            });
        }
        positions
            .checked_mul(self.width)
            .ok_or(PositionError::ActivationMismatch {
                expected: usize::MAX,
                actual: 0,
            })
    }

    fn rotate_pair_in_place(
        &self,
        first: &mut [f32],
        second: &mut [f32],
        positions: usize,
        inverse: bool,
    ) -> Result<(), PositionError> {
        let expected = self.active_len(positions)?;
        for values in [&*first, &*second] {
            if values.len() != expected {
                return Err(PositionError::ActivationMismatch {
                    expected,
                    actual: values.len(),
                });
            }
        }

        let head_width = self.width / self.heads;
        let pairs = head_width / 2;
        for pair in 0..pairs {
            let exponent = (2 * pair) as f32 / head_width as f32;
            let denominator = 10_000.0f32.powf(exponent);
            for position in 0..positions {
                let theta = position as f32 / denominator;
                let (sin, cos) = theta.sin_cos();
                for head in 0..self.heads {
                    let base = position * self.width + head * head_width;
                    let i0 = base + 2 * pair;
                    let i1 = i0 + 1;

                    for values in [&mut *first, &mut *second] {
                        let x0 = values[i0];
                        let x1 = values[i1];
                        if inverse {
                            values[i0] = x0 * cos + x1 * sin;
                            values[i1] = -x0 * sin + x1 * cos;
                        } else {
                            values[i0] = x0 * cos - x1 * sin;
                            values[i1] = x0 * sin + x1 * cos;
                        }
                    }
                }
            }
        }
        Ok(())
    }

}

impl PositionalEncoding for Rotary {
    fn kind(&self) -> &'static str {
        "rope"
    }

    fn max_positions(&self) -> usize {
        self.block
    }

    fn width(&self) -> usize {
        self.width
    }

    fn add_forward(&self, activations: &mut [f32], positions: usize) -> Result<(), PositionError> {
        let expected = self.active_len(positions)?;
        if activations.len() != expected {
            return Err(PositionError::ActivationMismatch {
                expected,
                actual: activations.len(),
            });
        }
        Ok(())
    }

    fn apply_qk(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        positions: usize,
        heads: usize,
    ) -> Result<(), PositionError> {
        if heads != self.heads {
            return Err(PositionError::HeadWidthMismatch {
                width: self.width,
                heads,
            });
        }
        validate_qk(q, k, positions, self.width, heads)?;
        self.rotate_pair_in_place(q, k, positions, false)
    }

    fn backward_qk(
        &self,
        dq: &mut [f32],
        dk: &mut [f32],
        positions: usize,
        heads: usize,
    ) -> Result<(), PositionError> {
        if heads != self.heads {
            return Err(PositionError::HeadWidthMismatch {
                width: self.width,
                heads,
            });
        }
        validate_qk(dq, dk, positions, self.width, heads)?;
        self.rotate_pair_in_place(dq, dk, positions, true)
    }
}

impl TrainablePositionalEncoding for Rotary {
    fn accumulate_backward(
        &self,
        activation_grads: &[f32],
        positions: usize,
        parameter_grads: &mut [f32],
    ) -> Result<(), PositionError> {
        let expected = self.active_len(positions)?;
        if activation_grads.len() != expected {
            return Err(PositionError::GradientMismatch {
                expected,
                actual: activation_grads.len(),
            });
        }
        let reserved = self
            .block
            .checked_mul(self.width)
            .ok_or(PositionError::GradientMismatch {
                expected: usize::MAX,
                actual: parameter_grads.len(),
            })?;
        if parameter_grads.len() != reserved {
            return Err(PositionError::GradientMismatch {
                expected: reserved,
                actual: parameter_grads.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_absolute_forward_matches_legacy_nested_loop_exactly() {
        let block = 4;
        let width = 3;
        let weights: Vec<f32> = (0..block * width)
            .map(|i| (i as f32 - 5.0) / 7.0)
            .collect();
        let mut expected: Vec<f32> = (0..2 * width)
            .map(|i| (i as f32 + 1.0) / 11.0)
            .collect();
        let mut observed = expected.clone();

        for i in 0..2 {
            for j in 0..width {
                expected[i * width + j] += weights[i * width + j];
            }
        }

        let positional = LearnedAbsolute::new(&weights, block, width).unwrap();
        positional.add_forward(&mut observed, 2).unwrap();
        assert_eq!(observed, expected);
        assert_eq!(positional.kind(), "learned_absolute");
    }

    #[test]
    fn learned_absolute_backward_matches_legacy_gradient_exactly() {
        let block = 4;
        let width = 3;
        let weights = vec![0.0; block * width];
        let dx: Vec<f32> = (0..2 * width).map(|i| (i as f32 - 2.0) / 5.0).collect();
        let mut expected = vec![0.25f32; block * width];
        let mut observed = expected.clone();

        for i in 0..2 {
            for j in 0..width {
                expected[i * width + j] += dx[i * width + j];
            }
        }

        let positional = LearnedAbsolute::new(&weights, block, width).unwrap();
        positional.accumulate_backward(&dx, 2, &mut observed).unwrap();
        assert_eq!(observed, expected);
    }

    #[test]
    fn alibi_slopes_match_reference_schedule_for_power_and_non_power_heads() {
        assert_eq!(alibi_slopes(1).unwrap(), vec![0.00390625]);
        assert_eq!(
            alibi_slopes(4).unwrap(),
            vec![0.25, 0.0625, 0.015625, 0.00390625]
        );
        assert_eq!(
            alibi_slopes(3).unwrap(),
            vec![0.0625, 0.00390625, 0.25]
        );
        assert!(alibi_slopes(0).is_err());
    }

    #[test]
    fn alibi_is_parameter_free_and_keeps_reserved_position_gradient_inert() {
        let alibi = Alibi::new(8, 12, 3).unwrap();
        assert_eq!(alibi.kind(), "alibi");
        assert_eq!(alibi.max_positions(), 8);
        assert_eq!(alibi.width(), 12);
        assert_eq!(alibi.slopes(), alibi_slopes(3).unwrap());

        let mut activations = vec![0.25; 4 * 12];
        let before = activations.clone();
        alibi.add_forward(&mut activations, 4).unwrap();
        assert_eq!(activations, before);

        let dx = vec![0.5; 4 * 12];
        let mut reserved = vec![0.0; 8 * 12];
        alibi.accumulate_backward(&dx, 4, &mut reserved).unwrap();
        assert!(reserved.iter().all(|&x| x == 0.0));
        assert!(Alibi::new(8, 10, 3).is_err());
    }

    #[test]
    fn rope_position_zero_is_identity_and_pair_norm_is_preserved() {
        let rope = Rotary::new(8, 8, 2).unwrap();
        let mut q: Vec<f32> = (0..32).map(|i| (i as f32 - 7.0) / 9.0).collect();
        let mut k = q.clone();
        let original = q.clone();
        rope.apply_qk(&mut q, &mut k, 4, 2).unwrap();

        assert_eq!(&q[..8], &original[..8]);
        for position in 0..4 {
            for head in 0..2 {
                for pair in 0..2 {
                    let base = position * 8 + head * 4 + pair * 2;
                    let before = original[base] * original[base]
                        + original[base + 1] * original[base + 1];
                    let after = q[base] * q[base] + q[base + 1] * q[base + 1];
                    assert!((before - after).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn rope_backward_inverts_forward_rotation() {
        let rope = Rotary::new(8, 8, 2).unwrap();
        let mut q: Vec<f32> = (0..32).map(|i| (i as f32 - 3.0) / 11.0).collect();
        let mut k: Vec<f32> = (0..32).map(|i| (i as f32 + 2.0) / 13.0).collect();
        let q0 = q.clone();
        let k0 = k.clone();
        rope.apply_qk(&mut q, &mut k, 4, 2).unwrap();
        rope.backward_qk(&mut q, &mut k, 4, 2).unwrap();
        for (a, b) in q.iter().zip(q0.iter()).chain(k.iter().zip(k0.iter())) {
            assert!((*a - *b).abs() < 2e-6);
        }
    }

    #[test]
    fn rope_reserved_position_gradient_is_inert_and_odd_head_width_rejected() {
        let rope = Rotary::new(8, 8, 2).unwrap();
        let dx = vec![0.25; 4 * 8];
        let mut reserved = vec![0.0; 8 * 8];
        rope.accumulate_backward(&dx, 4, &mut reserved).unwrap();
        assert!(reserved.iter().all(|&x| x == 0.0));
        assert!(matches!(
            Rotary::new(8, 6, 2),
            Err(PositionError::HeadWidthMismatch { .. })
        ));
    }

    fn legacy_rotate(
        values: &mut [f32],
        positions: usize,
        width: usize,
        heads: usize,
        inverse: bool,
    ) {
        let head_width = width / heads;
        let pairs = head_width / 2;
        for position in 0..positions {
            for head in 0..heads {
                let base = position * width + head * head_width;
                for pair in 0..pairs {
                    let i0 = base + 2 * pair;
                    let i1 = i0 + 1;
                    let exponent = (2 * pair) as f32 / head_width as f32;
                    let theta = position as f32 / 10_000.0f32.powf(exponent);
                    let (sin, cos) = theta.sin_cos();
                    let x0 = values[i0];
                    let x1 = values[i1];
                    if inverse {
                        values[i0] = x0 * cos + x1 * sin;
                        values[i1] = -x0 * sin + x1 * cos;
                    } else {
                        values[i0] = x0 * cos - x1 * sin;
                        values[i1] = x0 * sin + x1 * cos;
                    }
                }
            }
        }
    }

    #[test]
    fn rope_trig_reuse_is_bit_exact_with_legacy_rotation() {
        let rope = Rotary::new(8, 16, 4).unwrap();
        let mut q: Vec<f32> = (0..8 * 16)
            .map(|i| (i as f32 - 37.0) / 29.0)
            .collect();
        let mut k: Vec<f32> = (0..8 * 16)
            .map(|i| (i as f32 + 11.0) / 31.0)
            .collect();
        let mut legacy_q = q.clone();
        let mut legacy_k = k.clone();

        legacy_rotate(&mut legacy_q, 8, 16, 4, false);
        legacy_rotate(&mut legacy_k, 8, 16, 4, false);
        rope.apply_qk(&mut q, &mut k, 8, 4).unwrap();
        assert_eq!(q, legacy_q);
        assert_eq!(k, legacy_k);

        legacy_rotate(&mut legacy_q, 8, 16, 4, true);
        legacy_rotate(&mut legacy_k, 8, 16, 4, true);
        rope.backward_qk(&mut q, &mut k, 8, 4).unwrap();
        assert_eq!(q, legacy_q);
        assert_eq!(k, legacy_k);
    }

    #[test]
    fn context_and_shape_fail_explicitly() {
        let weights = vec![0.0; 8];
        let positional = LearnedAbsolute::new(&weights, 4, 2).unwrap();
        let mut too_long = vec![0.0; 10];
        assert_eq!(
            positional.add_forward(&mut too_long, 5),
            Err(PositionError::ContextExceeded {
                positions: 5,
                max_positions: 4,
            })
        );
        let mut wrong = vec![0.0; 3];
        assert!(matches!(
            positional.add_forward(&mut wrong, 2),
            Err(PositionError::ActivationMismatch { .. })
        ));
    }
}
