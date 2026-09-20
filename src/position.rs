//! Positional-encoding boundary for Brain experiments.
//!
//! The current supported implementation is learned absolute position
//! embeddings. The model owns the parameter storage; this module owns the
//! semantics for applying it and accumulating its gradient.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositionError {
    ZeroShape,
    StorageMismatch { expected: usize, actual: usize },
    ContextExceeded { positions: usize, max_positions: usize },
    ActivationMismatch { expected: usize, actual: usize },
    GradientMismatch { expected: usize, actual: usize },
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
        }
    }
}

impl std::error::Error for PositionError {}

pub trait PositionalEncoding {
    fn kind(&self) -> &'static str;
    fn max_positions(&self) -> usize;
    fn width(&self) -> usize;
    fn add_forward(&self, activations: &mut [f32], positions: usize) -> Result<(), PositionError>;
}

pub trait TrainablePositionalEncoding: PositionalEncoding {
    fn accumulate_backward(
        &self,
        activation_grads: &[f32],
        positions: usize,
        parameter_grads: &mut [f32],
    ) -> Result<(), PositionError>;
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
