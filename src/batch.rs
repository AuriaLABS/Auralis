//! Reference minibatch implementation.
//!
//! Foundation prioritizes mathematically correct batch semantics and
//! reproducibility. Engine (0.3) can vectorize/parallelize this API later
//! without changing the training contract.

use crate::experiment::deterministic_index;
use crate::model::Gpt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceBatch {
    pub inputs: Vec<Vec<usize>>,
    pub targets: Vec<Vec<usize>>,
}

impl SequenceBatch {
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

pub fn deterministic_batch(
    tokens: &[usize],
    block: usize,
    batch_size: usize,
    seed: u64,
    global_step: u64,
) -> Result<SequenceBatch, &'static str> {
    if block == 0 || batch_size == 0 {
        return Err("block and batch_size must be positive");
    }
    if tokens.len() <= block {
        return Err("not enough tokens for one next-token sequence");
    }

    let n_starts = tokens.len() - block;
    let mut inputs = Vec::with_capacity(batch_size);
    let mut targets = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let start = deterministic_index(seed, global_step, sample as u64, n_starts);
        inputs.push(tokens[start..start + block].to_vec());
        targets.push(tokens[start + 1..start + block + 1].to_vec());
    }

    Ok(SequenceBatch { inputs, targets })
}

/// Compute a mean minibatch loss and mean parameter gradient.
///
/// This intentionally evaluates examples one by one and averages their
/// gradients. It establishes the exact semantics against which later SIMD/GPU
/// kernels can be tested.
pub fn backward_batch_into(
    gpt: &Gpt,
    batch: &SequenceBatch,
    grads: &mut [f32],
) -> Result<f32, &'static str> {
    if batch.is_empty() || batch.inputs.len() != batch.targets.len() {
        return Err("invalid empty or mismatched batch");
    }
    if grads.len() != gpt.collect_params().len() {
        return Err("gradient buffer has wrong size");
    }

    grads.fill(0.0);
    let mut scratch = vec![0.0f32; grads.len()];
    let mut loss_sum = 0.0f32;

    for (x, y) in batch.inputs.iter().zip(&batch.targets) {
        if x.len() != y.len() || x.is_empty() || x.len() > gpt.cfg.block {
            return Err("invalid sequence in batch");
        }
        scratch.fill(0.0);
        loss_sum += gpt.backward_into(x, y, &mut scratch);
        for (dst, src) in grads.iter_mut().zip(&scratch) {
            *dst += *src;
        }
    }

    let inv = 1.0 / batch.len() as f32;
    for g in grads {
        *g *= inv;
    }

    Ok(loss_sum * inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Config;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn batch_generation_is_exactly_reproducible() {
        let tokens: Vec<usize> = (0..64).map(|x| x % 7).collect();
        let a = deterministic_batch(&tokens, 8, 4, 123, 9).unwrap();
        let b = deterministic_batch(&tokens, 8, 4, 123, 9).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 4);
        assert!(a.inputs.iter().all(|x| x.len() == 8));
        assert!(a.targets.iter().all(|x| x.len() == 8));
    }

    #[test]
    fn batch_gradient_equals_mean_of_individual_gradients() {
        let cfg = Config {
            vocab: 5,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(55);
        let gpt = Gpt::new(cfg, &mut rng);
        let batch = SequenceBatch {
            inputs: vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4]],
            targets: vec![vec![1, 2, 3, 4], vec![2, 3, 4, 0]],
        };

        let n = gpt.collect_params().len();
        let mut batch_grad = vec![0.0; n];
        let batch_loss = backward_batch_into(&gpt, &batch, &mut batch_grad).unwrap();

        let mut g0 = vec![0.0; n];
        let mut g1 = vec![0.0; n];
        let l0 = gpt.backward_into(&batch.inputs[0], &batch.targets[0], &mut g0);
        let l1 = gpt.backward_into(&batch.inputs[1], &batch.targets[1], &mut g1);

        assert!((batch_loss - 0.5 * (l0 + l1)).abs() < 1e-6);
        for i in 0..n {
            let expected = 0.5 * (g0[i] + g1[i]);
            assert!(
                (batch_grad[i] - expected).abs() < 1e-6,
                "gradient mismatch at {i}: {} vs {expected}",
                batch_grad[i]
            );
        }
    }
}
