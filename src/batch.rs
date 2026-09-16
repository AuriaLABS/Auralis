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
    deterministic_batch_from_stream(tokens, block, batch_size, seed, global_step, 0)
}

/// Build a deterministic minibatch beginning at an explicit sample stream.
///
/// Splitting one effective batch into several calls with consecutive stream
/// offsets yields the same examples as one larger call. Gradient accumulation
/// can therefore change memory usage without changing data selection.
pub fn deterministic_batch_from_stream(
    tokens: &[usize],
    block: usize,
    batch_size: usize,
    seed: u64,
    global_step: u64,
    stream_offset: u64,
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
        let stream = stream_offset
            .checked_add(sample as u64)
            .ok_or("batch stream overflow")?;
        let start = deterministic_index(seed, global_step, stream, n_starts);
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

/// Engine E1 path: compute exactly the same deterministic minibatch gradient
/// without materializing `SequenceBatch` or allocating a per-call scratch
/// gradient. The caller owns both reusable gradient buffers.
pub(crate) fn backward_deterministic_batch_from_stream_into(
    gpt: &Gpt,
    tokens: &[usize],
    block: usize,
    batch_size: usize,
    seed: u64,
    global_step: u64,
    stream_offset: u64,
    grads: &mut [f32],
    scratch: &mut [f32],
) -> Result<f32, &'static str> {
    if block == 0 || batch_size == 0 {
        return Err("block and batch_size must be positive");
    }
    if block > gpt.cfg.block {
        return Err("batch block exceeds model context");
    }
    if tokens.len() <= block {
        return Err("not enough tokens for one next-token sequence");
    }
    if grads.len() != scratch.len() {
        return Err("gradient scratch buffer has wrong size");
    }

    grads.fill(0.0);
    let n_starts = tokens.len() - block;
    let mut loss_sum = 0.0f32;

    for sample in 0..batch_size {
        let stream = stream_offset
            .checked_add(sample as u64)
            .ok_or("batch stream overflow")?;
        let start = deterministic_index(seed, global_step, stream, n_starts);
        let x = &tokens[start..start + block];
        let y = &tokens[start + 1..start + block + 1];

        scratch.fill(0.0);
        loss_sum += gpt.backward_into(x, y, scratch);
        for (dst, src) in grads.iter_mut().zip(scratch.iter()) {
            *dst += *src;
        }
    }

    let inv = 1.0 / batch_size as f32;
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
    fn consecutive_streams_equal_one_larger_batch() {
        let tokens: Vec<usize> = (0..96).map(|x| x % 11).collect();
        let whole = deterministic_batch(&tokens, 8, 6, 77, 4).unwrap();
        let first = deterministic_batch_from_stream(&tokens, 8, 2, 77, 4, 0).unwrap();
        let second = deterministic_batch_from_stream(&tokens, 8, 4, 77, 4, 2).unwrap();

        let mut inputs = first.inputs;
        inputs.extend(second.inputs);
        let mut targets = first.targets;
        targets.extend(second.targets);
        assert_eq!(whole, SequenceBatch { inputs, targets });
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

    #[test]
    fn streaming_batch_matches_materialized_reference_exactly() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(81);
        let gpt = Gpt::new(cfg, &mut rng);
        let tokens: Vec<usize> = (0..96).map(|i| (i * 3 + 1) % cfg.vocab).collect();
        let batch = deterministic_batch_from_stream(&tokens, cfg.block, 3, 700, 9, 4).unwrap();
        let n = gpt.collect_params().len();
        let mut reference = vec![0.0; n];
        let mut candidate = vec![0.0; n];
        let mut scratch = vec![0.0; n];

        let reference_loss = backward_batch_into(&gpt, &batch, &mut reference).unwrap();
        let candidate_loss = backward_deterministic_batch_from_stream_into(
            &gpt,
            &tokens,
            cfg.block,
            3,
            700,
            9,
            4,
            &mut candidate,
            &mut scratch,
        )
        .unwrap();

        assert_eq!(candidate_loss, reference_loss);
        assert_eq!(candidate, reference);
    }
}
