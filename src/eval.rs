//! Deterministic evaluation over a token stream.
//!
//! Foundation initially reuses `backward_into` as a correctness reference to
//! obtain cross-entropy. It does **not** update the model or optimizer. A later
//! Foundation change will replace the scratch-gradient path with a forward-only
//! loss API and must match these results within tolerance.

use crate::metrics::LossAccumulator;
use crate::model::Gpt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalMetrics {
    pub mean_loss: f32,
    pub perplexity: f32,
    pub predicted_tokens: u64,
    pub windows: usize,
}

pub fn evaluate_tokens_reference(gpt: &Gpt, tokens: &[usize]) -> Result<EvalMetrics, &'static str> {
    if tokens.len() < 2 { return Err("evaluation split needs at least two tokens"); }
    if tokens.iter().any(|&t| t >= gpt.cfg.vocab) { return Err("evaluation token is outside model vocabulary"); }

    let mut acc = LossAccumulator::default();
    let mut scratch = vec![0.0f32; gpt.collect_params().len()];
    let mut windows = 0usize;
    let mut pos = 0usize;

    while pos + 1 < tokens.len() {
        let end = (pos + gpt.cfg.block + 1).min(tokens.len());
        let x = &tokens[pos..end - 1];
        let y = &tokens[pos + 1..end];
        if x.is_empty() { break; }
        scratch.fill(0.0);
        let loss = gpt.backward_into(x, y, &mut scratch);
        acc.add(loss, x.len())?;
        windows += 1;
        pos = end - 1;
    }

    let mean_loss = acc.mean_loss().ok_or("evaluation produced no predictions")?;
    let perplexity = acc.perplexity().ok_or("evaluation produced no predictions")?;
    Ok(EvalMetrics { mean_loss, perplexity, predicted_tokens: acc.tokens(), windows })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Config;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn evaluator_covers_every_next_token_pair_exactly_once() {
        let cfg = Config { vocab: 5, n_embd: 8, n_head: 2, n_layer: 1, block: 4, n_ff: 16 };
        let mut rng = StdRng::seed_from_u64(13);
        let gpt = Gpt::new(cfg, &mut rng);
        let tokens = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0];
        let m = evaluate_tokens_reference(&gpt, &tokens).unwrap();
        assert_eq!(m.predicted_tokens, (tokens.len() - 1) as u64);
        assert_eq!(m.windows, 3);
        assert!(m.mean_loss.is_finite() && m.mean_loss > 0.0);
        assert!(m.perplexity.is_finite() && m.perplexity >= 1.0);
    }

    #[test]
    fn evaluator_is_deterministic() {
        let cfg = Config { vocab: 3, n_embd: 8, n_head: 2, n_layer: 1, block: 3, n_ff: 16 };
        let mut rng = StdRng::seed_from_u64(21);
        let gpt = Gpt::new(cfg, &mut rng);
        let tokens = [0, 1, 2, 0, 1, 2, 0];
        let a = evaluate_tokens_reference(&gpt, &tokens).unwrap();
        let b = evaluate_tokens_reference(&gpt, &tokens).unwrap();
        assert_eq!(a, b);
    }
}
