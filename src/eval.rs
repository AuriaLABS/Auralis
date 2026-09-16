//! Deterministic forward-only evaluation over a token stream.
//!
//! Validation and test compute next-token cross-entropy through `Gpt::loss`.
//! No parameter-gradient buffer is allocated and backward propagation is never
//! executed. Window aggregation remains token-weighted and deterministic.

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
    if tokens.len() < 2 {
        return Err("evaluation split needs at least two tokens");
    }
    if tokens.iter().any(|&t| t >= gpt.cfg.vocab) {
        return Err("evaluation token is outside model vocabulary");
    }

    let mut acc = LossAccumulator::default();
    let mut windows = 0usize;
    let mut pos = 0usize;

    while pos + 1 < tokens.len() {
        let end = (pos + gpt.cfg.block + 1).min(tokens.len());
        let x = &tokens[pos..end - 1];
        let y = &tokens[pos + 1..end];
        if x.is_empty() {
            break;
        }
        let loss = gpt.loss(x, y);
        acc.add(loss, x.len())?;
        windows += 1;
        pos = end - 1;
    }

    let mean_loss = acc.mean_loss().ok_or("evaluation produced no predictions")?;
    let perplexity = acc.perplexity().ok_or("evaluation produced no predictions")?;
    Ok(EvalMetrics {
        mean_loss,
        perplexity,
        predicted_tokens: acc.tokens(),
        windows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Config;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn evaluator_covers_every_next_token_pair_exactly_once() {
        let cfg = Config {
            vocab: 5,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
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
    fn evaluator_is_deterministic_and_does_not_mutate_model() {
        let cfg = Config {
            vocab: 3,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 3,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(21);
        let gpt = Gpt::new(cfg, &mut rng);
        let tokens = [0, 1, 2, 0, 1, 2, 0];
        let before = gpt.collect_params();
        let a = evaluate_tokens_reference(&gpt, &tokens).unwrap();
        let b = evaluate_tokens_reference(&gpt, &tokens).unwrap();
        assert_eq!(a, b);
        assert_eq!(before, gpt.collect_params());
    }
}
