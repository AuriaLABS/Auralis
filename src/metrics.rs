//! Canonical experiment metrics shared by trainers and evaluators.

use std::time::Duration;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LossAccumulator {
    weighted_loss_sum: f64,
    tokens: u64,
}

impl LossAccumulator {
    pub fn add(&mut self, mean_loss: f32, tokens: usize) -> Result<(), &'static str> {
        if !mean_loss.is_finite() || mean_loss < 0.0 {
            return Err("loss must be finite and non-negative");
        }
        if tokens == 0 {
            return Err("metric update needs at least one token");
        }
        self.weighted_loss_sum += mean_loss as f64 * tokens as f64;
        self.tokens = self.tokens.checked_add(tokens as u64).ok_or("token counter overflow")?;
        Ok(())
    }

    pub fn tokens(&self) -> u64 { self.tokens }

    pub fn mean_loss(&self) -> Option<f32> {
        (self.tokens > 0).then(|| (self.weighted_loss_sum / self.tokens as f64) as f32)
    }

    pub fn perplexity(&self) -> Option<f32> { self.mean_loss().map(perplexity) }
}

pub fn perplexity(mean_cross_entropy: f32) -> f32 {
    if mean_cross_entropy.is_nan() || mean_cross_entropy < 0.0 { return f32::NAN; }
    mean_cross_entropy.min(f32::MAX.ln()).exp()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Throughput {
    pub tokens: u64,
    pub elapsed_seconds: f64,
    pub tokens_per_second: f64,
}

impl Throughput {
    pub fn from_duration(tokens: u64, elapsed: Duration) -> Self {
        let secs = elapsed.as_secs_f64();
        let rate = if secs > 0.0 { tokens as f64 / secs } else if tokens == 0 { 0.0 } else { f64::INFINITY };
        Self { tokens, elapsed_seconds: secs, tokens_per_second: rate }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_aggregation_is_token_weighted() {
        let mut m = LossAccumulator::default();
        m.add(2.0, 10).unwrap();
        m.add(1.0, 30).unwrap();
        assert_eq!(m.tokens(), 40);
        assert!((m.mean_loss().unwrap() - 1.25).abs() < 1e-6);
    }

    #[test]
    fn perplexity_matches_exp_loss() {
        let loss = 2.0f32;
        assert!((perplexity(loss) - loss.exp()).abs() < 1e-6);
    }

    #[test]
    fn throughput_uses_elapsed_wall_time() {
        let t = Throughput::from_duration(2_000, Duration::from_millis(500));
        assert_eq!(t.tokens, 2_000);
        assert!((t.tokens_per_second - 4_000.0).abs() < 1e-9);
    }
}
