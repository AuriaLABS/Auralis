//! Sampling policies independent from the model architecture.

use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { temperature: 0.8, top_k: None, top_p: None }
    }
}

impl SamplingConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.temperature.is_finite() || self.temperature < 0.0 { return Err("temperature must be finite and non-negative"); }
        if matches!(self.top_k, Some(0)) { return Err("top_k must be positive"); }
        if let Some(p) = self.top_p {
            if !p.is_finite() || p <= 0.0 || p > 1.0 { return Err("top_p must be in (0, 1]"); }
        }
        Ok(())
    }
}

pub fn sample_token(logits: &[f32], cfg: SamplingConfig, rng: &mut impl Rng) -> Result<usize, &'static str> {
    cfg.validate()?;
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) { return Err("logits must be non-empty and finite"); }
    if cfg.temperature <= 1e-6 { return Ok(argmax(logits)); }

    let inv_t = 1.0 / cfg.temperature.max(1e-6);
    let mut candidates: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &x)| (i, x * inv_t)).collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if let Some(k) = cfg.top_k { candidates.truncate(k.min(candidates.len())); }

    let max_logit = candidates[0].1;
    let mut weights: Vec<f32> = candidates.iter().map(|(_, logit)| (*logit - max_logit).exp()).collect();
    if let Some(p) = cfg.top_p {
        let total = weights.iter().sum::<f32>().max(1e-20);
        let mut cumulative = 0.0f32;
        let mut keep = candidates.len();
        for (i, w) in weights.iter().enumerate() {
            cumulative += *w / total;
            if cumulative >= p { keep = i + 1; break; }
        }
        keep = keep.max(1);
        candidates.truncate(keep);
        weights.truncate(keep);
    }

    let total = weights.iter().sum::<f32>();
    if !total.is_finite() || total <= 0.0 { return Ok(candidates[0].0); }
    let mut r = rng.gen::<f32>() * total;
    for ((token, _), weight) in candidates.iter().zip(&weights) {
        r -= *weight;
        if r <= 0.0 { return Ok(*token); }
    }
    Ok(candidates.last().unwrap().0)
}

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).map(|(i, _)| i).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn zero_temperature_is_greedy() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..20 {
            assert_eq!(sample_token(&[0.1, 3.0, 2.0], SamplingConfig { temperature: 0.0, top_k: None, top_p: None }, &mut rng).unwrap(), 1);
        }
    }

    #[test]
    fn top_k_one_is_always_argmax() {
        let mut rng = StdRng::seed_from_u64(2);
        let cfg = SamplingConfig { temperature: 1.0, top_k: Some(1), top_p: None };
        for _ in 0..100 { assert_eq!(sample_token(&[1.0, 4.0, 2.0, 3.0], cfg, &mut rng).unwrap(), 1); }
    }

    #[test]
    fn tiny_top_p_keeps_best_token() {
        let mut rng = StdRng::seed_from_u64(3);
        let cfg = SamplingConfig { temperature: 1.0, top_k: None, top_p: Some(0.01) };
        for _ in 0..100 { assert_eq!(sample_token(&[1.0, 4.0, 2.0], cfg, &mut rng).unwrap(), 1); }
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let cfg = SamplingConfig { temperature: 0.9, top_k: Some(3), top_p: Some(0.9) };
        let logits = [0.2, 1.1, 0.7, 0.9, -0.2];
        let mut a = StdRng::seed_from_u64(44);
        let mut b = StdRng::seed_from_u64(44);
        let sa: Vec<_> = (0..50).map(|_| sample_token(&logits, cfg, &mut a).unwrap()).collect();
        let sb: Vec<_> = (0..50).map(|_| sample_token(&logits, cfg, &mut b).unwrap()).collect();
        assert_eq!(sa, sb);
    }

    #[test]
    fn invalid_sampling_config_is_rejected() {
        let mut rng = StdRng::seed_from_u64(5);
        assert!(sample_token(&[1.0, 2.0], SamplingConfig { temperature: 1.0, top_k: Some(0), top_p: None }, &mut rng).is_err());
        assert!(sample_token(&[1.0, 2.0], SamplingConfig { temperature: 1.0, top_k: None, top_p: Some(1.1) }, &mut rng).is_err());
    }
}
