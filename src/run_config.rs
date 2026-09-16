//! Complete reproducible run configuration.
//!
//! CLI parsing may override a few fields, but training semantics live here so
//! checkpoints, tests and future agents can reason about one explicit contract.

use crate::experiment::SplitConfig;
use crate::training::TrainConfig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunConfig {
    pub seed: u64,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f32,
    pub grad_clip_norm: f32,
    pub train_fraction: f32,
    pub validation_fraction: f32,
    pub bpe_merges: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            seed: 0xA11CE,
            batch_size: 4,
            gradient_accumulation_steps: 1,
            learning_rate: 3e-3,
            grad_clip_norm: 1.0,
            train_fraction: 0.90,
            validation_fraction: 0.05,
            bpe_merges: 64,
        }
    }
}

impl RunConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.batch_size == 0 {
            return Err("batch_size must be positive");
        }
        if self.gradient_accumulation_steps == 0 {
            return Err("gradient_accumulation_steps must be positive");
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning_rate must be finite and positive");
        }
        if !self.grad_clip_norm.is_finite() || self.grad_clip_norm <= 0.0 {
            return Err("grad_clip_norm must be finite and positive");
        }
        if !self.train_fraction.is_finite()
            || !self.validation_fraction.is_finite()
            || self.train_fraction <= 0.0
            || self.validation_fraction <= 0.0
            || self.train_fraction + self.validation_fraction >= 1.0
        {
            return Err("invalid train/validation fractions");
        }
        self.effective_batch_size()?;
        Ok(())
    }

    pub fn effective_batch_size(&self) -> Result<usize, &'static str> {
        self.batch_size
            .checked_mul(self.gradient_accumulation_steps)
            .ok_or("effective batch size overflow")
    }

    pub fn split_config(&self) -> SplitConfig {
        SplitConfig {
            train: self.train_fraction,
            validation: self.validation_fraction,
        }
    }

    pub fn train_config(&self) -> TrainConfig {
        TrainConfig {
            seed: self.seed,
            batch_size: self.batch_size,
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            grad_clip_norm: self.grad_clip_norm,
        }
    }

    pub fn with_cli_overrides(
        mut self,
        seed: u64,
        batch_size: usize,
        gradient_accumulation_steps: usize,
    ) -> Self {
        self.seed = seed;
        self.batch_size = batch_size;
        self.gradient_accumulation_steps = gradient_accumulation_steps;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_translate_without_hidden_changes() {
        let cfg = RunConfig::default();
        cfg.validate().unwrap();
        let split = cfg.split_config();
        let train = cfg.train_config();
        assert_eq!(split.train, cfg.train_fraction);
        assert_eq!(split.validation, cfg.validation_fraction);
        assert_eq!(train.seed, cfg.seed);
        assert_eq!(train.batch_size, cfg.batch_size);
        assert_eq!(
            train.gradient_accumulation_steps,
            cfg.gradient_accumulation_steps
        );
        assert_eq!(train.grad_clip_norm, cfg.grad_clip_norm);
        assert_eq!(cfg.effective_batch_size().unwrap(), 4);
    }

    #[test]
    fn invalid_hyperparameters_fail_closed() {
        let mut cfg = RunConfig::default();
        cfg.learning_rate = 0.0;
        assert!(cfg.validate().is_err());

        let mut cfg = RunConfig::default();
        cfg.batch_size = 0;
        assert!(cfg.validate().is_err());

        let mut cfg = RunConfig::default();
        cfg.gradient_accumulation_steps = 0;
        assert!(cfg.validate().is_err());

        let mut cfg = RunConfig::default();
        cfg.train_fraction = 0.99;
        cfg.validation_fraction = 0.02;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn cli_overrides_preserve_the_rest_of_the_contract() {
        let defaults = RunConfig::default();
        let cfg = defaults.with_cli_overrides(7, 3, 5);
        assert_eq!(cfg.seed, 7);
        assert_eq!(cfg.batch_size, 3);
        assert_eq!(cfg.gradient_accumulation_steps, 5);
        assert_eq!(cfg.learning_rate, defaults.learning_rate);
        assert_eq!(cfg.bpe_merges, defaults.bpe_merges);
        assert_eq!(cfg.effective_batch_size().unwrap(), 15);
    }
}
