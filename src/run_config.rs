//! Complete reproducible run configuration.
//!
//! CLI parsing may override a few fields, but training semantics live here so
//! checkpoints, tests and future agents can reason about one explicit contract.

use crate::experiment::SplitConfig;
use crate::training::TrainConfig;
use std::fs;
use std::path::Path;
use std::str::FromStr;

pub const RUN_CONFIG_SCHEMA_VERSION: u32 = 1;

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

    /// Canonical, deterministic representation of the effective run config.
    ///
    /// Only fields that affect reproducible run semantics belong here. Unknown
    /// fields are rejected by [`RunConfig::decode`], so secrets or unrelated
    /// environment values cannot silently enter the persisted config contract.
    pub fn encode(&self) -> String {
        format!(
            concat!(
                "auralis_run_config={}\n",
                "seed={}\n",
                "batch_size={}\n",
                "gradient_accumulation_steps={}\n",
                "learning_rate={}\n",
                "grad_clip_norm={}\n",
                "train_fraction={}\n",
                "validation_fraction={}\n",
                "bpe_merges={}\n"
            ),
            RUN_CONFIG_SCHEMA_VERSION,
            self.seed,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.learning_rate,
            self.grad_clip_norm,
            self.train_fraction,
            self.validation_fraction,
            self.bpe_merges,
        )
    }

    /// Decode a strict run-config file.
    ///
    /// Blank lines and `#` comments are ignored, but unknown and duplicate keys
    /// fail closed. Future schema versions require an explicit migration path
    /// instead of being interpreted as the current contract.
    pub fn decode(text: &str) -> Result<Self, String> {
        let mut version = None;
        let mut seed = None;
        let mut batch_size = None;
        let mut gradient_accumulation_steps = None;
        let mut learning_rate = None;
        let mut grad_clip_norm = None;
        let mut train_fraction = None;
        let mut validation_fraction = None;
        let mut bpe_merges = None;

        for (line_index, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("invalid run config line {}: expected key=value", line_index + 1))?;
            let key = key.trim();
            let value = value.trim();
            if key.is_empty() || value.is_empty() {
                return Err(format!(
                    "invalid run config line {}: empty key or value",
                    line_index + 1
                ));
            }

            match key {
                "auralis_run_config" => set_once(&mut version, value, key)?,
                "seed" => set_once(&mut seed, value, key)?,
                "batch_size" => set_once(&mut batch_size, value, key)?,
                "gradient_accumulation_steps" => {
                    set_once(&mut gradient_accumulation_steps, value, key)?
                }
                "learning_rate" => set_once(&mut learning_rate, value, key)?,
                "grad_clip_norm" => set_once(&mut grad_clip_norm, value, key)?,
                "train_fraction" => set_once(&mut train_fraction, value, key)?,
                "validation_fraction" => set_once(&mut validation_fraction, value, key)?,
                "bpe_merges" => set_once(&mut bpe_merges, value, key)?,
                other => return Err(format!("unknown run config field {other}")),
            }
        }

        let version: u32 = required(version, "auralis_run_config")?;
        if version != RUN_CONFIG_SCHEMA_VERSION {
            return Err(format!(
                "run config schema version {version} is unsupported (expected {RUN_CONFIG_SCHEMA_VERSION}); explicit migration required"
            ));
        }

        let cfg = Self {
            seed: required(seed, "seed")?,
            batch_size: required(batch_size, "batch_size")?,
            gradient_accumulation_steps: required(
                gradient_accumulation_steps,
                "gradient_accumulation_steps",
            )?,
            learning_rate: required(learning_rate, "learning_rate")?,
            grad_clip_norm: required(grad_clip_norm, "grad_clip_norm")?,
            train_fraction: required(train_fraction, "train_fraction")?,
            validation_fraction: required(validation_fraction, "validation_fraction")?,
            bpe_merges: required(bpe_merges, "bpe_merges")?,
        };
        cfg.validate().map_err(|error| error.to_string())?;
        Ok(cfg)
    }

    pub fn fingerprint(&self) -> u64 {
        fingerprint_bytes(self.encode().as_bytes())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        fs::write(path, self.encode())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| format!("cannot read run config {}: {e}", path.display()))?;
        Self::decode(&text)
    }
}

fn set_once<T>(slot: &mut Option<T>, text: &str, key: &str) -> Result<(), String>
where
    T: FromStr,
{
    if slot.is_some() {
        return Err(format!("duplicate run config field {key}"));
    }
    let parsed = text
        .parse::<T>()
        .map_err(|_| format!("invalid run config value for {key}"))?;
    *slot = Some(parsed);
    Ok(())
}

fn required<T>(slot: Option<T>, key: &str) -> Result<T, String> {
    slot.ok_or_else(|| format!("run config missing {key}"))
}

fn fingerprint_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &byte in bytes {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    #[test]
    fn canonical_config_roundtrips_exactly() {
        let cfg = RunConfig::default().with_cli_overrides(7, 3, 5);
        let encoded = cfg.encode();
        let decoded = RunConfig::decode(&encoded).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(decoded.encode(), encoded);
    }

    #[test]
    fn fingerprint_is_stable_and_tracks_effective_changes() {
        let a = RunConfig::default();
        let b = RunConfig::decode(&a.encode()).unwrap();
        assert_eq!(a.fingerprint(), b.fingerprint());

        let mut changed = a;
        changed.batch_size += 1;
        assert_ne!(a.fingerprint(), changed.fingerprint());
    }

    #[test]
    fn comments_and_whitespace_do_not_change_effective_config() {
        let cfg = RunConfig::default();
        let decorated = format!("# local note\n\n{}\n", cfg.encode());
        let decoded = RunConfig::decode(&decorated).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(decoded.fingerprint(), cfg.fingerprint());
    }

    #[test]
    fn unknown_duplicate_missing_and_future_fields_fail_closed() {
        let cfg = RunConfig::default();
        let encoded = cfg.encode();

        let unknown = format!("{encoded}secret_token=never-persist-this\n");
        assert!(RunConfig::decode(&unknown)
            .unwrap_err()
            .contains("unknown run config field"));

        let duplicate = format!("{encoded}seed={}\n", cfg.seed);
        assert!(RunConfig::decode(&duplicate)
            .unwrap_err()
            .contains("duplicate run config field seed"));

        let missing = encoded
            .lines()
            .filter(|line| !line.starts_with("bpe_merges="))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(RunConfig::decode(&missing)
            .unwrap_err()
            .contains("run config missing bpe_merges"));

        let future = encoded.replacen(
            &format!("auralis_run_config={RUN_CONFIG_SCHEMA_VERSION}"),
            "auralis_run_config=999",
            1,
        );
        assert!(RunConfig::decode(&future)
            .unwrap_err()
            .contains("explicit migration required"));
    }

    #[test]
    fn save_and_load_preserve_effective_config() {
        let cfg = RunConfig::default().with_cli_overrides(123, 2, 4);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "auralis-run-config-{}-{unique}.cfg",
            std::process::id()
        ));

        cfg.save(&path).unwrap();
        let loaded = RunConfig::load(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(loaded, cfg);
        assert_eq!(loaded.fingerprint(), cfg.fingerprint());
    }
}
