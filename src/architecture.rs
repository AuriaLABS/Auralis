//! Versioned model-architecture configuration for Brain experiments.
//!
//! This contract is intentionally separate from RunConfig v1 so adding
//! architecture experiments does not reinterpret historical training configs.

use crate::model::Config;
use std::fs;
use std::path::Path;
use std::str::FromStr;

pub const ARCHITECTURE_CONFIG_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchitectureConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block: usize,
    pub n_ff: usize,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            n_embd: 32,
            n_head: 4,
            n_layer: 2,
            block: 32,
            n_ff: 96,
        }
    }
}

impl ArchitectureConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_embd == 0 || self.n_embd > 4096 {
            return Err("n_embd must be in 1..=4096".into());
        }
        if self.n_head == 0 || self.n_head > 256 {
            return Err("n_head must be in 1..=256".into());
        }
        if self.n_layer == 0 || self.n_layer > 64 {
            return Err("n_layer must be in 1..=64".into());
        }
        if self.block == 0 || self.block > 8192 {
            return Err("block must be in 1..=8192".into());
        }
        if self.n_ff == 0 || self.n_ff > 16384 {
            return Err("n_ff must be in 1..=16384".into());
        }
        if self.n_embd % self.n_head != 0 {
            return Err("n_embd must be divisible by n_head".into());
        }
        Ok(())
    }

    pub fn model_config(&self, vocab: usize) -> Result<Config, String> {
        self.validate()?;
        if vocab <= 1 || vocab > 65_536 {
            return Err("vocab must be in 2..=65536".into());
        }
        Ok(Config {
            vocab,
            n_embd: self.n_embd,
            n_head: self.n_head,
            n_layer: self.n_layer,
            block: self.block,
            n_ff: self.n_ff,
        })
    }

    pub fn matches_model(&self, cfg: Config) -> bool {
        self.n_embd == cfg.n_embd
            && self.n_head == cfg.n_head
            && self.n_layer == cfg.n_layer
            && self.block == cfg.block
            && self.n_ff == cfg.n_ff
    }

    pub fn encode(&self) -> String {
        format!(
            concat!(
                "auralis_architecture={}\n",
                "n_embd={}\n",
                "n_head={}\n",
                "n_layer={}\n",
                "block={}\n",
                "n_ff={}\n"
            ),
            ARCHITECTURE_CONFIG_SCHEMA_VERSION,
            self.n_embd,
            self.n_head,
            self.n_layer,
            self.block,
            self.n_ff,
        )
    }

    pub fn decode(text: &str) -> Result<Self, String> {
        let mut version = None;
        let mut n_embd = None;
        let mut n_head = None;
        let mut n_layer = None;
        let mut block = None;
        let mut n_ff = None;

        for (line_index, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("invalid architecture line {}: expected key=value", line_index + 1))?;
            let key = key.trim();
            let value = value.trim();
            match key {
                "auralis_architecture" => set_once(&mut version, value, key)?,
                "n_embd" => set_once(&mut n_embd, value, key)?,
                "n_head" => set_once(&mut n_head, value, key)?,
                "n_layer" => set_once(&mut n_layer, value, key)?,
                "block" => set_once(&mut block, value, key)?,
                "n_ff" => set_once(&mut n_ff, value, key)?,
                other => return Err(format!("unknown architecture field {other}")),
            }
        }

        let version: u32 = required(version, "auralis_architecture")?;
        if version != ARCHITECTURE_CONFIG_SCHEMA_VERSION {
            return Err(format!(
                "architecture schema version {version} is unsupported (expected {ARCHITECTURE_CONFIG_SCHEMA_VERSION})"
            ));
        }
        let cfg = Self {
            n_embd: required(n_embd, "n_embd")?,
            n_head: required(n_head, "n_head")?,
            n_layer: required(n_layer, "n_layer")?,
            block: required(block, "block")?,
            n_ff: required(n_ff, "n_ff")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| format!("cannot read architecture config {}: {e}", path.display()))?;
        Self::decode(&text)
    }

    pub fn fingerprint(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for byte in self.encode().bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    pub fn line(&self) -> String {
        format!(
            "architecture | schema={} n_embd={} n_head={} n_layer={} block={} n_ff={} fingerprint={:016x}",
            ARCHITECTURE_CONFIG_SCHEMA_VERSION,
            self.n_embd,
            self.n_head,
            self.n_layer,
            self.block,
            self.n_ff,
            self.fingerprint(),
        )
    }
}

fn set_once(slot: &mut Option<String>, value: &str, key: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("duplicate architecture field {key}"));
    }
    if value.is_empty() {
        return Err(format!("empty architecture field {key}"));
    }
    *slot = Some(value.to_string());
    Ok(())
}

fn required<T: FromStr>(value: Option<String>, key: &str) -> Result<T, String> {
    value
        .ok_or_else(|| format!("architecture missing {key}"))?
        .parse()
        .map_err(|_| format!("invalid architecture value for {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_historical_tiny_architecture() {
        let a = ArchitectureConfig::default();
        let cfg = a.model_config(101).unwrap();
        assert_eq!(cfg, Config::tiny(101));
    }

    #[test]
    fn one_and_multi_layer_head_configs_validate() {
        for cfg in [
            ArchitectureConfig { n_embd: 8, n_head: 1, n_layer: 1, block: 8, n_ff: 16 },
            ArchitectureConfig { n_embd: 16, n_head: 2, n_layer: 3, block: 16, n_ff: 32 },
            ArchitectureConfig { n_embd: 32, n_head: 4, n_layer: 4, block: 32, n_ff: 96 },
        ] {
            cfg.validate().unwrap();
            assert_eq!(ArchitectureConfig::decode(&cfg.encode()).unwrap(), cfg);
        }
    }

    #[test]
    fn invalid_or_future_configs_fail_closed() {
        let bad = ArchitectureConfig { n_embd: 10, n_head: 3, ..ArchitectureConfig::default() };
        assert!(bad.validate().is_err());
        assert!(ArchitectureConfig::decode(
            "auralis_architecture=2\nn_embd=32\nn_head=4\nn_layer=2\nblock=32\nn_ff=96\n"
        ).is_err());
        assert!(ArchitectureConfig::decode(
            "auralis_architecture=1\nn_embd=32\nn_head=4\nn_layer=2\nblock=32\nn_ff=96\nunknown=1\n"
        ).is_err());
    }
}
