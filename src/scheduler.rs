//! Versioned learning-rate scheduler policy.
//!
//! Scheduler semantics are kept outside RunConfig v1 and Adam. Non-historical
//! policies persist as a checkpoint-adjacent sidecar so exact resume can
//! recover policy parameters from optimizer global_step.

use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub const SCHEDULER_CONFIG_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerKind {
    Constant,
    LinearWarmup,
    Cosine,
}

impl SchedulerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::LinearWarmup => "linear_warmup",
            Self::Cosine => "cosine",
        }
    }
}

impl FromStr for SchedulerKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "constant" => Ok(Self::Constant),
            "linear_warmup" => Ok(Self::LinearWarmup),
            "cosine" => Ok(Self::Cosine),
            other => Err(format!("unknown scheduler kind {other}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchedulerConfig {
    pub kind: SchedulerKind,
    pub warmup_steps: u64,
    pub decay_steps: u64,
    pub min_lr_ratio: f32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            kind: SchedulerKind::Constant,
            warmup_steps: 0,
            decay_steps: 0,
            min_lr_ratio: 1.0,
        }
    }
}

impl SchedulerConfig {
    pub fn validate(self) -> Result<Self, String> {
        if !self.min_lr_ratio.is_finite() || !(0.0..=1.0).contains(&self.min_lr_ratio) {
            return Err("min_lr_ratio must be finite and in 0..=1".into());
        }
        match self.kind {
            SchedulerKind::Constant => {
                if self.warmup_steps != 0 || self.decay_steps != 0 || self.min_lr_ratio != 1.0 {
                    return Err(
                        "constant scheduler requires warmup_steps=0 decay_steps=0 min_lr_ratio=1"
                            .into(),
                    );
                }
            }
            SchedulerKind::LinearWarmup => {
                if self.warmup_steps == 0 {
                    return Err("linear_warmup requires warmup_steps > 0".into());
                }
                if self.decay_steps != 0 || self.min_lr_ratio != 1.0 {
                    return Err(
                        "linear_warmup requires decay_steps=0 min_lr_ratio=1".into(),
                    );
                }
            }
            SchedulerKind::Cosine => {
                if self.decay_steps == 0 {
                    return Err("cosine requires decay_steps > 0".into());
                }
            }
        }
        Ok(self)
    }

    /// LR used for the optimizer update whose zero-based global_step is given.
    ///
    /// The first optimizer update has global_step=0.
    pub fn learning_rate(self, base_lr: f32, global_step: u64) -> Result<f32, String> {
        self.validate()?;
        if !base_lr.is_finite() || base_lr <= 0.0 {
            return Err("base learning rate must be finite and positive".into());
        }
        let update = global_step.saturating_add(1);
        let ratio = match self.kind {
            SchedulerKind::Constant => 1.0,
            SchedulerKind::LinearWarmup => {
                (update.min(self.warmup_steps) as f64 / self.warmup_steps as f64) as f32
            }
            SchedulerKind::Cosine => {
                if self.warmup_steps > 0 && update <= self.warmup_steps {
                    (update as f64 / self.warmup_steps as f64) as f32
                } else {
                    let after_warmup = update.saturating_sub(self.warmup_steps);
                    let progress =
                        (after_warmup.min(self.decay_steps) as f64 / self.decay_steps as f64)
                            as f32;
                    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                    self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
                }
            }
        };
        Ok(base_lr * ratio)
    }

    pub fn encode(self) -> String {
        format!(
            concat!(
                "auralis_scheduler={}\n",
                "kind={}\n",
                "warmup_steps={}\n",
                "decay_steps={}\n",
                "min_lr_ratio={}\n"
            ),
            SCHEDULER_CONFIG_SCHEMA_VERSION,
            self.kind.as_str(),
            self.warmup_steps,
            self.decay_steps,
            self.min_lr_ratio,
        )
    }

    pub fn decode(text: &str) -> Result<Self, String> {
        let mut version = None;
        let mut kind = None;
        let mut warmup_steps = None;
        let mut decay_steps = None;
        let mut min_lr_ratio = None;

        for (line_index, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line.split_once('=').ok_or_else(|| {
                format!("invalid scheduler line {}: expected key=value", line_index + 1)
            })?;
            match key.trim() {
                "auralis_scheduler" => set_once(&mut version, value.trim(), key.trim())?,
                "kind" => set_once(&mut kind, value.trim(), key.trim())?,
                "warmup_steps" => set_once(&mut warmup_steps, value.trim(), key.trim())?,
                "decay_steps" => set_once(&mut decay_steps, value.trim(), key.trim())?,
                "min_lr_ratio" => set_once(&mut min_lr_ratio, value.trim(), key.trim())?,
                other => return Err(format!("unknown scheduler field {other}")),
            }
        }

        let version: u32 = required(version, "auralis_scheduler")?;
        if version != SCHEDULER_CONFIG_SCHEMA_VERSION {
            return Err(format!(
                "scheduler schema version {version} is unsupported (expected {SCHEDULER_CONFIG_SCHEMA_VERSION})"
            ));
        }
        Self {
            kind: required(kind, "kind")?,
            warmup_steps: required(warmup_steps, "warmup_steps")?,
            decay_steps: required(decay_steps, "decay_steps")?,
            min_lr_ratio: required(min_lr_ratio, "min_lr_ratio")?,
        }
        .validate()
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| format!("cannot read scheduler config {}: {e}", path.display()))?;
        Self::decode(&text)
    }

    pub fn save(self, path: impl AsRef<Path>) -> std::io::Result<()> {
        fs::write(path, self.encode())
    }

    pub fn fingerprint(self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for byte in self.encode().bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    pub fn line(self) -> String {
        format!(
            "scheduler | schema={} kind={} warmup_steps={} decay_steps={} min_lr_ratio={} fingerprint={:016x}",
            SCHEDULER_CONFIG_SCHEMA_VERSION,
            self.kind.as_str(),
            self.warmup_steps,
            self.decay_steps,
            self.min_lr_ratio,
            self.fingerprint(),
        )
    }
}

pub fn scheduler_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    let mut os = checkpoint.as_ref().as_os_str().to_os_string();
    os.push(".scheduler");
    PathBuf::from(os)
}

fn set_once<T: FromStr>(slot: &mut Option<T>, text: &str, key: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("duplicate scheduler field {key}"));
    }
    let value = text
        .parse::<T>()
        .map_err(|_| format!("invalid scheduler value for {key}"))?;
    *slot = Some(value);
    Ok(())
}

fn required<T>(slot: Option<T>, key: &str) -> Result<T, String> {
    slot.ok_or_else(|| format!("scheduler missing {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_reproduces_base_lr_exactly() {
        let cfg = SchedulerConfig::default();
        for step in [0, 1, 7, 1_000_000] {
            assert_eq!(cfg.learning_rate(3e-3, step).unwrap().to_bits(), (3e-3f32).to_bits());
        }
    }

    #[test]
    fn linear_warmup_curve_is_exact_at_named_steps() {
        let cfg = SchedulerConfig {
            kind: SchedulerKind::LinearWarmup,
            warmup_steps: 4,
            decay_steps: 0,
            min_lr_ratio: 1.0,
        };
        let observed: Vec<f32> = (0..6)
            .map(|step| cfg.learning_rate(1.0, step).unwrap())
            .collect();
        assert_eq!(observed, vec![0.25, 0.5, 0.75, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn cosine_respects_warmup_floor_and_terminal_ratio() {
        let cfg = SchedulerConfig {
            kind: SchedulerKind::Cosine,
            warmup_steps: 2,
            decay_steps: 4,
            min_lr_ratio: 0.1,
        };
        assert_eq!(cfg.learning_rate(1.0, 0).unwrap(), 0.5);
        assert_eq!(cfg.learning_rate(1.0, 1).unwrap(), 1.0);
        assert!((cfg.learning_rate(1.0, 5).unwrap() - 0.1).abs() < 1e-6);
        assert!((cfg.learning_rate(1.0, 50).unwrap() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn roundtrip_and_future_versions_fail_closed() {
        for cfg in [
            SchedulerConfig::default(),
            SchedulerConfig {
                kind: SchedulerKind::LinearWarmup,
                warmup_steps: 10,
                decay_steps: 0,
                min_lr_ratio: 1.0,
            },
            SchedulerConfig {
                kind: SchedulerKind::Cosine,
                warmup_steps: 2,
                decay_steps: 20,
                min_lr_ratio: 0.05,
            },
        ] {
            assert_eq!(SchedulerConfig::decode(&cfg.encode()).unwrap(), cfg);
        }
        assert!(SchedulerConfig::decode(
            "auralis_scheduler=2\nkind=constant\nwarmup_steps=0\ndecay_steps=0\nmin_lr_ratio=1\n"
        ).is_err());
    }
}
