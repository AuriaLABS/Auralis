//! Experimental recurrent-reasoning policy.
//!
//! A recurrent step re-applies the same physical transformer block stack.
//! Parameters are shared across steps; increasing reasoning_steps increases
//! compute and activation-cache cost, not parameter count.

pub const RECURRENT_CONFIG_SCHEMA_VERSION: u32 = 1;
pub const MAX_RECURRENT_STEPS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentConfig {
    pub reasoning_steps: usize,
}

impl Default for RecurrentConfig {
    fn default() -> Self {
        Self { reasoning_steps: 1 }
    }
}

impl RecurrentConfig {
    pub fn new(reasoning_steps: usize) -> Result<Self, String> {
        let config = Self { reasoning_steps };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(self) -> Result<Self, String> {
        if self.reasoning_steps == 0 || self.reasoning_steps > MAX_RECURRENT_STEPS {
            return Err(format!(
                "reasoning_steps must be in 1..={MAX_RECURRENT_STEPS}, got {}",
                self.reasoning_steps
            ));
        }
        Ok(self)
    }

    pub fn block_applications(self, physical_layers: usize) -> Result<usize, String> {
        self.validate()?;
        if physical_layers == 0 {
            return Err("recurrent reasoning requires at least one physical layer".into());
        }
        self.reasoning_steps
            .checked_mul(physical_layers)
            .ok_or_else(|| "recurrent block application count overflow".to_string())
    }

    pub fn line(self, physical_layers: usize) -> Result<String, String> {
        Ok(format!(
            "recurrent_config | schema={} reasoning_steps={} physical_layers={} block_applications={}",
            RECURRENT_CONFIG_SCHEMA_VERSION,
            self.reasoning_steps,
            physical_layers,
            self.block_applications(physical_layers)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_is_one_reasoning_step() {
        let cfg = RecurrentConfig::default();
        assert_eq!(cfg.reasoning_steps, 1);
        assert_eq!(cfg.block_applications(3).unwrap(), 3);
    }

    #[test]
    fn recurrent_budget_is_bounded_and_parameter_independent() {
        assert!(RecurrentConfig::new(0).is_err());
        assert!(RecurrentConfig::new(MAX_RECURRENT_STEPS + 1).is_err());
        assert_eq!(
            RecurrentConfig::new(3)
                .unwrap()
                .block_applications(2)
                .unwrap(),
            6
        );
    }
}
