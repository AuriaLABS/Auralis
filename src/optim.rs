//! Optimizer boundary and Adam reference implementation.
//!
//! Training code depends on Optimizer, not Adam internals. AURLIS03 keeps
//! its historical binary Adam payload; AdamState is the canonical,
//! versioned optimizer-state contract for experiments, diagnostics and future
//! checkpoint migrations.

use std::str::FromStr;

pub const OPTIMIZER_CONFIG_SCHEMA_VERSION: u32 = 1;
pub const OPTIMIZER_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerId {
    Adam,
}

impl OptimizerId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Adam => "adam",
        }
    }
}

impl FromStr for OptimizerId {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "adam" => Ok(Self::Adam),
            other => Err(format!("unknown optimizer id {other}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

impl AdamConfig {
    pub fn validate(self) -> Result<Self, String> {
        if !self.beta1.is_finite() || !(0.0..1.0).contains(&self.beta1) {
            return Err("adam beta1 must be finite and in [0,1)".into());
        }
        if !self.beta2.is_finite() || !(0.0..1.0).contains(&self.beta2) {
            return Err("adam beta2 must be finite and in [0,1)".into());
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err("adam eps must be finite and positive".into());
        }
        Ok(self)
    }

    pub fn encode(self) -> String {
        format!(
            concat!(
                "auralis_optimizer_config={}\n",
                "kind=adam\n",
                "beta1_bits={:08x}\n",
                "beta2_bits={:08x}\n",
                "eps_bits={:08x}\n"
            ),
            OPTIMIZER_CONFIG_SCHEMA_VERSION,
            self.beta1.to_bits(),
            self.beta2.to_bits(),
            self.eps.to_bits(),
        )
    }

    pub fn fingerprint(self) -> u64 {
        fingerprint_bytes(self.encode().as_bytes())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OptimizerDiagnostics<'a> {
    pub first_name: &'static str,
    pub first: &'a [f32],
    pub second_name: &'static str,
    pub second: &'a [f32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptimizerStateIdentity {
    pub state_schema: u32,
    pub optimizer: OptimizerId,
    pub config_fingerprint: u64,
    pub global_step: u64,
    pub parameter_count: usize,
    pub state_fingerprint: u64,
}

impl OptimizerStateIdentity {
    pub fn line(self) -> String {
        format!(
            "optimizer_state | schema={} kind={} config_fingerprint={:016x} global_step={} parameter_count={} state_fingerprint={:016x}",
            self.state_schema,
            self.optimizer.as_str(),
            self.config_fingerprint,
            self.global_step,
            self.parameter_count,
            self.state_fingerprint,
        )
    }
}

pub trait Optimizer {
    fn id(&self) -> OptimizerId;
    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, learning_rate: f32) -> Result<(), &'static str>;
    fn global_step(&self) -> u64;
    fn parameter_count(&self) -> usize;
    fn config_fingerprint(&self) -> u64;
    fn state_fingerprint(&self) -> u64;
    fn diagnostics(&self) -> OptimizerDiagnostics<'_>;

    fn update(
        &mut self,
        parameters: &mut [f32],
        gradients: &[f32],
    ) -> Result<(), &'static str>;

    fn state_identity(&self) -> OptimizerStateIdentity {
        OptimizerStateIdentity {
            state_schema: OPTIMIZER_STATE_SCHEMA_VERSION,
            optimizer: self.id(),
            config_fingerprint: self.config_fingerprint(),
            global_step: self.global_step(),
            parameter_count: self.parameter_count(),
            state_fingerprint: self.state_fingerprint(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdamState {
    pub version: u32,
    pub config: AdamConfig,
    pub learning_rate: f32,
    pub global_step: u64,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl AdamState {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != OPTIMIZER_STATE_SCHEMA_VERSION {
            return Err(format!(
                "optimizer state version {} is unsupported (expected {})",
                self.version, OPTIMIZER_STATE_SCHEMA_VERSION
            ));
        }
        self.config.validate()?;
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("optimizer learning rate must be finite and positive".into());
        }
        if self.m.len() != self.v.len() {
            return Err(format!(
                "adam state length mismatch: m={} v={}",
                self.m.len(),
                self.v.len()
            ));
        }
        if self.global_step > i32::MAX as u64 {
            return Err("adam global_step exceeds AURLIS03/i32 compatibility range".into());
        }
        if self.m.iter().chain(&self.v).any(|x| !x.is_finite()) {
            return Err("adam state contains non-finite moment".into());
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<String, String> {
        self.validate()?;
        Ok(format!(
            concat!(
                "auralis_optimizer_state={}\n",
                "kind=adam\n",
                "config_fingerprint={:016x}\n",
                "beta1_bits={:08x}\n",
                "beta2_bits={:08x}\n",
                "eps_bits={:08x}\n",
                "learning_rate_bits={:08x}\n",
                "global_step={}\n",
                "parameter_count={}\n",
                "m_bits={}\n",
                "v_bits={}\n"
            ),
            self.version,
            self.config.fingerprint(),
            self.config.beta1.to_bits(),
            self.config.beta2.to_bits(),
            self.config.eps.to_bits(),
            self.learning_rate.to_bits(),
            self.global_step,
            self.m.len(),
            encode_f32_bits(&self.m),
            encode_f32_bits(&self.v),
        ))
    }

    pub fn decode(text: &str) -> Result<Self, String> {
        fn value<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
            let mut found = None;
            for line in text.lines() {
                if let Some(v) = line.strip_prefix(key).and_then(|v| v.strip_prefix('=')) {
                    if found.is_some() {
                        return Err(format!("duplicate optimizer state field {key}"));
                    }
                    found = Some(v);
                }
            }
            found.ok_or_else(|| format!("optimizer state missing {key}"))
        }

        let version: u32 = value(text, "auralis_optimizer_state")?
            .parse()
            .map_err(|_| "invalid optimizer state version".to_string())?;
        if version != OPTIMIZER_STATE_SCHEMA_VERSION {
            return Err(format!(
                "optimizer state version {version} is unsupported (expected {OPTIMIZER_STATE_SCHEMA_VERSION})"
            ));
        }
        let id: OptimizerId = value(text, "kind")?.parse()?;
        if id != OptimizerId::Adam {
            return Err("AdamState requires kind=adam".into());
        }

        let beta1 = parse_f32_bits(value(text, "beta1_bits")?, "beta1_bits")?;
        let beta2 = parse_f32_bits(value(text, "beta2_bits")?, "beta2_bits")?;
        let eps = parse_f32_bits(value(text, "eps_bits")?, "eps_bits")?;
        let config = AdamConfig { beta1, beta2, eps }.validate()?;
        let stored_fingerprint = u64::from_str_radix(value(text, "config_fingerprint")?, 16)
            .map_err(|_| "invalid optimizer config fingerprint".to_string())?;
        if stored_fingerprint != config.fingerprint() {
            return Err(format!(
                "optimizer config fingerprint mismatch: stored={stored_fingerprint:016x} computed={:016x}",
                config.fingerprint()
            ));
        }

        let learning_rate =
            parse_f32_bits(value(text, "learning_rate_bits")?, "learning_rate_bits")?;
        let global_step: u64 = value(text, "global_step")?
            .parse()
            .map_err(|_| "invalid optimizer global_step".to_string())?;
        let parameter_count: usize = value(text, "parameter_count")?
            .parse()
            .map_err(|_| "invalid optimizer parameter_count".to_string())?;
        let m = decode_f32_bits(value(text, "m_bits")?, "m_bits")?;
        let v = decode_f32_bits(value(text, "v_bits")?, "v_bits")?;
        if m.len() != parameter_count || v.len() != parameter_count {
            return Err(format!(
                "optimizer parameter_count mismatch: declared={parameter_count} m={} v={}",
                m.len(),
                v.len()
            ));
        }

        let state = Self {
            version,
            config,
            learning_rate,
            global_step,
            m,
            v,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn fingerprint(&self) -> Result<u64, String> {
        Ok(fingerprint_bytes(self.encode()?.as_bytes()))
    }
}

pub struct Adam {
    learning_rate: f32,
    config: AdamConfig,
    t: i32,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    pub fn new(n: usize, learning_rate: f32) -> Self {
        Self::with_config(n, learning_rate, AdamConfig::default())
            .expect("default Adam config and caller learning rate must be valid")
    }

    pub fn with_config(
        n: usize,
        learning_rate: f32,
        config: AdamConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        validate_learning_rate(learning_rate).map_err(str::to_string)?;
        Ok(Self {
            learning_rate,
            config,
            t: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
        })
    }

    pub fn from_state(learning_rate: f32, t: i32, m: Vec<f32>, v: Vec<f32>) -> Self {
        Self::try_from_legacy_state(learning_rate, t, m, v)
            .expect("Adam::from_state received invalid state")
    }

    pub fn try_from_legacy_state(
        learning_rate: f32,
        t: i32,
        m: Vec<f32>,
        v: Vec<f32>,
    ) -> Result<Self, String> {
        if t < 0 {
            return Err("adam global step must not be negative".into());
        }
        let state = AdamState {
            version: OPTIMIZER_STATE_SCHEMA_VERSION,
            config: AdamConfig::default(),
            learning_rate,
            global_step: t as u64,
            m,
            v,
        };
        Self::try_from_state(state)
    }

    pub fn try_from_state(state: AdamState) -> Result<Self, String> {
        state.validate()?;
        Ok(Self {
            learning_rate: state.learning_rate,
            config: state.config,
            t: state.global_step as i32,
            m: state.m,
            v: state.v,
        })
    }

    pub fn state(&self) -> AdamState {
        AdamState {
            version: OPTIMIZER_STATE_SCHEMA_VERSION,
            config: self.config,
            learning_rate: self.learning_rate,
            global_step: self.global_step(),
            m: self.m.clone(),
            v: self.v.clone(),
        }
    }

    pub fn export(&self) -> (f32, i32, &[f32], &[f32]) {
        (self.learning_rate, self.t, &self.m, &self.v)
    }

    pub fn step(&mut self, parameters: &mut [f32], gradients: &[f32]) {
        self.update(parameters, gradients)
            .expect("Adam::step received incompatible parameter/gradient buffers");
    }

    pub fn config(&self) -> AdamConfig {
        self.config
    }
}

impl Optimizer for Adam {
    fn id(&self) -> OptimizerId {
        OptimizerId::Adam
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) -> Result<(), &'static str> {
        validate_learning_rate(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn global_step(&self) -> u64 {
        self.t.max(0) as u64
    }

    fn parameter_count(&self) -> usize {
        self.m.len()
    }

    fn config_fingerprint(&self) -> u64 {
        self.config.fingerprint()
    }

    fn state_fingerprint(&self) -> u64 {
        let mut h = self.config_fingerprint();
        hash_u32(&mut h, self.learning_rate.to_bits());
        hash_u64(&mut h, self.global_step());
        hash_u64(&mut h, self.parameter_count() as u64);
        for values in [&self.m, &self.v] {
            for &value in values.iter() {
                hash_u32(&mut h, value.to_bits());
            }
        }
        h
    }

    fn diagnostics(&self) -> OptimizerDiagnostics<'_> {
        OptimizerDiagnostics {
            first_name: "adam_m",
            first: &self.m,
            second_name: "adam_v",
            second: &self.v,
        }
    }

    fn update(
        &mut self,
        parameters: &mut [f32],
        gradients: &[f32],
    ) -> Result<(), &'static str> {
        if parameters.len() != gradients.len()
            || parameters.len() != self.m.len()
            || self.m.len() != self.v.len()
        {
            return Err("optimizer parameter/gradient/state length mismatch");
        }
        if self.t == i32::MAX {
            return Err("optimizer global step overflow");
        }
        self.t += 1;
        let t = self.t as f32;
        let b1t = 1.0 - self.config.beta1.powf(t);
        let b2t = 1.0 - self.config.beta2.powf(t);
        for i in 0..parameters.len() {
            let gi = gradients[i];
            self.m[i] =
                self.config.beta1 * self.m[i] + (1.0 - self.config.beta1) * gi;
            self.v[i] =
                self.config.beta2 * self.v[i] + (1.0 - self.config.beta2) * gi * gi;
            let mhat = self.m[i] / b1t;
            let vhat = self.v[i] / b2t;
            parameters[i] -=
                self.learning_rate * mhat / (vhat.sqrt() + self.config.eps);
        }
        Ok(())
    }
}

fn validate_learning_rate(value: f32) -> Result<(), &'static str> {
    if !value.is_finite() || value <= 0.0 {
        return Err("optimizer learning rate must be finite and positive");
    }
    Ok(())
}

fn encode_f32_bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_f32_bits(text: &str, key: &str) -> Result<Vec<f32>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    text.split(',')
        .map(|part| {
            u32::from_str_radix(part, 16)
                .map(f32::from_bits)
                .map_err(|_| format!("invalid {key} element"))
        })
        .collect()
}

fn parse_f32_bits(text: &str, key: &str) -> Result<f32, String> {
    u32::from_str_radix(text, 16)
        .map(f32::from_bits)
        .map_err(|_| format!("invalid {key}"))
}

fn fingerprint_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &byte in bytes {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn hash_u32(h: &mut u64, value: u32) {
    for byte in value.to_le_bytes() {
        *h ^= byte as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn hash_u64(h: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *h ^= byte as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_step(
        w: &mut [f32],
        g: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        t: i32,
    ) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let tf = t as f32;
        let b1t = 1.0 - beta1.powf(tf);
        let b2t = 1.0 - beta2.powf(tf);
        for i in 0..w.len() {
            let gi = g[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * gi;
            v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
            let mhat = m[i] / b1t;
            let vhat = v[i] / b2t;
            w[i] -= lr * mhat / (vhat.sqrt() + eps);
        }
    }

    #[test]
    fn adam_boundary_is_bit_exact_with_historical_math() {
        let mut reference_w = vec![0.1, -0.2, 0.3, -0.4];
        let mut observed_w = reference_w.clone();
        let mut m = vec![0.0; reference_w.len()];
        let mut v = vec![0.0; reference_w.len()];
        let mut adam = Adam::new(reference_w.len(), 3e-3);

        for step in 1..=4 {
            let g = vec![
                0.01 * step as f32,
                -0.02 * step as f32,
                0.03 * step as f32,
                -0.04 * step as f32,
            ];
            legacy_step(&mut reference_w, &g, &mut m, &mut v, 3e-3, step);
            Optimizer::update(&mut adam, &mut observed_w, &g).unwrap();
            assert_eq!(observed_w, reference_w);
            let (_, t, observed_m, observed_v) = adam.export();
            assert_eq!(t, step);
            assert_eq!(observed_m, m);
            assert_eq!(observed_v, v);
        }
    }

    #[test]
    fn optimizer_state_roundtrip_is_bit_exact_and_versioned() {
        let mut adam = Adam::new(3, 2e-3);
        let mut w = vec![1.0, 2.0, 3.0];
        adam.update(&mut w, &[0.5, -0.25, 0.125]).unwrap();

        let state = adam.state();
        let encoded = state.encode().unwrap();
        let decoded = AdamState::decode(&encoded).unwrap();
        assert_eq!(decoded, state);

        let restored = Adam::try_from_state(decoded).unwrap();
        assert_eq!(restored.state_identity(), adam.state_identity());
        assert_eq!(restored.state().encode().unwrap(), encoded);
    }

    #[test]
    fn state_rejects_future_version_bad_fingerprint_and_length_mismatch() {
        let state = Adam::new(2, 1e-3).state();
        let encoded = state.encode().unwrap();
        assert!(AdamState::decode(&encoded.replace(
            "auralis_optimizer_state=1",
            "auralis_optimizer_state=2"
        )).is_err());
        assert!(AdamState::decode(&encoded.replace(
            &format!("config_fingerprint={:016x}", state.config.fingerprint()),
            "config_fingerprint=0000000000000000"
        )).is_err());
        assert!(AdamState::decode(&encoded.replace(
            "parameter_count=2",
            "parameter_count=3"
        )).is_err());
    }

    #[test]
    fn optimizer_boundary_rejects_bad_lr_and_buffer_mismatch() {
        let mut adam = Adam::new(2, 1e-3);
        assert!(adam.set_learning_rate(f32::NAN).is_err());
        assert!(adam.update(&mut [1.0], &[0.1]).is_err());
        assert_eq!(adam.global_step(), 0);
    }
}
