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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineStepTiming {
    pub schema_version: u32,
    pub global_step: u64,
    pub tokens: usize,
    pub total_ns: u64,
    pub backward_accum_ns: u64,
    pub grad_process_ns: u64,
    pub optimizer_ns: u64,
    pub writeback_ns: u64,
    pub data_ns: Option<u64>,
    pub checkpoint_ns: Option<u64>,
    pub rss_kib: Option<u64>,
    pub alloc_calls: Option<u64>,
    pub alloc_bytes: Option<u64>,
}

impl EngineStepTiming {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn known_phase_ns(&self) -> u64 {
        self.backward_accum_ns
            .saturating_add(self.grad_process_ns)
            .saturating_add(self.optimizer_ns)
            .saturating_add(self.writeback_ns)
    }

    pub fn unattributed_ns(&self) -> u64 {
        self.total_ns.saturating_sub(self.known_phase_ns())
    }

    fn opt_json(value: Option<u64>) -> String {
        value.map(|v| v.to_string()).unwrap_or_else(|| "null".into())
    }

    pub fn human(&self) -> String {
        format!(
            concat!(
                "engine_step_timing | schema={} global_step={} tokens={} total_ns={} ",
                "backward_accum_ns={} grad_process_ns={} optimizer_ns={} writeback_ns={} ",
                "unattributed_ns={} data_ns={:?} checkpoint_ns={:?} rss_kib={:?} ",
                "alloc_calls={:?} alloc_bytes={:?}"
            ),
            self.schema_version,
            self.global_step,
            self.tokens,
            self.total_ns,
            self.backward_accum_ns,
            self.grad_process_ns,
            self.optimizer_ns,
            self.writeback_ns,
            self.unattributed_ns(),
            self.data_ns,
            self.checkpoint_ns,
            self.rss_kib,
            self.alloc_calls,
            self.alloc_bytes,
        )
    }

    pub fn json(&self) -> String {
        format!(
            concat!(
                "{{\"schema_version\":{},\"global_step\":{},\"tokens\":{},",
                "\"total_ns\":{},\"backward_accum_ns\":{},\"grad_process_ns\":{},",
                "\"optimizer_ns\":{},\"writeback_ns\":{},\"unattributed_ns\":{},",
                "\"data_ns\":{},\"checkpoint_ns\":{},\"rss_kib\":{},",
                "\"alloc_calls\":{},\"alloc_bytes\":{}}}"
            ),
            self.schema_version,
            self.global_step,
            self.tokens,
            self.total_ns,
            self.backward_accum_ns,
            self.grad_process_ns,
            self.optimizer_ns,
            self.writeback_ns,
            self.unattributed_ns(),
            Self::opt_json(self.data_ns),
            Self::opt_json(self.checkpoint_ns),
            Self::opt_json(self.rss_kib),
            Self::opt_json(self.alloc_calls),
            Self::opt_json(self.alloc_bytes),
        )
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

    #[test]
    fn engine_step_timing_formats_absent_fields_as_null() {
        let t = EngineStepTiming {
            schema_version: EngineStepTiming::SCHEMA_VERSION,
            global_step: 7,
            tokens: 128,
            total_ns: 100,
            backward_accum_ns: 60,
            grad_process_ns: 10,
            optimizer_ns: 15,
            writeback_ns: 5,
            data_ns: None,
            checkpoint_ns: None,
            rss_kib: Some(4096),
            alloc_calls: None,
            alloc_bytes: None,
        };
        assert_eq!(t.known_phase_ns(), 90);
        assert_eq!(t.unattributed_ns(), 10);
        let json = t.json();
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("\"data_ns\":null"));
        assert!(json.contains("\"checkpoint_ns\":null"));
        assert!(json.contains("\"rss_kib\":4096"));
        assert!(t.human().contains("engine_step_timing |"));
    }
}
