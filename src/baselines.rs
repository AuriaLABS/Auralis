//! #304 release baselines and performance gates.

pub const BASELINES_SCHEMA_VERSION: u32 = 1;
pub const WARMUP: u32 = 2;
pub const REPEATS: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Throughput,
    Latency,
    Memory,
    Checkpoint,
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Threshold {
    pub metric: Metric,
    pub max_regression_bps: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub metric: Metric,
    pub value: u32,
    pub runner: &'static str,
    pub config: &'static str,
}

pub fn higher_is_better(metric: Metric) -> bool {
    matches!(metric, Metric::Throughput | Metric::Prefill | Metric::Decode)
}

/// GPU kernels are experimental in the supported profile; they are not a release gate.
pub fn gpu_in_gate() -> bool {
    false
}

/// Compare only same-metric/same-runner/same-config samples against a predeclared threshold.
pub fn compare(base: &Sample, cand: &Sample, th: &Threshold) -> Verdict {
    if base.metric != cand.metric
        || base.metric != th.metric
        || base.runner != cand.runner
        || base.config != cand.config
        || base.value == 0
    {
        return Verdict::Inconclusive;
    }
    let old = base.value as u64;
    let new = cand.value as u64;
    let regression_bps = if higher_is_better(base.metric) {
        if new >= old {
            0
        } else {
            (old - new) * 10_000 / old
        }
    } else if new <= old {
        0
    } else {
        (new - old) * 10_000 / old
    };
    if regression_bps > u64::from(th.max_regression_bps) {
        Verdict::Fail
    } else {
        Verdict::Pass
    }
}

/// The RC gate passes only when every comparable metric passes.
/// A win on one metric does not hide a blocking regression.
pub fn gate(verdicts: &[Verdict]) -> bool {
    !verdicts.is_empty() && verdicts.iter().all(|v| *v == Verdict::Pass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn sample(metric: Metric, value: u32, runner: &'static str) -> Sample {
        Sample {
            metric,
            value,
            runner,
            config: "tiny-cpu",
        }
    }

    #[test]
    fn win_does_not_hide_regression_and_mismatch_is_inconclusive() {
        let thr_t = Threshold {
            metric: Metric::Throughput,
            max_regression_bps: 500,
        };
        let thr_m = Threshold {
            metric: Metric::Memory,
            max_regression_bps: 500,
        };
        let t_ok = compare(
            &sample(Metric::Throughput, 1000, "ci-cpu"),
            &sample(Metric::Throughput, 1100, "ci-cpu"),
            &thr_t,
        );
        let m_bad = compare(
            &sample(Metric::Memory, 1000, "ci-cpu"),
            &sample(Metric::Memory, 1300, "ci-cpu"),
            &thr_m,
        );
        assert_eq!(t_ok, Verdict::Pass);
        assert_eq!(m_bad, Verdict::Fail);
        assert!(!gate(&[t_ok, m_bad]));
        assert!(gate(&[t_ok]));
        let mismatch = compare(
            &sample(Metric::Latency, 100, "ci-cpu"),
            &sample(Metric::Latency, 90, "other-host"),
            &Threshold {
                metric: Metric::Latency,
                max_regression_bps: 500,
            },
        );
        assert_eq!(mismatch, Verdict::Inconclusive);
        assert!(!gate(&[mismatch]));
        assert!(!gpu_in_gate());
        assert_eq!(WARMUP, 2);
        assert_eq!(REPEATS, 3);
    }

    #[test]
    fn baselines_doc_matches_contract() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = fs::read_to_string(root.join("docs/baselines.md")).expect("docs/baselines.md");
        for needle in [
            "BASELINES_SCHEMA_VERSION = 1",
            "a win on one metric does not hide a blocking regression",
            "incomparable runners are inconclusive",
            "GPU is not a release gate",
            "Does **not** cut `v1.0.0`",
        ] {
            assert!(text.contains(needle), "baselines.md missing {needle}");
        }
        let readme = fs::read_to_string(root.join("README.md")).expect("README.md");
        assert!(readme.contains("docs/baselines.md"));
        let versions = fs::read_to_string(root.join("docs/versions.md")).expect("versions.md");
        assert!(versions.contains("BASELINES_SCHEMA_VERSION = 1"));
        assert_eq!(BASELINES_SCHEMA_VERSION, 1);
    }
}
