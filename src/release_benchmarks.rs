//! Benchmarks gate for local release-check.
//!
//! Uses `BASELINES_SCHEMA_VERSION = 1`. Does not invent numbers.
//! GPU is not a release gate.

use crate::release::{Gate, GateStatus};

pub const REQUIRED_RELEASE_BENCHES: &[&str] = &["matmul", "forward"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchRun {
    pub id: &'static str,
    pub comparable: bool,
    pub blocking_regression: bool,
}

/// Absent snapshot → pending. The CLI does not invent numbers.
/// A supplied snapshot must include each required bench, comparable, with no blocking regression.
/// Incomparable runners and a blocking regression are fail-closed.
/// A win on one metric does not hide a blocking regression. GPU is not a required bench.
pub fn benchmarks_gate(runs: Option<&[BenchRun]>) -> Gate {
    match runs {
        None => Gate {
            id: "benchmarks",
            status: GateStatus::Pending,
            kind: "pending-external",
            detail: "benchmarks not supplied; release-check does not invent numbers".into(),
        },
        Some(runs) => {
            for id in REQUIRED_RELEASE_BENCHES {
                match runs.iter().find(|run| run.id == *id) {
                    None => {
                        return Gate {
                            id: "benchmarks",
                            status: GateStatus::Fail,
                            kind: "automated",
                            detail: format!("missing required bench {id}"),
                        };
                    }
                    Some(run) if !run.comparable => {
                        return Gate {
                            id: "benchmarks",
                            status: GateStatus::Fail,
                            kind: "automated",
                            detail: format!("{id} Inconclusive"),
                        };
                    }
                    Some(run) if run.blocking_regression => {
                        return Gate {
                            id: "benchmarks",
                            status: GateStatus::Fail,
                            kind: "automated",
                            detail: format!("{id} blocking regression"),
                        };
                    }
                    Some(_) => {}
                }
            }
            Gate {
                id: "benchmarks",
                status: GateStatus::Pass,
                kind: "automated",
                detail: "BASELINES_SCHEMA_VERSION = 1 matmul/forward comparable".into(),
            }
        }
    }
}

/// Parse a supplied snapshot. Does not invent numbers. GPU is not a required bench.
/// Line form: `bench=matmul comparable=true|false blocking_regression=true|false`.
pub fn parse_benches(text: &str) -> Result<Vec<BenchRun>, String> {
    let mut runs: Vec<BenchRun> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut id = None;
        let mut comparable = None;
        let mut blocking_regression = None;
        for part in line.split_whitespace() {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("invalid bench line {}", i + 1))?;
            match key {
                "bench" => {
                    let known = REQUIRED_RELEASE_BENCHES
                        .iter()
                        .find(|candidate| **candidate == value)
                        .copied();
                    id = Some(known.ok_or_else(|| {
                        format!("unknown bench {value}; GPU is not a release gate")
                    })?);
                }
                "comparable" => comparable = Some(parse_bool(value, i + 1)?),
                "blocking_regression" => blocking_regression = Some(parse_bool(value, i + 1)?),
                other => return Err(format!("unknown bench field {other}")),
            }
        }
        let id = id.ok_or_else(|| format!("bench line {} missing bench", i + 1))?;
        if runs.iter().any(|run| run.id == id) {
            return Err(format!("duplicate bench {id}"));
        }
        runs.push(BenchRun {
            id,
            comparable: comparable.ok_or_else(|| format!("bench {id} missing comparable"))?,
            blocking_regression: blocking_regression
                .ok_or_else(|| format!("bench {id} missing blocking_regression"))?,
        });
    }
    Ok(runs)
}

fn parse_bool(value: &str, line: usize) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("invalid bool on line {line}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn green() -> Vec<BenchRun> {
        REQUIRED_RELEASE_BENCHES
            .iter()
            .map(|id| BenchRun {
                id,
                comparable: true,
                blocking_regression: false,
            })
            .collect()
    }

    #[test]
    fn benchmarks_pass_or_fail_closed_without_inventing_numbers() {
        assert_eq!(benchmarks_gate(None).status, GateStatus::Pending);
        let ok = green();
        assert_eq!(benchmarks_gate(Some(&ok)).status, GateStatus::Pass);
        let gpu_only = [BenchRun {
            id: "gpu",
            comparable: true,
            blocking_regression: false,
        }];
        assert_eq!(benchmarks_gate(Some(&gpu_only)).status, GateStatus::Fail);
        let mut inconclusive = green();
        inconclusive[0].comparable = false;
        assert_eq!(
            benchmarks_gate(Some(&inconclusive)).status,
            GateStatus::Fail
        );
        let mut regressed = green();
        regressed[1].blocking_regression = true;
        assert_eq!(benchmarks_gate(Some(&regressed)).status, GateStatus::Fail);
        let parsed = parse_benches(
            "bench=matmul comparable=true blocking_regression=false\nbench=forward comparable=true blocking_regression=false\n",
        )
        .unwrap();
        assert_eq!(benchmarks_gate(Some(&parsed)).status, GateStatus::Pass);
        assert!(parse_benches("bench=gpu comparable=true blocking_regression=false\n").is_err());

        let root = std::env::temp_dir().join(format!(
            "auralis-release-bench-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        for rel in crate::release::DEFAULT_RELEASE_ARTIFACTS {
            let path = root.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, "ok\n").unwrap();
        }
        crate::release_card::write_stub(&root);
        let local = crate::release::check_release(&root);
        let pass = crate::release::check_release_full(
            &root,
            None,
            None,
            None,
            None,
            None,
            Some(&ok),
            None,
        );
        let fail = crate::release::check_release_full(
            &root,
            None,
            None,
            None,
            None,
            None,
            Some(&gpu_only),
            None,
        );
        let _ = std::fs::remove_dir_all(&root);
        assert!(local.automated_pass);
        assert!(local
            .gates
            .iter()
            .any(|g| g.id == "benchmarks" && g.status == GateStatus::Pending));
        assert!(pass.automated_pass);
        assert!(!fail.automated_pass);

        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("does not invent numbers"));
        assert!(docs.contains("`BASELINES_SCHEMA_VERSION = 1`"));
        assert!(docs.contains("GPU is not a release gate"));
        assert!(docs.contains("Does **not** cut `v1.0.0`"));
    }
}
