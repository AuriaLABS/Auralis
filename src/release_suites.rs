//! Suites/evals gate for local release-check.
//!
//! Required ids match the capability battery: reasoning, code, memory, agent.
//! Does not invent scores, ppl, or SOTA.

use crate::release::{Gate, GateStatus};

pub const REQUIRED_RELEASE_SUITES: &[&str] = &["reasoning", "code", "memory", "agent"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuiteRun {
    pub id: &'static str,
    pub complete: bool,
}

/// Absent snapshot → pending. The CLI does not invent suite results.
/// A supplied snapshot must mark every required suite complete.
/// A missing or incomplete required suite is fail-closed.
pub fn suites_evals_gate(runs: Option<&[SuiteRun]>) -> Gate {
    match runs {
        None => Gate {
            id: "suites_evals",
            status: GateStatus::Pending,
            kind: "pending-external",
            detail: "suites/evals not supplied; release-check does not invent scores".into(),
        },
        Some(runs) => {
            for id in REQUIRED_RELEASE_SUITES {
                match runs.iter().find(|run| run.id == *id) {
                    None => {
                        return Gate {
                            id: "suites_evals",
                            status: GateStatus::Fail,
                            kind: "automated",
                            detail: format!("missing required suite {id}"),
                        };
                    }
                    Some(run) if !run.complete => {
                        return Gate {
                            id: "suites_evals",
                            status: GateStatus::Fail,
                            kind: "automated",
                            detail: format!("incomplete required suite {id}"),
                        };
                    }
                    Some(_) => {}
                }
            }
            Gate {
                id: "suites_evals",
                status: GateStatus::Pass,
                kind: "automated",
                detail: "reasoning/code/memory/agent complete".into(),
            }
        }
    }
}

/// Parse a supplied snapshot. Does not invent scores.
/// Line form: `suite=reasoning complete=true|false`. Unknown ids fail closed.
pub fn parse_suites(text: &str) -> Result<Vec<SuiteRun>, String> {
    let mut runs: Vec<SuiteRun> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut id = None;
        let mut complete = None;
        for part in line.split_whitespace() {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("invalid suite line {}", i + 1))?;
            match key {
                "suite" => {
                    let known = REQUIRED_RELEASE_SUITES
                        .iter()
                        .find(|candidate| **candidate == value)
                        .copied();
                    id = Some(known.ok_or_else(|| format!("unknown suite {value}"))?);
                }
                "complete" => {
                    complete = Some(match value {
                        "true" => true,
                        "false" => false,
                        _ => return Err(format!("invalid complete on line {}", i + 1)),
                    });
                }
                other => return Err(format!("unknown suite field {other}")),
            }
        }
        let id = id.ok_or_else(|| format!("suite line {} missing suite", i + 1))?;
        if runs.iter().any(|run| run.id == id) {
            return Err(format!("duplicate suite {id}"));
        }
        runs.push(SuiteRun {
            id,
            complete: complete.ok_or_else(|| format!("suite {id} missing complete"))?,
        });
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete() -> Vec<SuiteRun> {
        REQUIRED_RELEASE_SUITES
            .iter()
            .map(|id| SuiteRun {
                id,
                complete: true,
            })
            .collect()
    }

    #[test]
    fn suites_evals_pass_or_fail_closed_without_inventing_scores() {
        assert_eq!(suites_evals_gate(None).status, GateStatus::Pending);
        let green = complete();
        assert_eq!(suites_evals_gate(Some(&green)).status, GateStatus::Pass);
        let missing: Vec<_> = green.iter().copied().filter(|r| r.id != "memory").collect();
        assert_eq!(suites_evals_gate(Some(&missing)).status, GateStatus::Fail);
        let mut incomplete = complete();
        incomplete[0].complete = false;
        assert_eq!(
            suites_evals_gate(Some(&incomplete)).status,
            GateStatus::Fail
        );
        let parsed = parse_suites(
            "suite=reasoning complete=true\nsuite=code complete=true\nsuite=memory complete=true\nsuite=agent complete=true\n",
        )
        .unwrap();
        assert_eq!(suites_evals_gate(Some(&parsed)).status, GateStatus::Pass);
        assert!(parse_suites("suite=sota complete=true\n").is_err());

        let root = std::env::temp_dir().join(format!(
            "auralis-release-suites-{}",
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
        let pass = crate::release::check_release_full(&root, None, None, None, None, Some(&green), None, None);
        let fail =
            crate::release::check_release_full(&root, None, None, None, None, Some(&missing), None, None);
        let _ = std::fs::remove_dir_all(&root);
        assert!(local.automated_pass);
        assert!(local
            .gates
            .iter()
            .any(|g| g.id == "suites_evals" && g.status == GateStatus::Pending));
        assert!(pass.automated_pass);
        assert!(!fail.automated_pass);

        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("does not invent scores"));
        assert!(docs.contains("reasoning/code/memory/agent"));
        assert!(docs.contains("Does **not** cut `v1.0.0`"));
    }
}
