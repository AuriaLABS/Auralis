//! Local release-candidate checklist.
//!
//! Reports verifiable gates only. Never creates tags or records human approval.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateStatus {
    Pass,
    Pending,
    Fail,
}

impl GateStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Pending => "pending",
            Self::Fail => "fail",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gate {
    pub id: &'static str,
    pub status: GateStatus,
    pub kind: &'static str,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseReport {
    pub root: String,
    pub automated_pass: bool,
    pub human_approval: GateStatus,
    pub gates: Vec<Gate>,
}

impl ReleaseReport {
    pub fn human(&self) -> String {
        let mut out = format!(
            "release-check | root={} automated_pass={} human_approval={}\n",
            self.root,
            self.automated_pass,
            self.human_approval.as_str()
        );
        for gate in &self.gates {
            out.push_str(&format!(
                "gate | id={} status={} kind={} detail={}\n",
                gate.id,
                gate.status.as_str(),
                gate.kind,
                gate.detail
            ));
        }
        out.push_str("note | this tool never creates tags or approves a release\n");
        out
    }

    pub fn json(&self) -> String {
        let gates = self
            .gates
            .iter()
            .map(|g| {
                format!(
                    "{{\"id\":\"{}\",\"status\":\"{}\",\"kind\":\"{}\",\"detail\":\"{}\"}}",
                    g.id,
                    g.status.as_str(),
                    g.kind,
                    escape(&g.detail)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"root\":\"{}\",\"automated_pass\":{},\"human_approval\":\"{}\",\"creates_tags\":false,\"gates\":[{}]}}",
            escape(&self.root),
            self.automated_pass,
            self.human_approval.as_str(),
            gates
        )
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn exists(root: &Path, rel: &str) -> bool {
    root.join(rel).is_file()
}

fn file_gate(id: &'static str, root: &Path, rel: &str) -> Gate {
    if exists(root, rel) {
        Gate {
            id,
            status: GateStatus::Pass,
            kind: "automated",
            detail: format!("{rel} present"),
        }
    } else {
        Gate {
            id,
            status: GateStatus::Fail,
            kind: "automated",
            detail: format!("{rel} missing"),
        }
    }
}

fn pending(id: &'static str, detail: &str) -> Gate {
    Gate {
        id,
        status: GateStatus::Pending,
        kind: "pending-external",
        detail: detail.to_string(),
    }
}

pub fn check_release(root: impl AsRef<Path>) -> ReleaseReport {
    let root = root.as_ref();
    let mut gates = vec![
        file_gate("cargo_toml", root, "Cargo.toml"),
        file_gate("readme", root, "README.md"),
        file_gate("vision", root, "VISION.md"),
        file_gate("roadmap", root, "ROADMAP.md"),
        file_gate("ci_genesis", root, ".github/workflows/genesis.yml"),
        file_gate("ci_pruebas", root, ".github/workflows/pruebas.yml"),
        file_gate("docs_ci", root, "docs/ci-pruebas.md"),
    ];
    if exists(root, "docs/model-card.md") {
        gates.push(file_gate("model_card", root, "docs/model-card.md"));
    } else {
        gates.push(pending(
            "model_card",
            "docs/model-card.md not required for local automated pass yet",
        ));
    }
    gates.push(pending(
        "rc_tag",
        "git tag/RC identity is not created or validated by this tool",
    ));
    gates.push(pending(
        "live_ci",
        "live GitHub check-run status is not queried locally",
    ));
    gates.push(pending(
        "artifacts_checksums",
        "release artifact checksums are not generated here",
    ));
    gates.push(Gate {
        id: "human_approval",
        status: GateStatus::Pending,
        kind: "human",
        detail: "human review required; tool cannot approve".into(),
    });

    let automated_pass = gates
        .iter()
        .filter(|g| g.kind == "automated")
        .all(|g| g.status == GateStatus::Pass);

    ReleaseReport {
        root: root.display().to_string(),
        automated_pass,
        human_approval: GateStatus::Pending,
        gates,
    }
}

pub fn default_root() -> PathBuf {
    env_root().unwrap_or_else(|| PathBuf::from("."))
}

fn env_root() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    if cwd.join("Cargo.toml").is_file() {
        return Some(cwd);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tree(root: &Path, files: &[&str]) {
        for rel in files {
            let path = root.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, "ok\n").unwrap();
        }
    }

    #[test]
    fn complete_tree_is_automated_pass_without_human_approval() {
        let root = std::env::temp_dir().join(format!(
            "auralis-release-ok-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_tree(
            &root,
            &[
                "Cargo.toml",
                "README.md",
                "VISION.md",
                "ROADMAP.md",
                ".github/workflows/genesis.yml",
                ".github/workflows/pruebas.yml",
                "docs/ci-pruebas.md",
            ],
        );
        let report = check_release(&root);
        let _ = fs::remove_dir_all(&root);
        assert!(report.automated_pass);
        assert_eq!(report.human_approval, GateStatus::Pending);
        assert!(report.human().contains("never creates tags"));
        assert!(report.json().contains("\"creates_tags\":false"));
    }

    #[test]
    fn missing_ci_workflow_fails_automated_pass() {
        let root = std::env::temp_dir().join(format!(
            "auralis-release-fail-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_tree(&root, &["Cargo.toml", "README.md"]);
        let report = check_release(&root);
        let _ = fs::remove_dir_all(&root);
        assert!(!report.automated_pass);
        assert!(report
            .gates
            .iter()
            .any(|g| g.id == "ci_genesis" && g.status == GateStatus::Fail));
    }
}
