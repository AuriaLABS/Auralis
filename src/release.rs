//! Local release-candidate checklist and artifact provenance.
//!
//! Reports verifiable gates only. Never creates tags or records human approval.

use crate::experiment::fingerprint_bytes;
use crate::manifest::build_revision;
use std::fs;
use std::path::{Path, PathBuf};

pub const RELEASE_MANIFEST_VERSION: u32 = 1;

pub const DEFAULT_RELEASE_ARTIFACTS: &[&str] = &[
    "Cargo.toml",
    "README.md",
    "VISION.md",
    "ROADMAP.md",
    ".github/workflows/genesis.yml",
    ".github/workflows/pruebas.yml",
    "docs/ci-pruebas.md",
];

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
        "use auralis release-manifest to emit/verify checksums",
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseArtifact {
    pub path: String,
    pub bytes: u64,
    pub checksum: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseManifest {
    pub version: u32,
    pub code_revision: String,
    pub rustc: String,
    pub artifacts: Vec<ReleaseArtifact>,
}

impl ReleaseManifest {
    pub fn capture(root: impl AsRef<Path>, paths: &[&str]) -> Result<Self, String> {
        let root = root.as_ref();
        let mut artifacts = Vec::new();
        for path in paths {
            artifacts.push(hash_artifact(root, path)?);
        }
        artifacts.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Self {
            version: RELEASE_MANIFEST_VERSION,
            code_revision: build_revision().to_string(),
            rustc: option_env!("RUSTC_VERSION")
                .unwrap_or(env!("CARGO_PKG_VERSION"))
                .to_string(),
            artifacts,
        })
    }

    pub fn encode(&self) -> String {
        let mut out = format!(
            "auralis_release={}\ncode_revision={}\nrustc={}\nartifact_count={}\n",
            self.version, self.code_revision, self.rustc, self.artifacts.len()
        );
        for art in &self.artifacts {
            out.push_str(&format!(
                "artifact={} bytes={} checksum={:016x}\n",
                art.path, art.bytes, art.checksum
            ));
        }
        out
    }

    pub fn decode(text: &str) -> Result<Self, String> {
        let mut version = None;
        let mut code_revision = None;
        let mut rustc = None;
        let mut artifact_count = None;
        let mut artifacts = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("invalid release manifest line {}", i + 1))?;
            match key {
                "auralis_release" => {
                    if version.is_some() {
                        return Err("duplicate auralis_release".into());
                    }
                    version = Some(
                        value
                            .parse::<u32>()
                            .map_err(|_| "invalid auralis_release".to_string())?,
                    );
                }
                "code_revision" => {
                    if code_revision.is_some() {
                        return Err("duplicate code_revision".into());
                    }
                    code_revision = Some(value.to_string());
                }
                "rustc" => {
                    if rustc.is_some() {
                        return Err("duplicate rustc".into());
                    }
                    rustc = Some(value.to_string());
                }
                "artifact_count" => {
                    if artifact_count.is_some() {
                        return Err("duplicate artifact_count".into());
                    }
                    artifact_count = Some(
                        value
                            .parse::<usize>()
                            .map_err(|_| "invalid artifact_count".to_string())?,
                    );
                }
                "artifact" => {
                    let mut parts = value.split_whitespace();
                    let path = parts
                        .next()
                        .ok_or_else(|| "artifact missing path".to_string())?
                        .to_string();
                    let bytes = parse_kv(parts.next(), "bytes")?;
                    let checksum = parse_hex(parts.next(), "checksum")?;
                    artifacts.push(ReleaseArtifact {
                        path,
                        bytes,
                        checksum,
                    });
                }
                other => return Err(format!("unknown release manifest field {other}")),
            }
        }
        let version = version.ok_or("release manifest missing auralis_release")?;
        if version != RELEASE_MANIFEST_VERSION {
            return Err(format!(
                "release manifest version {version} is unsupported (expected {RELEASE_MANIFEST_VERSION}); explicit migration required"
            ));
        }
        let expected = artifact_count.ok_or("release manifest missing artifact_count")?;
        if expected != artifacts.len() {
            return Err(format!(
                "artifact_count mismatch: declared={expected} listed={}",
                artifacts.len()
            ));
        }
        artifacts.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Self {
            version,
            code_revision: code_revision.ok_or("release manifest missing code_revision")?,
            rustc: rustc.ok_or("release manifest missing rustc")?,
            artifacts,
        })
    }

    pub fn verify(&self, root: impl AsRef<Path>) -> Result<(), String> {
        let live = Self::capture(
            root,
            &self
                .artifacts
                .iter()
                .map(|a| a.path.as_str())
                .collect::<Vec<_>>(),
        )?;
        if live.artifacts != self.artifacts {
            return Err("release artifact checksum or size mismatch".into());
        }
        Ok(())
    }
}

fn parse_kv(item: Option<&str>, key: &str) -> Result<u64, String> {
    let item = item.ok_or_else(|| format!("artifact missing {key}"))?;
    let value = item
        .strip_prefix(&format!("{key}="))
        .ok_or_else(|| format!("artifact missing {key}"))?;
    value
        .parse::<u64>()
        .map_err(|_| format!("invalid artifact {key}"))
}

fn parse_hex(item: Option<&str>, key: &str) -> Result<u64, String> {
    let item = item.ok_or_else(|| format!("artifact missing {key}"))?;
    let value = item
        .strip_prefix(&format!("{key}="))
        .ok_or_else(|| format!("artifact missing {key}"))?;
    u64::from_str_radix(value, 16).map_err(|_| format!("invalid artifact {key}"))
}

fn hash_artifact(root: &Path, rel: &str) -> Result<ReleaseArtifact, String> {
    if rel.is_empty() || rel.starts_with('/') || rel.contains("..") {
        return Err(format!("refusing unsafe artifact path {rel}"));
    }
    let path = root.join(rel);
    let bytes = fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(ReleaseArtifact {
        path: rel.replace('\\', "/"),
        bytes: bytes.len() as u64,
        checksum: fingerprint_bytes(&bytes),
    })
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

    #[test]
    fn release_manifest_roundtrip_and_mismatch() {
        let root = std::env::temp_dir().join(format!(
            "auralis-release-man-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_tree(&root, &["Cargo.toml", "README.md"]);
        let manifest = ReleaseManifest::capture(&root, &["Cargo.toml", "README.md"]).unwrap();
        let decoded = ReleaseManifest::decode(&manifest.encode()).unwrap();
        assert_eq!(decoded, manifest);
        decoded.verify(&root).unwrap();

        fs::write(root.join("README.md"), "changed\n").unwrap();
        assert!(decoded.verify(&root).unwrap_err().contains("mismatch"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn release_manifest_rejects_path_escape() {
        let root = std::env::temp_dir().join(format!(
            "auralis-release-esc-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        assert!(ReleaseManifest::capture(&root, &["../secret"]).is_err());
        let _ = fs::remove_dir_all(&root);
    }
}
