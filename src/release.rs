//! Local release-candidate checklist and artifact provenance.
//!
//! Reports verifiable gates only. Never creates tags or records human approval.

use crate::ci_matrix::required_pr_jobs;
use crate::experiment::fingerprint_bytes;
use crate::manifest::build_revision;
use std::fs;
use std::path::{Path, PathBuf};

pub const RELEASE_MANIFEST_VERSION: u32 = 1;
pub const RELEASE_REPORT_SCHEMA_VERSION: u32 = 1;

pub const DEFAULT_RELEASE_ARTIFACTS: &[&str] = &[
    "Cargo.toml",
    "README.md",
    "VISION.md",
    "ROADMAP.md",
    ".github/workflows/genesis.yml",
    ".github/workflows/pruebas.yml",
    "docs/ci-pruebas.md",
    "LICENSE",
    "docs/api-policy.md",
    "docs/packaging.md",
    "docs/baselines.md",
    "docs/post-tag.md",
];

fn artifact_gate_id(rel: &str) -> &'static str {
    match rel {
        "Cargo.toml" => "cargo_toml",
        "README.md" => "readme",
        "VISION.md" => "vision",
        "ROADMAP.md" => "roadmap",
        ".github/workflows/genesis.yml" => "ci_genesis",
        ".github/workflows/pruebas.yml" => "ci_pruebas",
        "docs/ci-pruebas.md" => "docs_ci",
        "LICENSE" => "license",
        "docs/api-policy.md" => "api_policy",
        "docs/packaging.md" => "packaging",
        "docs/baselines.md" => "baselines",
        "docs/post-tag.md" => "post_tag",
        _ => "artifact",
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TagCandidate {
    pub name: &'static str,
    pub tag_sha: &'static str,
    pub source_sha: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CiRun {
    pub id: &'static str,
    pub conclusion: &'static str,
}

/// This checker never creates a git tag. Tagging remains a human #84 action.
pub fn create_tag() -> Result<(), &'static str> {
    Err("tag creation forbidden")
}

/// Absent candidate → pending. Present `v1.0.0` must share tag/source SHA.
/// A mismatch is fail-closed. The function never writes a tag.
pub fn rc_tag_gate(candidate: Option<TagCandidate>) -> Gate {
    match candidate {
        None => pending(
            "rc_tag",
            "no tag candidate supplied; tool never creates `v1.0.0`",
        ),
        Some(tag) if tag.name != "v1.0.0" => Gate {
            id: "rc_tag",
            status: GateStatus::Fail,
            kind: "automated",
            detail: format!("unsupported tag name {}", tag.name),
        },
        Some(tag) if tag.tag_sha.is_empty() || tag.source_sha.is_empty() => Gate {
            id: "rc_tag",
            status: GateStatus::Fail,
            kind: "automated",
            detail: "empty tag or source SHA".into(),
        },
        Some(tag) if tag.tag_sha != tag.source_sha => Gate {
            id: "rc_tag",
            status: GateStatus::Fail,
            kind: "automated",
            detail: format!("tag/source SHA mismatch {} != {}", tag.tag_sha, tag.source_sha),
        },
        Some(tag) => Gate {
            id: "rc_tag",
            status: GateStatus::Pass,
            kind: "automated",
            detail: format!("`v1.0.0` converges to {}", tag.tag_sha),
        },
    }
}

/// Absent snapshot → pending. The local checker never queries GitHub.
/// A supplied snapshot must include `required_pr_jobs()` at `success`.
/// Missing required job or non-success conclusion is fail-closed.
/// `benches` does not block.
pub fn live_ci_gate(runs: Option<&[CiRun]>) -> Gate {
    match runs {
        None => pending(
            "live_ci",
            "live GitHub check-run status is not queried locally",
        ),
        Some(runs) => {
            for job in required_pr_jobs() {
                match runs.iter().find(|r| r.id == job.id) {
                    None => {
                        return Gate {
                            id: "live_ci",
                            status: GateStatus::Fail,
                            kind: "automated",
                            detail: format!("missing required job {}", job.id),
                        };
                    }
                    Some(run) if run.conclusion != "success" => {
                        return Gate {
                            id: "live_ci",
                            status: GateStatus::Fail,
                            kind: "automated",
                            detail: format!("{} conclusion {}", job.id, run.conclusion),
                        };
                    }
                    Some(_) => {}
                }
            }
            Gate {
                id: "live_ci",
                status: GateStatus::Pass,
                kind: "automated",
                detail: "required_pr_jobs all success".into(),
            }
        }
    }
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
            "{{\"schema\":{},\"root\":\"{}\",\"automated_pass\":{},\"human_approval\":\"{}\",\"creates_tags\":false,\"gates\":[{}]}}",
            RELEASE_REPORT_SCHEMA_VERSION,
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
    check_release_with(root, None)
}

pub fn check_release_with(
    root: impl AsRef<Path>,
    tag: Option<TagCandidate>,
) -> ReleaseReport {
    check_release_with_ci(root, tag, None)
}

pub fn check_release_with_ci(
    root: impl AsRef<Path>,
    tag: Option<TagCandidate>,
    ci: Option<&[CiRun]>,
) -> ReleaseReport {
    check_release_full(root, tag, ci, None, None, None, None, None)
}

pub fn check_release_full(
    root: impl AsRef<Path>,
    tag: Option<TagCandidate>,
    ci: Option<&[CiRun]>,
    stored_manifest: Option<&str>,
    compat: Option<crate::release_compat::CompatFixture>,
    suites: Option<&[crate::release_suites::SuiteRun]>,
    benches: Option<&[crate::release_benchmarks::BenchRun]>,
    blockers: Option<&[crate::release_blockers::DeclaredBlocker]>,
) -> ReleaseReport {
    let root = root.as_ref();
    let mut gates: Vec<Gate> = DEFAULT_RELEASE_ARTIFACTS
        .iter()
        .map(|rel| file_gate(artifact_gate_id(rel), root, rel))
        .collect();
    gates.push(crate::release_card::model_card_gate(root));
    gates.push(rc_tag_gate(tag));
    gates.push(live_ci_gate(ci));
    gates.push(crate::release_checksums::artifacts_checksums_gate(
        root,
        stored_manifest,
    ));
    gates.push(crate::release_compat::compatibility_fixtures_gate(compat));
    gates.push(crate::release_suites::suites_evals_gate(suites));
    gates.push(crate::release_benchmarks::benchmarks_gate(benches));
    gates.push(crate::release_blockers::declared_blockers_gate(blockers));
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
        crate::release_card::write_stub(root);
    }

    const COMPLETE: &[&str] = DEFAULT_RELEASE_ARTIFACTS;

    fn green_ci() -> Vec<CiRun> {
        required_pr_jobs()
            .into_iter()
            .map(|j| CiRun {
                id: j.id,
                conclusion: "success",
            })
            .collect()
    }

    #[test]
    fn complete_tree_is_automated_pass_without_human_approval() {
        let root = std::env::temp_dir().join(format!(
            "auralis-release-ok-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_tree(&root, COMPLETE);
        let report = check_release(&root);
        let _ = fs::remove_dir_all(&root);
        assert!(report.automated_pass);
        assert_eq!(report.human_approval, GateStatus::Pending);
        assert!(report.human().contains("never creates tags"));
        assert!(report.json().contains("\"creates_tags\":false"));
        assert!(report.json().contains("\"schema\":1"));
        assert!(report.json().contains("\"human_approval\":\"pending\""));
        assert_eq!(RELEASE_REPORT_SCHEMA_VERSION, 1);
        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("`RELEASE_REPORT_SCHEMA_VERSION = 1`"));
        assert!(docs.contains("la herramienta no aprueba ni crea el tag"));
        assert!(report
            .gates
            .iter()
            .any(|g| g.id == "post_tag" && g.status == GateStatus::Pass));
        assert!(report
            .gates
            .iter()
            .any(|g| g.id == "rc_tag" && g.status == GateStatus::Pending));
        assert!(report
            .gates
            .iter()
            .any(|g| g.id == "live_ci" && g.status == GateStatus::Pending));
        assert!(report
            .gates
            .iter()
            .any(|g| g.id == "declared_blockers" && g.status == GateStatus::Pending));
        assert_eq!(create_tag(), Err("tag creation forbidden"));
    }

    #[test]
    fn rc_tag_converges_or_fails_closed_and_never_creates() {
        let aligned = TagCandidate {
            name: "v1.0.0",
            tag_sha: "abc",
            source_sha: "abc",
        };
        let drifted = TagCandidate {
            name: "v1.0.0",
            tag_sha: "abc",
            source_sha: "def",
        };
        assert_eq!(rc_tag_gate(Some(aligned)).status, GateStatus::Pass);
        assert_eq!(rc_tag_gate(Some(drifted)).status, GateStatus::Fail);
        assert_eq!(rc_tag_gate(None).status, GateStatus::Pending);
        assert_eq!(
            rc_tag_gate(Some(TagCandidate {
                name: "v0.9.0",
                tag_sha: "abc",
                source_sha: "abc",
            }))
            .status,
            GateStatus::Fail
        );
        assert_eq!(create_tag(), Err("tag creation forbidden"));

        let root = std::env::temp_dir().join(format!(
            "auralis-release-tag-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_tree(&root, COMPLETE);
        let pass = check_release_with(&root, Some(aligned));
        let fail = check_release_with(&root, Some(drifted));
        let _ = fs::remove_dir_all(&root);
        assert!(pass.automated_pass);
        assert!(!fail.automated_pass);
    }

    #[test]
    fn live_ci_required_jobs_pass_or_fail_closed_without_network() {
        let green = green_ci();
        assert_eq!(live_ci_gate(None).status, GateStatus::Pending);
        assert_eq!(live_ci_gate(Some(&green)).status, GateStatus::Pass);
        assert!(live_ci_gate(None)
            .detail
            .contains("live GitHub check-run status is not queried locally"));

        let mut missing = green.clone();
        missing.retain(|r| r.id != "unit");
        assert_eq!(live_ci_gate(Some(&missing)).status, GateStatus::Fail);

        let mut red = green.clone();
        for run in &mut red {
            if run.id == "core" {
                run.conclusion = "failure";
            }
        }
        assert_eq!(live_ci_gate(Some(&red)).status, GateStatus::Fail);

        let with_skipped_benches = {
            let mut runs = green.clone();
            runs.push(CiRun {
                id: "benches",
                conclusion: "skipped",
            });
            runs
        };
        assert_eq!(
            live_ci_gate(Some(&with_skipped_benches)).status,
            GateStatus::Pass
        );

        let root = std::env::temp_dir().join(format!(
            "auralis-release-ci-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_tree(&root, COMPLETE);
        let pass = check_release_with_ci(&root, None, Some(&green));
        let fail = check_release_with_ci(&root, None, Some(&red));
        let local = check_release(&root);
        let _ = fs::remove_dir_all(&root);
        assert!(pass.automated_pass);
        assert!(!fail.automated_pass);
        assert!(local.automated_pass);
        assert!(local
            .gates
            .iter()
            .any(|g| g.id == "live_ci" && g.status == GateStatus::Pending));

        let docs = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/docs/rc-gate.md"
        ));
        assert!(docs.contains("live GitHub check-run status is not queried locally"));
        assert!(docs.contains("`required_pr_jobs`"));
        assert!(docs.contains("Does **not** cut `v1.0.0`"));
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
    fn missing_post_tag_contract_fails_automated_pass() {
        let root = std::env::temp_dir().join(format!(
            "auralis-release-post-tag-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let partial: Vec<&str> = COMPLETE
            .iter()
            .copied()
            .filter(|p| *p != "docs/post-tag.md")
            .collect();
        write_tree(&root, &partial);
        let report = check_release(&root);
        let _ = fs::remove_dir_all(&root);
        assert!(!report.automated_pass);
        assert!(report
            .gates
            .iter()
            .any(|g| g.id == "post_tag" && g.status == GateStatus::Fail));
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
