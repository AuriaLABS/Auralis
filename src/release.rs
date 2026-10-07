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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TagCandidate {
    pub name: String,
    pub tag_sha: String,
    pub source_sha: String,
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

/// This checker never approves a release. Approval remains a human #84 action.
pub fn approve() -> Result<(), &'static str> {
    Err("approval forbidden")
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

/// Parse a supplied tag candidate. Does not create a tag.
/// Line form: `name=v1.0.0 tag_sha=<40 hex> source_sha=<40 hex>`.
pub fn parse_tag(text: &str) -> Result<TagCandidate, String> {
    let mut name = None;
    let mut tag_sha = None;
    let mut source_sha = None;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        for part in line.split_whitespace() {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("invalid tag line {}", i + 1))?;
            match key {
                "name" => name = Some(value.to_string()),
                "tag_sha" => tag_sha = Some(require_sha(value)?),
                "source_sha" => source_sha = Some(require_sha(value)?),
                other => return Err(format!("unknown tag field {other}")),
            }
        }
    }
    Ok(TagCandidate {
        name: name.ok_or("tag candidate missing name")?,
        tag_sha: tag_sha.ok_or("tag candidate missing tag_sha")?,
        source_sha: source_sha.ok_or("tag candidate missing source_sha")?,
    })
}

fn require_sha(value: &str) -> Result<String, String> {
    if value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(value.to_string())
    } else {
        Err("tag SHA must be 40 hex; tool never creates v1.0.0".into())
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

/// Parse a supplied CI snapshot. Does not query GitHub.
/// Line form: `job=unit conclusion=success`. Unknown jobs fail closed.
/// `benches` is not a required release job.
pub fn parse_ci(text: &str) -> Result<Vec<CiRun>, String> {
    let allowed: Vec<&str> = required_pr_jobs().iter().map(|job| job.id).collect();
    let mut runs: Vec<CiRun> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut id = None;
        let mut conclusion = None;
        for part in line.split_whitespace() {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("invalid ci line {}", i + 1))?;
            match key {
                "job" => {
                    if value == "benches" {
                        return Err("benches does not block and is not a release CI job".into());
                    }
                    id = Some(
                        allowed
                            .iter()
                            .find(|candidate| **candidate == value)
                            .copied()
                            .ok_or_else(|| format!("unknown ci job {value}"))?,
                    );
                }
                "conclusion" => {
                    conclusion = Some(match value {
                        "success" => "success",
                        "failure" => "failure",
                        "skipped" => "skipped",
                        "cancelled" => "cancelled",
                        _ => return Err(format!("unknown conclusion {value}")),
                    });
                }
                other => return Err(format!("unknown ci field {other}")),
            }
        }
        let id = id.ok_or_else(|| format!("ci line {} missing job", i + 1))?;
        if runs.iter().any(|run| run.id == id) {
            return Err(format!("duplicate ci job {id}"));
        }
        runs.push(CiRun {
            id,
            conclusion: conclusion.ok_or_else(|| format!("ci job {id} missing conclusion"))?,
        });
    }
    Ok(runs)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseReport {
    pub root: String,
    pub automated_pass: bool,
    pub human_approval: GateStatus,
    pub code_revision: String,
    pub rustc: String,
    pub target: String,
    pub features: String,
    pub cargo_lock: String,
    pub limitations: String,
    pub bench_run_ids: String,
    pub eval_run_ids: String,
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
        out.push_str(&format!(
            "provenance | code_revision={} rustc={} target={} features={} cargo_lock={} limitations={} bench_run_ids={} eval_run_ids={}\n",
            self.code_revision, self.rustc, self.target, self.features, self.cargo_lock, self.limitations, self.bench_run_ids, self.eval_run_ids
        ));
        for gate in &self.gates {
            out.push_str(&format!(
                "gate | id={} status={} kind={} detail={}\n",
                gate.id,
                gate.status.as_str(),
                gate.kind,
                gate.detail
            ));
        }
        let pending_ids = self
            .gates
            .iter()
            .filter(|g| g.status == GateStatus::Pending)
            .map(|g| g.id)
            .collect::<Vec<_>>()
            .join(",");
        out.push_str(&format!("pending | ids={pending_ids}\n"));
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
        let pass_count = self
            .gates
            .iter()
            .filter(|g| g.status == GateStatus::Pass)
            .count();
        let fail_count = self
            .gates
            .iter()
            .filter(|g| g.status == GateStatus::Fail)
            .count();
        let pending_count = self
            .gates
            .iter()
            .filter(|g| g.status == GateStatus::Pending)
            .count();
        let pending_ids = self
            .gates
            .iter()
            .filter(|g| g.status == GateStatus::Pending)
            .map(|g| format!("\"{}\"", escape(g.id)))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":{},\"root\":\"{}\",\"automated_pass\":{},\"human_approval\":\"{}\",\"creates_tags\":false,\"pass_count\":{},\"fail_count\":{},\"pending_count\":{},\"pending_ids\":[{}],\"code_revision\":\"{}\",\"rustc\":\"{}\",\"target\":\"{}\",\"features\":\"{}\",\"cargo_lock\":\"{}\",\"limitations\":\"{}\",\"bench_run_ids\":\"{}\",\"eval_run_ids\":\"{}\",\"gates\":[{}]}}",
            RELEASE_REPORT_SCHEMA_VERSION,
            escape(&self.root),
            self.automated_pass,
            self.human_approval.as_str(),
            pass_count,
            fail_count,
            pending_count,
            pending_ids,
            escape(&self.code_revision),
            escape(&self.rustc),
            escape(&self.target),
            escape(&self.features),
            escape(&self.cargo_lock),
            escape(&self.limitations),
            escape(&self.bench_run_ids),
            escape(&self.eval_run_ids),
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
    let provenance = ReleaseManifest::capture(root, DEFAULT_RELEASE_ARTIFACTS).ok();

    ReleaseReport {
        root: root.display().to_string(),
        automated_pass,
        human_approval: GateStatus::Pending,
        code_revision: provenance
            .as_ref()
            .map(|m| m.code_revision.clone())
            .unwrap_or_else(|| "unrecorded".to_string()),
        rustc: provenance
            .as_ref()
            .map(|m| m.rustc.clone())
            .unwrap_or_else(|| "unrecorded".to_string()),
        target: provenance
            .as_ref()
            .map(|m| m.target.clone())
            .unwrap_or_else(|| "unrecorded".to_string()),
        features: provenance
            .as_ref()
            .map(|m| m.features.clone())
            .unwrap_or_else(|| "unrecorded".to_string()),
        cargo_lock: provenance
            .as_ref()
            .map(|m| m.cargo_lock.clone())
            .unwrap_or_else(|| "unrecorded".to_string()),
        bench_run_ids: provenance
            .as_ref()
            .map(|m| m.bench_run_ids.clone())
            .unwrap_or_default(),
        eval_run_ids: provenance
            .as_ref()
            .map(|m| m.eval_run_ids.clone())
            .unwrap_or_default(),
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
    pub target: String,
    pub features: String,
    pub limitations: String,
    pub bench_run_ids: String,
    pub eval_run_ids: String,
    pub checkpoint_id: String,
    pub model_id: String,
    pub cargo_lock: String,
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
            rustc: rustc_field(),
            target: option_env!("TARGET").unwrap_or("unrecorded").to_string(),
            features: "unrecorded".to_string(),
            limitations: "docs/model-card.md".to_string(),
            bench_run_ids: String::new(),
            eval_run_ids: String::new(),
            checkpoint_id: String::new(),
            model_id: String::new(),
            cargo_lock: cargo_lock_field(root),
            artifacts,
        })
    }

    pub fn encode(&self) -> String {
        let mut out = format!(
            "auralis_release={}\ncode_revision={}\nrustc={}\ntarget={}\nfeatures={}\nlimitations={}\nbench_run_ids={}\neval_run_ids={}\ncheckpoint_id={}\nmodel_id={}\ncargo_lock={}\nartifact_count={}\n",
            self.version,
            self.code_revision,
            self.rustc,
            self.target,
            self.features,
            self.limitations,
            self.bench_run_ids,
            self.eval_run_ids,
            self.checkpoint_id,
            self.model_id,
            self.cargo_lock,
            self.artifacts.len()
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
        let mut target = None;
        let mut features = None;
        let mut limitations = None;
        let mut bench_run_ids = None;
        let mut eval_run_ids = None;
        let mut checkpoint_id = None;
        let mut model_id = None;
        let mut cargo_lock = None;
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
                "target" => {
                    if target.is_some() {
                        return Err("duplicate target".into());
                    }
                    target = Some(value.to_string());
                }
                "features" => {
                    if features.is_some() {
                        return Err("duplicate features".into());
                    }
                    features = Some(value.to_string());
                }
                "limitations" => {
                    if limitations.is_some() {
                        return Err("duplicate limitations".into());
                    }
                    limitations = Some(value.to_string());
                }
                "bench_run_ids" => {
                    if bench_run_ids.is_some() {
                        return Err("duplicate bench_run_ids".into());
                    }
                    bench_run_ids = Some(value.to_string());
                }
                "eval_run_ids" => {
                    if eval_run_ids.is_some() {
                        return Err("duplicate eval_run_ids".into());
                    }
                    eval_run_ids = Some(value.to_string());
                }
                "checkpoint_id" => {
                    if checkpoint_id.is_some() {
                        return Err("duplicate checkpoint_id".into());
                    }
                    checkpoint_id = Some(value.to_string());
                }
                "model_id" => {
                    if model_id.is_some() {
                        return Err("duplicate model_id".into());
                    }
                    model_id = Some(value.to_string());
                }
                "cargo_lock" => {
                    if cargo_lock.is_some() {
                        return Err("duplicate cargo_lock".into());
                    }
                    cargo_lock = Some(value.to_string());
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
            target: target.unwrap_or_else(|| "unrecorded".to_string()),
            features: features.unwrap_or_else(|| "unrecorded".to_string()),
            limitations: limitations.unwrap_or_default(),
            bench_run_ids: bench_run_ids.unwrap_or_default(),
            eval_run_ids: eval_run_ids.unwrap_or_default(),
            checkpoint_id: checkpoint_id.unwrap_or_default(),
            model_id: model_id.unwrap_or_default(),
            cargo_lock: cargo_lock.unwrap_or_else(|| "unrecorded".to_string()),
            artifacts,
        })
    }

    pub fn verify(&self, root: impl AsRef<Path>) -> Result<(), String> {
        crate::release_sha::code_revision_is_honest(self)?;
        rustc_is_honest(&self.rustc)?;
        if !self.bench_run_ids.is_empty() || !self.eval_run_ids.is_empty() {
            return Err("bench_run_ids and eval_run_ids must stay empty; tool does not invent runs".into());
        }
        let root = root.as_ref();
        let live = Self::capture(
            root,
            &self
                .artifacts
                .iter()
                .map(|a| a.path.as_str())
                .collect::<Vec<_>>(),
        )?;
        crate::release_sha::revisions_match(&self.code_revision, &live.code_revision)?;
        if live.artifacts != self.artifacts {
            return Err("release artifact checksum or size mismatch".into());
        }
        if !self.limitations.is_empty() && !root.join(&self.limitations).is_file() {
            return Err(format!("limitations reference missing {}", self.limitations));
        }
        for (label, rel) in [("checkpoint_id", &self.checkpoint_id), ("model_id", &self.model_id)] {
            if rel.is_empty() {
                continue;
            }
            if rel.starts_with('/') || rel.contains("..") {
                return Err(format!("{label} refuses unsafe path {rel}"));
            }
            if !root.join(rel).is_file() {
                return Err(format!("{label} reference missing {rel}"));
            }
        }
        if self.cargo_lock != "unrecorded" {
            if self.cargo_lock != cargo_lock_field(root) {
                return Err("cargo_lock checksum mismatch".into());
            }
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

pub fn record_features(supplied: Option<&str>) -> Result<String, String> {
    let Some(raw) = supplied else {
        return Ok("unrecorded".to_string());
    };
    let mut names = Vec::new();
    for part in raw.split(',') {
        let name = part.trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
        {
            return Err(format!("features refuses invented or unsafe name {part:?}"));
        }
        names.push(name.to_string());
    }
    names.sort();
    names.dedup();
    Ok(names.join(","))
}

fn rustc_field() -> String {
    option_env!("RUSTC_VERSION")
        .unwrap_or("unrecorded")
        .to_string()
}

fn rustc_is_honest(value: &str) -> Result<(), String> {
    if value == "unrecorded" {
        return Ok(());
    }
    if value.is_empty() || value == env!("CARGO_PKG_VERSION") {
        return Err("rustc refuses the package version; toolchain stays unrecorded".into());
    }
    Ok(())
}

fn cargo_lock_field(root: &Path) -> String {
    if !root.join("Cargo.lock").is_file() {
        return "unrecorded".to_string();
    }
    match hash_artifact(root, "Cargo.lock") {
        Ok(art) => format!("{:016x}", art.checksum),
        Err(_) => "unrecorded".to_string(),
    }
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
        assert!(report.human().contains("pending | ids="));
        assert!(report.human().contains("provenance | rustc=") || report.human().contains("code_revision="));
        assert!(report.human().contains("code_revision="));
        assert!(report.human().contains("bench_run_ids="));
        assert!(report.human().contains("eval_run_ids="));
        assert!(report.human().contains("limitations="));
        assert!(report.json().contains("\"limitations\":\"docs/model-card.md\""));
        assert!(report.eval_run_ids.is_empty());
        assert!(report.human().contains("features="));
        assert!(report.human().contains("cargo_lock="));
        assert!(report.human().contains("target="));
        assert!(report.human().contains("human_approval"));
        assert!(report.json().contains("\"creates_tags\":false"));
        assert!(report.json().contains("\"pending_count\":"));
        assert!(report.json().contains("\"pending_ids\":"));
        assert!(report.json().contains("\"human_approval\""));
        assert!(report.gates.iter().any(|g| g.id == "human_approval" && g.status == GateStatus::Pending));
        assert!(report.human_approval == GateStatus::Pending);
        assert!(report.json().contains("\"bench_run_ids\":\"\""));
        assert!(report.json().contains("\"eval_run_ids\":\"\""));
        assert!(report.json().contains("\"code_revision\":"));
        assert!(report.json().contains("\"rustc\":\"unrecorded\"") || option_env!("RUSTC_VERSION").is_some());
        assert!(report.json().contains("\"target\":\"unrecorded\"") || option_env!("TARGET").is_some());
        assert!(!report.json().contains(&format!("\"rustc\":\"{}\"", env!("CARGO_PKG_VERSION"))));
        assert!(!report.json().contains("\"code_revision\":\"invented\""));
        assert!(report.json().contains("\"schema\":1"));
        assert!(report.json().contains("\"human_approval\":\"pending\""));
        assert_eq!(RELEASE_REPORT_SCHEMA_VERSION, 1);
        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("`RELEASE_REPORT_SCHEMA_VERSION = 1`"));
        assert!(docs.contains("la herramienta no aprueba ni crea el tag"));
        assert!(docs.contains("no inventa runs"));
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
    fn parse_ci_does_not_query_github() {
        assert!(parse_ci("job=benches conclusion=success\n").is_err());
        let text = required_pr_jobs()
            .into_iter()
            .map(|job| format!("job={} conclusion=success\n", job.id))
            .collect::<String>();
        let runs = parse_ci(&text).unwrap();
        assert_eq!(live_ci_gate(Some(&runs)).status, GateStatus::Pass);
        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("no consulta GitHub"));
    }

    #[test]
    fn rc_tag_converges_or_fails_closed_and_never_creates() {
        let aligned = TagCandidate {
            name: "v1.0.0".into(),
            tag_sha: "abc".into(),
            source_sha: "abc".into(),
        };
        let drifted = TagCandidate {
            name: "v1.0.0".into(),
            tag_sha: "abc".into(),
            source_sha: "def".into(),
        };
        assert_eq!(rc_tag_gate(Some(aligned.clone())).status, GateStatus::Pass);
        assert_eq!(rc_tag_gate(Some(drifted.clone())).status, GateStatus::Fail);
        assert_eq!(rc_tag_gate(None).status, GateStatus::Pending);
        assert_eq!(
            rc_tag_gate(Some(TagCandidate {
                name: "v0.9.0".into(),
                tag_sha: "abc".into(),
                source_sha: "abc".into(),
            }))
            .status,
            GateStatus::Fail
        );
        assert_eq!(create_tag(), Err("tag creation forbidden"));
        assert_eq!(approve(), Err("approval forbidden"));
        let verify = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/verify.md"));
        assert!(verify.contains("approval forbidden"));
        assert!(verify.contains("tag creation forbidden"));
        assert!(verify.contains("Does **not** cut `v1.0.0`") || verify.contains("No corta `v1.0.0`"));
        let cli = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/cli.md"));
        assert!(cli.contains("approval forbidden"));
        assert!(cli.contains("tag creation forbidden"));
        let rc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc.md"));
        assert!(rc.contains("approval forbidden"));
        assert!(rc.contains("tag creation forbidden"));
        let post = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/post-tag.md"));
        assert!(post.contains("tag creation forbidden"));
        assert!(post.contains("approval forbidden"));
        let sha = "a".repeat(40);
        let parsed = parse_tag(&format!("name=v1.0.0 tag_sha={sha} source_sha={sha}\n")).unwrap();
        assert_eq!(rc_tag_gate(Some(parsed)).status, GateStatus::Pass);
        assert!(parse_tag("name=v1.0.0 tag_sha=abc source_sha=abc\n").is_err());

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
        assert!(manifest.bench_run_ids.is_empty());
        assert!(manifest.eval_run_ids.is_empty());
        assert!(manifest.encode().contains("bench_run_ids=\n"));
        assert!(manifest.checkpoint_id.is_empty());
        assert!(manifest.model_id.is_empty());
        assert!(manifest.encode().contains("checkpoint_id=\n"));
        assert!(manifest.cargo_lock == "unrecorded" || manifest.cargo_lock.len() == 16);
        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("no inventan runs"));
        assert!(docs.contains("no inventa checkpoint_id"));
        assert!(docs.contains("no inventa versiones de dependencias"));
        assert_eq!(record_features(None).unwrap(), "unrecorded");
        assert_eq!(record_features(Some("b,a")).unwrap(), "a,b");
        assert!(record_features(Some("GPU")).is_err());
        assert!(docs.contains("no inventa features"));
        let mut invented = decoded.clone();
        invented.bench_run_ids = "fake-run".into();
        assert!(invented.verify(&root).unwrap_err().contains("does not invent runs"));
        let mut pkg = decoded.clone();
        pkg.rustc = env!("CARGO_PKG_VERSION").into();
        assert!(pkg.verify(&root).unwrap_err().contains("unrecorded"));


        let mut placeholder = decoded.clone();
        placeholder.code_revision = "not-a-sha".into();
        assert!(placeholder.verify(&root).unwrap_err().contains("code_revision"));
        decoded.verify(&root).unwrap();

        fs::write(root.join("ckpt.bin"), "ckpt\n").unwrap();
        let mut with_ids = decoded.clone();
        with_ids.checkpoint_id = "ckpt.bin".into();
        with_ids.model_id = "ckpt.bin".into();
        with_ids.verify(&root).unwrap();
        with_ids.checkpoint_id = "../secret".into();
        assert!(with_ids.verify(&root).unwrap_err().contains("unsafe"));

        fs::write(root.join("Cargo.lock"), "lock\n").unwrap();
        let locked = ReleaseManifest::capture(&root, &["Cargo.toml", "README.md"]).unwrap();
        assert_ne!(locked.cargo_lock, "unrecorded");
        locked.verify(&root).unwrap();
        fs::write(root.join("Cargo.lock"), "changed\n").unwrap();
        assert!(locked.verify(&root).unwrap_err().contains("cargo_lock"));

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
