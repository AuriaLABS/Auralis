//! Local release-candidate checklist and artifact provenance.
//!
//! Reports verifiable gates only. Never creates tags or records human approval.

use crate::ci_matrix::required_pr_jobs;
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
    "LICENSE",
    "docs/api-policy.md",
    "docs/packaging.md",
    "docs/baselines.md",
    "docs/post-tag.md",
];
