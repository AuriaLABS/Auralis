//! Compatibility-fixture gate for local release-check.
//!
//! Uses `COMPAT_SCHEMA_VERSION = 1`. Does not rewrite frozen formats.

use crate::compat::{accept, migrate, Format, COMPAT_SCHEMA_VERSION};
use crate::release::{Gate, GateStatus};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompatFixture {
    pub checkpoint: u32,
    pub manifest: u32,
    pub session: u32,
}

/// Absent fixture → pending. The CLI does not invent compatibility rows.
/// Present v1/v1/v1 that `accept` → pass.
/// version 0 or unknown → fail-closed.
/// `migrate` returns the same version; it never rewrites semantics.
pub fn compatibility_fixtures_gate(fixture: Option<CompatFixture>) -> Gate {
    match fixture {
        None => Gate {
            id: "compatibility_fixtures",
            status: GateStatus::Pending,
            kind: "pending-external",
            detail: "compatibility fixtures not supplied; migrate never rewrites".into(),
        },
        Some(fix) => {
            for (format, version) in [
                (Format::Checkpoint, fix.checkpoint),
                (Format::Manifest, fix.manifest),
                (Format::Session, fix.session),
            ] {
                if accept(format, version).is_err() {
                    return Gate {
                        id: "compatibility_fixtures",
                        status: GateStatus::Fail,
                        kind: "automated",
                        detail: format!("{format:?} version {version} fail-closed"),
                    };
                }
                match migrate(format, version) {
                    Ok(out) if out == version => {}
                    _ => {
                        return Gate {
                            id: "compatibility_fixtures",
                            status: GateStatus::Fail,
                            kind: "automated",
                            detail: format!("{format:?} migrate rewrote {version}"),
                        };
                    }
                }
            }
            Gate {
                id: "compatibility_fixtures",
                status: GateStatus::Pass,
                kind: "automated",
                detail: format!(
                    "COMPAT_SCHEMA_VERSION = {COMPAT_SCHEMA_VERSION} fixtures accept v1"
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_fixtures_pass_or_fail_closed_without_rewrite() {
        assert_eq!(
            compatibility_fixtures_gate(None).status,
            GateStatus::Pending
        );
        let v1 = CompatFixture {
            checkpoint: 1,
            manifest: 1,
            session: 1,
        };
        assert_eq!(
            compatibility_fixtures_gate(Some(v1)).status,
            GateStatus::Pass
        );
        let bad = CompatFixture {
            checkpoint: 1,
            manifest: 0,
            session: 1,
        };
        assert_eq!(
            compatibility_fixtures_gate(Some(bad)).status,
            GateStatus::Fail
        );
        let future = CompatFixture {
            checkpoint: 2,
            manifest: 1,
            session: 1,
        };
        assert_eq!(
            compatibility_fixtures_gate(Some(future)).status,
            GateStatus::Fail
        );
        let root = std::env::temp_dir().join(format!(
            "auralis-release-compat-{}",
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
        let pass = crate::release::check_release_full(&root, None, None, None, Some(v1), None, None);
        let fail = crate::release::check_release_full(&root, None, None, None, Some(bad), None, None);
        let _ = std::fs::remove_dir_all(&root);
        assert!(local.automated_pass);
        assert!(local.gates.iter().any(|g| g.id == "compatibility_fixtures"
            && g.status == GateStatus::Pending));
        assert!(pass.automated_pass);
        assert!(!fail.automated_pass);

        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("`COMPAT_SCHEMA_VERSION = 1`"));
        assert!(docs.contains("migrate never rewrites"));
        assert!(docs.contains("`check_release_full`"));
        assert!(docs.contains("Does **not** cut `v1.0.0`"));
    }
}
