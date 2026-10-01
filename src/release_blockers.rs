//! Declared-blockers gate for local release-check.
//!
//! Does not query GitHub. #68 GPU kernels are experimental and not a v1 release gate.

use crate::release::{Gate, GateStatus};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeclaredBlocker {
    pub number: u64,
    pub open: bool,
    pub release_gate: bool,
}

/// Absent snapshot → pending. The CLI does not query GitHub.
/// An open issue marked `release_gate` is fail-closed.
/// Open #68 with `release_gate = false` does not fail the gate.
pub fn declared_blockers_gate(blockers: Option<&[DeclaredBlocker]>) -> Gate {
    match blockers {
        None => Gate {
            id: "declared_blockers",
            status: GateStatus::Pending,
            kind: "pending-external",
            detail: "declared blockers not supplied; release-check does not query GitHub".into(),
        },
        Some(blockers) => {
            for blocker in blockers {
                if blocker.open && blocker.release_gate {
                    return Gate {
                        id: "declared_blockers",
                        status: GateStatus::Fail,
                        kind: "automated",
                        detail: format!("open release blocker {}", blocker.number),
                    };
                }
            }
            Gate {
                id: "declared_blockers",
                status: GateStatus::Pass,
                kind: "automated",
                detail: "no open declared release blocker; #68 is not a v1 release gate".into(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_blockers_fail_closed_without_querying_github() {
        assert_eq!(declared_blockers_gate(None).status, GateStatus::Pending);
        let gpu_only = [DeclaredBlocker {
            number: 68,
            open: true,
            release_gate: false,
        }];
        assert_eq!(
            declared_blockers_gate(Some(&gpu_only)).status,
            GateStatus::Pass
        );
        let release = [DeclaredBlocker {
            number: 121,
            open: true,
            release_gate: true,
        }];
        assert_eq!(declared_blockers_gate(Some(&release)).status, GateStatus::Fail);
        let closed = [DeclaredBlocker {
            number: 121,
            open: false,
            release_gate: true,
        }];
        assert_eq!(declared_blockers_gate(Some(&closed)).status, GateStatus::Pass);

        let root = std::env::temp_dir().join(format!(
            "auralis-release-blockers-{}",
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
            None,
            Some(&gpu_only),
        );
        let fail = crate::release::check_release_full(
            &root,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&release),
        );
        let _ = std::fs::remove_dir_all(&root);
        assert!(local.automated_pass);
        assert!(local
            .gates
            .iter()
            .any(|g| g.id == "declared_blockers" && g.status == GateStatus::Pending));
        assert!(pass.automated_pass);
        assert!(!fail.automated_pass);

        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("does not query GitHub"));
        assert!(docs.contains("#68 is not a v1 release gate"));
        assert!(docs.contains("Does **not** cut `v1.0.0`"));
    }
}
