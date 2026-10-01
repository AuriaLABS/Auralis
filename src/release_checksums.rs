//! Checksum gate for local release-check.
//!
//! Does not query the network and does not invent a stored manifest.

use crate::release::{Gate, GateStatus, ReleaseManifest, RELEASE_MANIFEST_VERSION};
use std::path::Path;

/// Absent stored manifest → pending.
/// A supplied `RELEASE_MANIFEST_VERSION = 1` text must decode and `verify`.
/// Decode error or checksum mismatch is fail-closed.
pub fn artifacts_checksums_gate(root: &Path, stored: Option<&str>) -> Gate {
    match stored {
        None => Gate {
            id: "artifacts_checksums",
            status: GateStatus::Pending,
            kind: "pending-external",
            detail: "use auralis release-manifest to emit/verify checksums".into(),
        },
        Some(text) => match ReleaseManifest::decode(text).and_then(|m| {
            m.verify(root)?;
            Ok(m)
        }) {
            Ok(manifest) => Gate {
                id: "artifacts_checksums",
                status: GateStatus::Pass,
                kind: "automated",
                detail: format!(
                    "RELEASE_MANIFEST_VERSION = {} verified {} artifacts",
                    manifest.version,
                    manifest.artifacts.len()
                ),
            },
            Err(err) => Gate {
                id: "artifacts_checksums",
                status: GateStatus::Fail,
                kind: "automated",
                detail: err,
            },
        },
    }
}

const _SCHEMA: u32 = RELEASE_MANIFEST_VERSION;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::release::{check_release, check_release_full, DEFAULT_RELEASE_ARTIFACTS};
    use std::fs;

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

    #[test]
    fn artifacts_checksums_verify_or_fail_closed_without_inventing() {
        let root = std::env::temp_dir().join(format!(
            "auralis-release-sum-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_tree(&root, DEFAULT_RELEASE_ARTIFACTS);
        let stored = ReleaseManifest::capture(&root, DEFAULT_RELEASE_ARTIFACTS)
            .unwrap()
            .encode();

        assert_eq!(
            artifacts_checksums_gate(&root, None).status,
            GateStatus::Pending
        );
        assert_eq!(
            artifacts_checksums_gate(&root, Some(&stored)).status,
            GateStatus::Pass
        );
        assert_eq!(
            artifacts_checksums_gate(&root, Some("not-a-manifest")).status,
            GateStatus::Fail
        );

        fs::write(root.join("README.md"), "changed\n").unwrap();
        assert_eq!(
            artifacts_checksums_gate(&root, Some(&stored)).status,
            GateStatus::Fail
        );
        write_tree(&root, DEFAULT_RELEASE_ARTIFACTS);
        let stored = ReleaseManifest::capture(&root, DEFAULT_RELEASE_ARTIFACTS)
            .unwrap()
            .encode();
        let local = check_release(&root);
        let pass = check_release_full(&root, None, None, Some(&stored), None, None, None);
        let fail = check_release_full(&root, None, None, Some("not-a-manifest"), None, None, None);
        let _ = fs::remove_dir_all(&root);
        assert!(local.automated_pass);
        assert!(local.gates.iter().any(|g| g.id == "artifacts_checksums"
            && g.status == GateStatus::Pending));
        assert!(pass.automated_pass);
        assert!(pass.gates.iter().any(|g| g.id == "artifacts_checksums"
            && g.status == GateStatus::Pass));
        assert!(!fail.automated_pass);

        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("use auralis release-manifest to emit/verify checksums"));
        assert!(docs.contains("`RELEASE_MANIFEST_VERSION = 1`"));
        assert!(docs.contains("`check_release_full`"));
        assert!(docs.contains("Does **not** cut `v1.0.0`"));
    }
}
