//! Commit SHA honesty for the release manifest.
//!
//! Does not invent a SHA and does not create a tag.

use crate::manifest::build_revision;
use crate::release::ReleaseManifest;

/// `unknown` when `AURALIS_COMMIT_SHA` was unset. A 40-hex value is accepted.
/// Anything else is fail-closed. This never fabricates a revision.
pub fn code_revision_is_honest(manifest: &ReleaseManifest) -> Result<(), String> {
    if manifest.code_revision == "unknown" || is_sha(&manifest.code_revision) {
        Ok(())
    } else {
        Err("code_revision is neither unknown nor a commit sha".into())
    }
}

fn is_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_revision_stays_unknown_without_inventing_a_sha() {
        let root = std::env::temp_dir().join(format!("auralis-release-sha-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("README.md"), "ok\n").unwrap();
        crate::release_card::write_stub(&root);
        let manifest = ReleaseManifest::capture(&root, &["README.md"]).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(manifest.code_revision, build_revision());
        code_revision_is_honest(&manifest).unwrap();
        let mut fake = manifest.clone();
        fake.code_revision = "not-a-sha".into();
        assert!(code_revision_is_honest(&fake).is_err());
        fake.code_revision = "a".repeat(40);
        code_revision_is_honest(&fake).unwrap();
        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("no inventa el SHA"));
        assert!(docs.contains("Does **not** cut `v1.0.0`"));
    }
}
