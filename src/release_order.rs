//! Deterministic release-manifest ordering.
//!
//! Does not invent commit SHAs, run ids, or dependency versions.

use crate::release::ReleaseManifest;

/// Artifact lines are sorted by path. `code_revision` is `unknown` unless
/// `AURALIS_COMMIT_SHA` was set at build time. This does not invent a SHA.
pub fn manifest_is_ordered(manifest: &ReleaseManifest) -> Result<(), String> {
    let mut paths: Vec<&str> = manifest.artifacts.iter().map(|a| a.path.as_str()).collect();
    let mut sorted = paths.clone();
    sorted.sort_unstable();
    if paths != sorted {
        return Err("artifact inventory is not sorted".into());
    }
    let encoded = manifest.encode();
    let keys = [
        "auralis_release=",
        "code_revision=",
        "rustc=",
        "target=",
        "features=",
        "limitations=",
        "bench_run_ids=",
        "eval_run_ids=",
        "checkpoint_id=",
        "model_id=",
        "artifact_count=",
    ];
    let mut last = 0usize;
    for key in keys {
        let pos = encoded
            .find(key)
            .ok_or_else(|| format!("missing ordered field {key}"))?;
        if pos < last {
            return Err(format!("field {key} is out of order"));
        }
        last = pos;
    }
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
    fn manifest_order_does_not_invent_a_sha() {
        let root = std::env::temp_dir().join(format!("auralis-release-order-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for rel in ["README.md", "Cargo.toml"] {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "ok\n").unwrap();
        }
        crate::release_card::write_stub(&root);
        let manifest = ReleaseManifest::capture(&root, &["README.md", "Cargo.toml"]).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        manifest_is_ordered(&manifest).unwrap();
        assert!(manifest.encode().find("auralis_release=").unwrap() < manifest.encode().find("artifact_count=").unwrap());
        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("orden determinista"));
        assert!(docs.contains("no inventa el SHA"));
    }
}
