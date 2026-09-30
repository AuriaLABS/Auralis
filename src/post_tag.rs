//! #305 post-tag verification contract.

pub const POST_TAG_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Converge,
    Mismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Incident {
    None,
    ReleaseIncident,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactSha {
    pub tag: &'static str,
    pub source: &'static str,
    pub binary: &'static str,
    pub checksum: &'static str,
    pub manifest: &'static str,
}

/// Tag, source, binaries, checksums and manifest converge only when every SHA is identical.
pub fn converge(a: &ArtifactSha) -> bool {
    a.tag == a.source && a.source == a.binary && a.binary == a.checksum && a.checksum == a.manifest
}

pub fn mismatch(a: &ArtifactSha) -> bool {
    !converge(a)
}

/// Any discrepancy between tag/manifest/artifact is a release incident.
pub fn incident(a: &ArtifactSha) -> Incident {
    if mismatch(a) {
        Incident::ReleaseIncident
    } else {
        Incident::None
    }
}

/// Never mutate artifacts associated with an already-published tag.
pub fn mutate_tagged_artifact(_tag: &str) -> Result<(), &'static str> {
    Err("retrospective mutation forbidden")
}

/// This contract never creates a git tag. Tagging remains #84 and is still forbidden here.
pub fn create_tag() -> Result<(), &'static str> {
    Err("tag creation forbidden")
}

/// Rebuild/smoke runs from the published tag, not from the live checkout.
pub fn smoke_ok(from_tag: bool) -> bool {
    from_tag
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn aligned() -> ArtifactSha {
        ArtifactSha {
            tag: "abc",
            source: "abc",
            binary: "abc",
            checksum: "abc",
            manifest: "abc",
        }
    }

    fn drifted() -> ArtifactSha {
        ArtifactSha {
            tag: "abc",
            source: "abc",
            binary: "def",
            checksum: "abc",
            manifest: "abc",
        }
    }

    #[test]
    fn converge_mismatch_incident_and_never_tag_or_mutate() {
        assert!(converge(&aligned()));
        assert!(!mismatch(&aligned()));
        assert_eq!(incident(&aligned()), Incident::None);
        assert!(!converge(&drifted()));
        assert!(mismatch(&drifted()));
        assert_eq!(incident(&drifted()), Incident::ReleaseIncident);
        assert_eq!(mutate_tagged_artifact("v1.0.0"), Err("retrospective mutation forbidden"));
        assert_eq!(create_tag(), Err("tag creation forbidden"));
        assert!(smoke_ok(true));
        assert!(!smoke_ok(false));
    }

    #[test]
    fn post_tag_doc_matches_contract() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = fs::read_to_string(root.join("docs/post-tag.md")).expect("docs/post-tag.md");
        for needle in [
            "POST_TAG_SCHEMA_VERSION = 1",
            "tag, source, binaries, checksums and manifest converge",
            "mismatch is a release incident",
            "never mutate tagged artifacts",
            "Does **not** cut `v1.0.0`",
        ] {
            assert!(text.contains(needle), "post-tag.md missing {needle}");
        }
        let readme = fs::read_to_string(root.join("README.md")).expect("README.md");
        assert!(readme.contains("docs/post-tag.md"));
        let versions = fs::read_to_string(root.join("docs/versions.md")).expect("versions.md");
        assert!(versions.contains("POST_TAG_SCHEMA_VERSION = 1"));
        assert_eq!(POST_TAG_SCHEMA_VERSION, 1);
    }
}
