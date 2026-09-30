//! #303 packaging and install-smoke contract.

pub const PACKAGING_SCHEMA_VERSION: u32 = 1;
pub const LICENSE_ID: &str = "MIT";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Source,
    License,
    Checksum,
    Manifest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub name: &'static str,
    pub kind: Kind,
}

pub fn layout() -> &'static [Slot] {
    &[
        Slot { name: "LICENSE", kind: Kind::License },
        Slot { name: "auralis-source.tar", kind: Kind::Source },
        Slot { name: "SHA256SUMS", kind: Kind::Checksum },
        Slot { name: "release-manifest", kind: Kind::Manifest },
    ]
}

/// Publish is blocked when the license inventory or checksum is missing,
/// or when a secret leaked into the artifact set.
pub fn publish_ok(has_license: bool, has_checksum: bool, has_secret: bool) -> bool {
    has_license && has_checksum && !has_secret
}

/// Smoke must run against a downloaded release artifact, not the dev tree.
pub fn smoke_ok(from_dev_checkout: bool) -> bool {
    !from_dev_checkout
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn license_blocks_publish_and_smoke_rejects_dev_tree() {
        assert_eq!(LICENSE_ID, "MIT");
        assert!(layout().iter().any(|s| s.name == "LICENSE"));
        assert!(publish_ok(true, true, false));
        assert!(!publish_ok(false, true, false));
        assert!(!publish_ok(true, false, false));
        assert!(!publish_ok(true, true, true));
        assert!(smoke_ok(false));
        assert!(!smoke_ok(true));
    }

    #[test]
    fn packaging_doc_matches_contract() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = fs::read_to_string(root.join("docs/packaging.md")).expect("docs/packaging.md");
        for needle in [
            "PACKAGING_SCHEMA_VERSION = 1",
            "smoke uses release artifacts, not the dev checkout",
            "missing license or checksum blocks publish",
            "Does **not** cut v1.0.0",
        ] {
            assert!(text.contains(needle), "packaging.md missing {needle}");
        }
        assert!(root.join("LICENSE").is_file(), "missing LICENSE");
        let license = fs::read_to_string(root.join("LICENSE")).expect("LICENSE");
        assert!(license.contains("MIT License"));
        let readme = fs::read_to_string(root.join("README.md")).expect("README.md");
        assert!(readme.contains("docs/packaging.md"));
        let versions = fs::read_to_string(root.join("docs/versions.md")).expect("versions.md");
        assert!(versions.contains("PACKAGING_SCHEMA_VERSION = 1"));
        assert_eq!(PACKAGING_SCHEMA_VERSION, 1);
    }
}
