//! #302 public API policy. Internal is not stable by visibility.

pub const API_POLICY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Public,
    Internal,
    Experimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bump {
    Patch,
    Minor,
    Major,
}

pub fn classify(name: &str) -> Surface {
    match name {
        "cli" | "checkpoint-v1" | "local-api" | "session" => Surface::Public,
        "gpu-kernels" | "numeric" => Surface::Experimental,
        _ => Surface::Internal,
    }
}

pub fn supported(surface: Surface) -> bool {
    matches!(surface, Surface::Public)
}

/// A `pub` Rust item does not mint a compatibility promise.
pub fn from_visibility(_is_pub: bool) -> Surface {
    Surface::Internal
}

pub fn bump(surface: Surface, breaking: bool, additive: bool) -> Option<Bump> {
    match surface {
        Surface::Public if breaking => Some(Bump::Major),
        Surface::Public if additive => Some(Bump::Minor),
        Surface::Public => Some(Bump::Patch),
        Surface::Experimental | Surface::Internal => None,
    }
}

pub fn deprecate_ok(surface: Surface, announced: bool) -> bool {
    surface == Surface::Public && announced
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn internal_is_not_stable_by_visibility_and_experimental_has_no_major() {
        assert_eq!(classify("cli"), Surface::Public);
        assert!(supported(Surface::Public));
        assert_eq!(from_visibility(true), Surface::Internal);
        assert!(!supported(from_visibility(true)));
        assert_eq!(classify("src_optim"), Surface::Internal);
        assert!(!supported(Surface::Internal));
        assert_eq!(classify("gpu-kernels"), Surface::Experimental);
        assert!(!supported(Surface::Experimental));
        assert_eq!(bump(Surface::Public, true, false), Some(Bump::Major));
        assert_eq!(bump(Surface::Public, false, true), Some(Bump::Minor));
        assert_eq!(bump(Surface::Experimental, true, false), None);
        assert!(deprecate_ok(Surface::Public, true));
        assert!(!deprecate_ok(Surface::Public, false));
        assert!(!deprecate_ok(Surface::Internal, true));
    }

    #[test]
    fn api_policy_doc_matches_contract() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = fs::read_to_string(root.join("docs/api-policy.md")).expect("docs/api-policy.md");
        for needle in [
            "API_POLICY_SCHEMA_VERSION = 1",
            "internal is not stable by visibility",
            "experimental may break without a major",
            "Does **not** freeze every pub item",
        ] {
            assert!(text.contains(needle), "api-policy.md missing {needle}");
        }
        let readme = fs::read_to_string(root.join("README.md")).expect("README.md");
        assert!(readme.contains("docs/api-policy.md"));
        let versions = fs::read_to_string(root.join("docs/versions.md")).expect("versions.md");
        assert!(versions.contains("API_POLICY_SCHEMA_VERSION = 1"));
        assert_eq!(API_POLICY_SCHEMA_VERSION, 1);
    }
}
