//! #305 post-tag verification. Never creates the tag.

pub const POST_TAG_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostTagError {
    Mismatch,
    TagForbidden,
    MutateForbidden,
}

/// Tag, source, artifact and manifest must share one SHA.
pub fn converge(tag: &str, source: &str, artifact: &str, manifest: &str) -> Result<(), PostTagError> {
    if tag.is_empty() || source.is_empty() || artifact.is_empty() || manifest.is_empty() {
        return Err(PostTagError::Mismatch);
    }
    if tag == source && source == artifact && artifact == manifest {
        Ok(())
    } else {
        Err(PostTagError::Mismatch)
    }
}

/// A SHA mismatch is a release incident, not a rebuild hint.
pub fn incident(err: PostTagError) -> bool {
    err == PostTagError::Mismatch
}

/// Tagged artifacts must not be rewritten in place.
pub fn mutate_tagged_artifact() -> Result<(), PostTagError> {
    Err(PostTagError::MutateForbidden)
}

/// This contract verifies a tag; it never creates one.
pub fn create_tag() -> Result<(), PostTagError> {
    Err(PostTagError::TagForbidden)
}

/// Smoke after tag uses published artifacts, not the dev checkout.
pub fn smoke_ok(from_dev_checkout: bool) -> bool {
    !from_dev_checkout
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn converge_or_incident_and_never_tags() {
        assert_eq!(converge("abc", "abc", "abc", "abc"), Ok(()));
        let miss = converge("abc", "abc", "def", "abc");
        assert_eq!(miss, Err(PostTagError::Mismatch));
        assert!(incident(miss.unwrap_err()));
        assert_eq!(converge("", "abc", "abc", "abc"), Err(PostTagError::Mismatch));
        assert_eq!(create_tag(), Err(PostTagError::TagForbidden));
        assert_eq!(mutate_tagged_artifact(), Err(PostTagError::MutateForbidden));
        assert!(smoke_ok(false));
        assert!(!smoke_ok(true));
        assert_eq!(POST_TAG_SCHEMA_VERSION, 1);
    }

    #[test]
    fn post_tag_doc_matches_contract() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = fs::read_to_string(root.join("docs/post-tag.md")).expect("docs/post-tag.md");
        for needle in [
            "POST_TAG_SCHEMA_VERSION = 1",
            "tag source artifact and manifest must share one SHA",
            "mismatch is a release incident",
            "never creates the tag",
            "Does **not** cut `v1.0.0`",
        ] {
            assert!(text.contains(needle), "post-tag.md missing {needle}");
        }
        let readme = fs::read_to_string(root.join("README.md")).expect("README.md");
        assert!(readme.contains("docs/post-tag.md"));
        let versions = fs::read_to_string(root.join("docs/versions.md")).expect("versions.md");
        assert!(versions.contains("POST_TAG_SCHEMA_VERSION = 1"));
    }
}
