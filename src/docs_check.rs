//! Cheap docs/code sync checks. No network.

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    fn root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
    }

    fn docs(name: &str) -> String {
        let path = root().join("docs").join(name);
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("missing {name}: {e}"))
    }

    fn readme() -> String {
        fs::read_to_string(root().join("README.md")).expect("README.md")
    }

    #[test]
    fn required_docs_exist() {
        for name in [
            "bench-registry.md",
            "ci-pruebas.md",
            "sec-unsafe.md",
            "verify.md",
            "e1-workspace-profile.md",
            "e3-cpu-parallelism.md",
        ] {
            let path = root().join("docs").join(name);
            assert!(path.is_file(), "missing docs/{name}");
        }
        assert!(root().join("VISION.md").is_file());
        assert!(root().join("ROADMAP.md").is_file());
    }

    #[test]
    fn readme_documents_genesis_cycle() {
        let text = readme();
        for needle in [
            "cargo test",
            "cargo run --release -- check",
            "cargo run --release -- train",
            "cargo run --release -- eval",
            "cargo run --release -- chat",
            "VISION.md",
            "ROADMAP.md",
            "docs/verify.md",
        ] {
            assert!(text.contains(needle), "README.md missing {needle}");
        }
    }

    #[test]
    fn verify_page_lists_gate_commands() {
        let text = docs("verify.md");
        for needle in [
            "cargo test --lib",
            "cargo test --test fuzz_parsers",
            "cargo run --release -- check",
            "train-fresh",
            "bench list --json",
            "sec-audit",
        ] {
            assert!(text.contains(needle), "docs/verify.md missing {needle}");
        }
    }

    #[test]
    fn bench_registry_documents_cli_verbs() {
        let text = docs("bench-registry.md");
        for needle in [
            "auralis bench list",
            "auralis bench describe",
            "--json",
            "--csv",
            "bench_format::render",
        ] {
            assert!(text.contains(needle), "docs/bench-registry.md missing {needle}");
        }
    }

    #[test]
    fn sec_docs_name_current_limits() {
        let text = docs("sec-unsafe.md");
        for needle in [
            "EXPECTED_UNSAFE_BLOCKS",
            "MAX_CHECKPOINT_STRING",
            "MAX_BPE_TABLE",
            "sec_path::confine",
            "sec_overflow",
        ] {
            assert!(text.contains(needle), "docs/sec-unsafe.md missing {needle}");
        }
        assert_eq!(crate::sec::EXPECTED_UNSAFE_BLOCKS, 0);
        assert_eq!(crate::sec::MAX_CHECKPOINT_STRING, 1_048_576);
        assert_eq!(crate::sec::MAX_BPE_TABLE, 65_536);
    }
}
