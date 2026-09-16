//! Cheap docs/code sync checks. No network.

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    fn docs(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs").join(name);
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("missing {name}: {e}"))
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
}
