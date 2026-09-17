//! Cheap docs/code sync checks. No network.

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

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

    fn markdown_files() -> Vec<PathBuf> {
        let mut out = vec![
            root().join("README.md"),
            root().join("VISION.md"),
            root().join("ROADMAP.md"),
        ];
        let docs_dir = root().join("docs");
        if let Ok(entries) = fs::read_dir(&docs_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("md") {
                    out.push(path);
                }
            }
        }
        out.sort();
        out
    }

    fn internal_targets(markdown: &str) -> Vec<String> {
        let mut targets = Vec::new();
        let bytes = markdown.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'[' {
                if let Some(close) = markdown[i..].find(']') {
                    let after = i + close + 1;
                    if after < bytes.len() && bytes[after] == b'(' {
                        if let Some(end) = markdown[after + 1..].find(')') {
                            let raw = markdown[after + 1..after + 1 + end].trim();
                            let href = raw.split('#').next().unwrap_or("").trim();
                            if !href.is_empty()
                                && !href.starts_with("http://")
                                && !href.starts_with("https://")
                                && !href.starts_with("mailto:")
                            {
                                targets.push(href.to_string());
                            }
                            i = after + 1 + end + 1;
                            continue;
                        }
                    }
                }
            }
            i += 1;
        }
        targets
    }

    fn usage_block() -> String {
        let src = fs::read_to_string(root().join("src/main.rs")).expect("src/main.rs");
        let start = src.find("fn usage()").expect("fn usage() in main.rs");
        src[start..].to_string()
    }

    fn rust_fences(markdown: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = markdown;
        while let Some(start) = rest.find("```rust\n") {
            rest = &rest[start + 8..];
            match rest.find("```") {
                Some(end) => {
                    out.push(rest[..end].trim().to_string());
                    rest = &rest[end + 3..];
                }
                None => break,
            }
        }
        out
    }

    const CLI_LINES: &[&str] = &[
        "auralis train ",
        "auralis train-fresh ",
        "auralis config ",
        "auralis inspect ",
        "auralis release-check ",
        "auralis release-manifest ",
        "auralis sec-audit ",
        "auralis bench list",
        "auralis bench describe",
        "auralis eval ",
        "auralis chat ",
        "auralis check",
        "auralis bpe",
    ];

    const VERIFY_SNIPPET: &str = concat!(
        "assert!(!auralis::ci_matrix::required_pr_jobs().is_empty());\n",
        "assert_eq!(auralis::sec::EXPECTED_UNSAFE_BLOCKS, 0);\n",
        "assert!(!auralis::bench::list_json().is_empty());"
    );

    #[test]
    fn required_docs_exist() {
        for name in [
            "bench-registry.md",
            "ci-pruebas.md",
            "sec-unsafe.md",
            "verify.md",
            "cli.md",
            "versions.md",
            "e1-workspace-profile.md",
            "e3-cpu-parallelism.md",
        ] {
            let path = root().join("docs").join(name);
            assert!(path.is_file(), "missing docs/{name}");
        }
        assert!(root().join("VISION.md").is_file());
        assert!(root().join("ROADMAP.md").is_file());
        assert!(root().join("examples/tiny.cfg").is_file());
    }

    #[test]
    fn documented_versions_match_code() {
        let cargo = fs::read_to_string(root().join("Cargo.toml")).unwrap();
        assert!(cargo.contains("version = \"0.1.0\""));
        assert!(cargo.contains("edition = \"2021\""));
        assert_eq!(env!("CARGO_PKG_VERSION"), "0.1.0");
        assert_eq!(crate::run_config::RUN_CONFIG_SCHEMA_VERSION, 1);
        assert_eq!(crate::manifest::MANIFEST_VERSION, 4);
        let versions = docs("versions.md");
        for needle in [
            "`0.1.0`",
            "`2021`",
            "RUN_CONFIG_SCHEMA_VERSION = 1",
            "MANIFEST_VERSION = 4",
        ] {
            assert!(versions.contains(needle), "docs/versions.md missing {needle}");
        }
    }

    #[test]
    fn example_config_loads_and_matches_schema() {
        let cfg = crate::run_config::RunConfig::load(root().join("examples/tiny.cfg"))
            .expect("examples/tiny.cfg");
        assert_eq!(crate::run_config::RUN_CONFIG_SCHEMA_VERSION, 1);
        assert_eq!(cfg.seed, 659_918);
        assert_eq!(cfg.batch_size, 4);
        assert_eq!(cfg.gradient_accumulation_steps, 1);
        assert_eq!(cfg.bpe_merges, 64);
        cfg.validate().unwrap();
        assert!(docs("verify.md").contains("examples/tiny.cfg"));
        assert!(docs("cli.md").contains("examples/tiny.cfg"));
    }

    #[test]
    fn internal_markdown_links_resolve() {
        let mut dangling = Vec::new();
        for file in markdown_files() {
            let text = fs::read_to_string(&file).unwrap();
            let base = file.parent().unwrap();
            for href in internal_targets(&text) {
                let target = base.join(&href);
                if !target.exists() {
                    dangling.push(format!(
                        "{} -> {}",
                        file.strip_prefix(root()).unwrap_or(&file).display(),
                        href
                    ));
                }
            }
        }
        assert!(
            dangling.is_empty(),
            "dangling internal markdown links:\n{}",
            dangling.join("\n")
        );
    }

    #[test]
    fn cli_doc_matches_usage_fn() {
        let usage = usage_block();
        let cli = docs("cli.md");
        for line in CLI_LINES {
            assert!(usage.contains(line), "fn usage() missing {line}");
            assert!(cli.contains(line.trim()), "docs/cli.md missing {line}");
        }
        for flag in ["--config", "--json", "--csv", "--out", "--verify"] {
            assert!(usage.contains(flag), "fn usage() missing {flag}");
            assert!(cli.contains(flag), "docs/cli.md missing {flag}");
        }
    }

    #[test]
    fn verify_rust_snippet_is_canonical_and_runs() {
        let fences = rust_fences(&docs("verify.md"));
        assert_eq!(fences, [VERIFY_SNIPPET]);
        assert!(!crate::ci_matrix::required_pr_jobs().is_empty());
        assert_eq!(crate::sec::EXPECTED_UNSAFE_BLOCKS, 0);
        assert!(!crate::bench::list_json().is_empty());
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
            "docs/cli.md",
            "docs/versions.md",
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
            "cli.md",
            "examples/tiny.cfg",
            "versions.md",
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
