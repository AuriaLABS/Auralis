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

    fn section_after<'a>(text: &'a str, heading: &str) -> &'a str {
        let start = text.find(heading).unwrap_or(0);
        let rest = &text[start..];
        match rest.find("\n## ") {
            Some(end) if end > 0 => &rest[..end],
            _ => rest,
        }
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

    const CLEAN_ENV: &[&str] = &[
        "cargo run --release -- check",
        "cargo run --release -- bpe",
        "cargo run --release -- train-fresh 2 /tmp/auralis-pruebas.bin 659918 2 1",
        "ok=true",
        "run_summary |",
        "cargo run --release --bin auralis_eval_profile -- 4",
        "eval_profile | mode=loss",
        "eval_profile | mode=generate_one",
    ];

    const VERIFY_SNIPPET: &str = concat!(
        "assert!(!auralis::ci_matrix::required_pr_jobs().is_empty());\n",
        "assert_eq!(auralis::sec::EXPECTED_UNSAFE_BLOCKS, 0);\n",
        "assert!(!auralis::bench::list_json().is_empty());"
    );

    #[test]
    fn required_docs_exist() {
        for name in [
            "architecture-config.md",
            "bench-registry.md",
            "brain-ab.md",
            "ci-pruebas.md",
            "sec-unsafe.md",
            "scheduler.md",
            "optimizer-boundary.md",
            "optimizer-variants.md",
            "rope-experiment.md",
            "external-memory.md",
            "memory-model-integration.md",
            "recurrent-reasoning.md",
            "kv-cache.md",
            "local-attention.md",
            "eval-registry.md",
            "reasoning-suite.md",
            "memory-suite.md",
            "code-suite.md",
            "moe-router.md",
            "gradient-clipping.md",
            "verify.md",
            "cli.md",
            "versions.md",
            "rc-gate.md",
            "model-card.md",
            "clean-env.md",
            "e1-workspace-profile.md",
            "e3-cpu-parallelism.md",
            "ENGINE_ARCHITECTURE.md",
            "ENGINE_LAYOUT.md",
        ] {
            let path = root().join("docs").join(name);
            assert!(path.is_file(), "missing docs/{name}");
        }
        assert!(root().join("VISION.md").is_file());
        assert!(root().join("ROADMAP.md").is_file());
        assert!(root().join("examples/tiny.cfg").is_file());
    }

    #[test]
    fn engine_architecture_doc_names_current_invariants() {
        let text = docs("ENGINE_ARCHITECTURE.md");
        for needle in [
            "# Auralis Engine 0.3 — arquitectura actual",
            "## 2. Rutas de ejecución",
            "## 3. Ownership y memoria",
            "## 4. Invariantes de shapes",
            "## 7. Kernels CPU",
            "## 8. Backend boundary",
            "## 9. Diagnósticos numéricos",
            "## 10. Observabilidad de rendimiento",
            "## 12. Checkpoint y reproducibilidad",
            "## 13. Política unsafe / SIMD / threads",
            "## 14. Evidencia reproducible",
            "## 15. Límites conocidos",
            "train_step_reuse",
            "matmul_row_slices_into",
            "EngineStepTiming",
            "ScalarCpuBackend",
            "OptimizedCpuBackend",
            "MockBackend",
            "auralis bench run",
            "Auralis-Scratch-Optimization: true",
            "no existe SIMD promovido",
            "no hay worker pool CPU productivo promovido",
        ] {
            assert!(
                text.contains(needle),
                "docs/ENGINE_ARCHITECTURE.md missing {needle}"
            );
        }

        let readme = readme();
        assert!(readme.contains("docs/ENGINE_ARCHITECTURE.md"));
        let roadmap =
            fs::read_to_string(root().join("ROADMAP.md")).expect("ROADMAP.md");
        assert!(roadmap.contains("docs/ENGINE_ARCHITECTURE.md"));
    }

    #[test]
    fn engine_layout_doc_names_current_layout() {
        let text = docs("ENGINE_LAYOUT.md");
        for needle in [
            "# Engine tensor layout audit",
            "Canonical layout today",
            "Physical transpose inventory",
            "packed_bt_persistent",
            "packed_bt_dynamic",
            "32x32x32",
            "32x32x96",
            "32x96x32",
            "32x32x100",
            "9x32x32",
            "not a cache-miss measurement",
            "RowSlices remains canonical",
            "Adam updates weights every training step",
            "auralis_layout_audit",
        ] {
            assert!(text.contains(needle), "docs/ENGINE_LAYOUT.md missing {needle}");
        }
    }

    #[test]
    fn architecture_config_docs_match_code() {
        let text = docs("architecture-config.md");
        for needle in [
            "auralis_architecture=5",
            "n_kv_head=4",
            "attention_window=0",
            "normalization=layernorm",
            "architecture-rmsnorm.cfg",
            "architecture-rope.cfg",
            "architecture-alibi.cfg",
            "position=learned_absolute",
            "rope",
            "alibi",
            "CHECKPOINT.architecture",
            "--model-config",
            "architecture-tiny.cfg",
            "architecture-small-1x1.cfg",
            "architecture-small-3x2.cfg",
            "RunConfig",
            "AURLIS02/AURLIS03",
        ] {
            assert!(text.contains(needle), "architecture-config.md missing {needle}");
        }
        for example in [
            "architecture-tiny.cfg",
            "architecture-small-1x1.cfg",
            "architecture-small-3x2.cfg",
            "architecture-rmsnorm.cfg",
            "architecture-rope.cfg",
            "architecture-alibi.cfg",
        ] {
            let cfg = crate::architecture::ArchitectureConfig::load(
                root().join("examples").join(example)
            ).unwrap_or_else(|e| panic!("{example}: {e}"));
            cfg.validate().unwrap();
        }
    }

    #[test]
    fn brain_ab_doc_names_reproducibility_contract() {
        let text = docs("brain-ab.md");
        for needle in [
            "same",
            "depth",
            "same training budget",
            "final training loss",
            "eval loss and perplexity",
            "canonical serialized optimizer-state bytes",
            "checkpoint bytes",
            "LayerNorm vs RMSNorm",
            "normalization policy",
            "positional policy",
            "Learned absolute vs RoPE",
            "Learned absolute vs ALiBi",
            "Optimizer comparisons",
            "optimizer-adamw",
            "optimizer-lion",
            "checkpoint_includes_optimizer",
            "does **not** declare a winner",
        ] {
            assert!(text.contains(needle), "docs/brain-ab.md missing {needle}");
        }
        assert!(readme().contains("docs/brain-ab.md"));
    }

    #[test]
    fn scheduler_docs_and_examples_match_code() {
        let text = docs("scheduler.md");
        for needle in [
            "auralis_scheduler=1",
            "--scheduler-config",
            "linear_warmup",
            "cosine",
            "CHECKPOINT.scheduler",
            "global_step=0",
            "Changing scheduler policy mid-run is rejected",
        ] {
            assert!(text.contains(needle), "docs/scheduler.md missing {needle}");
        }
        for example in [
            "scheduler-constant.cfg",
            "scheduler-warmup.cfg",
            "scheduler-cosine.cfg",
        ] {
            let cfg = crate::scheduler::SchedulerConfig::load(
                root().join("examples").join(example)
            ).unwrap_or_else(|e| panic!("{example}: {e}"));
            cfg.validate().unwrap();
        }
    }

    #[test]
    fn optimizer_boundary_doc_matches_code() {
        let text = docs("optimizer-boundary.md");
        for needle in [
            "dyn Optimizer",
            "AdamConfig",
            "OPTIMIZER_STATE_SCHEMA_VERSION = 1",
            "OptimizerStateIdentity",
            "AURLIS03 remains byte-format compatible",
            "Optimizer::set_learning_rate",
            "Future parameter groups",
        ] {
            assert!(text.contains(needle), "optimizer-boundary.md missing {needle}");
        }
        assert_eq!(crate::optim::OPTIMIZER_CONFIG_SCHEMA_VERSION, 1);
        assert_eq!(crate::optim::OPTIMIZER_STATE_SCHEMA_VERSION, 1);
    }

    #[test]
    fn optimizer_variants_doc_matches_contract() {
        let text = docs("optimizer-variants.md");
        for needle in [
            "AdamW",
            "Lion",
            "weight_decay=0",
            "decoupled weight decay",
            "AdamWState",
            "LionState",
            "auralis_optimizer_variant_bench",
            "does **not** measure model quality",
        ] {
            assert!(text.contains(needle), "optimizer-variants.md missing {needle}");
        }
        assert_eq!(crate::optim::ADAMW_STATE_SCHEMA_VERSION, 1);
        assert_eq!(crate::optim::LION_STATE_SCHEMA_VERSION, 1);
    }

    #[test]
    fn gradient_clipping_docs_and_examples_match_code() {
        let text = docs("gradient-clipping.md");
        for needle in [
            "RunConfig schema 2",
            "grad_clip_enabled=true",
            "clipping-off.cfg",
            "grad_norm_before_clip",
            "grad_norm_after_clip",
            "clip_applied",
            "schema 1 migrates explicitly",
            "resume only with clipping enabled",
        ] {
            assert!(text.contains(needle), "gradient-clipping.md missing {needle}");
        }

        let on = crate::run_config::RunConfig::load(root().join("examples/tiny.cfg"))
            .expect("examples/tiny.cfg");
        assert!(on.grad_clip_enabled);

        let off = crate::run_config::RunConfig::load(root().join("examples/clipping-off.cfg"))
            .expect("examples/clipping-off.cfg");
        assert!(!off.grad_clip_enabled);
        assert_eq!(crate::run_config::RUN_CONFIG_SCHEMA_VERSION, 2);
    }

    #[test]
    fn rope_experiment_doc_matches_contract() {
        let text = docs("rope-experiment.md");
        for needle in [
            "RoPE is **experimental**",
            "position=learned_absolute",
            "position=rope",
            "zero gradient",
            "auralis_brain_ab",
            "auralis_rope_context_bench",
            "contexts 4, 8, 16 and 32",
            "does **not** evaluate context beyond",
            "Changing the default requires a separate",
        ] {
            assert!(text.contains(needle), "rope-experiment.md missing {needle}");
        }
        assert_eq!(crate::architecture::ARCHITECTURE_CONFIG_SCHEMA_VERSION, 5);
        assert_eq!(crate::brain_ab::BRAIN_AB_SCHEMA_VERSION, 4);
    }

    #[test]
    fn alibi_experiment_doc_matches_contract() {
        let text = docs("alibi-experiment.md");
        for needle in [
            "ALiBi is **experimental**",
            "position=learned_absolute",
            "position=alibi",
            "zero gradient",
            "auralis_brain_ab",
            "auralis_alibi_context_bench",
            "contexts 4, 8, 16 and 32",
            "does **not** evaluate context beyond",
            "Changing the default requires a separate",
        ] {
            assert!(text.contains(needle), "alibi-experiment.md missing {needle}");
        }
        assert_eq!(crate::architecture::ARCHITECTURE_CONFIG_SCHEMA_VERSION, 5);
        assert_eq!(crate::brain_ab::BRAIN_AB_SCHEMA_VERSION, 4);
    }

    #[test]
    fn external_memory_doc_matches_contract() {
        let text = docs("external-memory.md");
        for needle in [
            "External memory is experimental",
            "write —",
            "read —",
            "query —",
            "reset —",
            "version —",
            "EXTERNAL_MEMORY_SCHEMA_VERSION = 1",
            "session is ephemeral",
            "**never serialized**",
            "reject-new",
            "auralis_memory_bench",
            "memory=off",
            "#42",
            "no claim that external memory improves reasoning",
        ] {
            assert!(text.contains(needle), "external-memory.md missing {needle}");
        }
        assert_eq!(crate::memory::EXTERNAL_MEMORY_SCHEMA_VERSION, 1);
        let bench = crate::bench::find("external-memory").expect("external-memory benchmark");
        assert_eq!(bench.bin, "auralis_memory_bench");
        assert_eq!(bench.default_args, "128 40");
    }

    #[test]
    fn memory_model_integration_doc_matches_contract() {
        let text = docs("memory-model-integration.md");
        for needle in [
            "Gpt::logits_with_memory",
            "MemoryInferenceMode::Off",
            "last_hidden_mean_add",
            "no implicit writes or resets occur",
            "0/3",
            "3/3",
            "auralis_memory_model_bench",
            "not** evidence of improved general reasoning",
            "No default changes",
        ] {
            assert!(
                text.contains(needle),
                "memory-model-integration.md missing {needle}"
            );
        }
        assert!(readme().contains("docs/memory-model-integration.md"));
        assert_eq!(
            crate::memory_integration::MEMORY_INTEGRATION_SCHEMA_VERSION,
            1
        );
        assert_eq!(crate::recurrent::RECURRENT_CONFIG_SCHEMA_VERSION, 1);
    }

    #[test]
    fn recurrent_reasoning_doc_matches_contract() {
        let text = docs("recurrent-reasoning.md");
        for needle in [
            "shared-weight recurrent computation",
            "reasoning_steps: 1",
            "bit-for-bit equality",
            "same weights",
            "12 total physical block applications",
            "R=1 -> 12 optimizer updates",
            "R=2 -> 6 optimizer updates",
            "R=3 -> 4 optimizer updates",
            "finite-difference",
            "default model remains one reasoning step",
        ] {
            assert!(
                text.contains(needle),
                "recurrent-reasoning.md missing {needle}"
            );
        }
        assert!(readme().contains("docs/recurrent-reasoning.md"));
        assert_eq!(crate::recurrent::RECURRENT_CONFIG_SCHEMA_VERSION, 1);
        let bench = crate::bench::find("recurrent-reasoning")
            .expect("recurrent-reasoning benchmark");
        assert_eq!(bench.bin, "auralis_recurrent_reasoning_bench");
    }

    #[test]
    fn gqa_mqa_doc_matches_contract() {
        let text = docs("gqa-mqa.md");
        for needle in [
            "MHA remains the default",
            "n_kv_head",
            "MQA",
            "GQA",
            "compact Wk/Wv",
            "KV-cache bytes",
            "run_attention_head_experiment",
            "auralis_gqa_mqa_bench",
            "quality evidence",
            "finite-difference",
            "prefill latency",
            "next-token latency",
            "does not auto-promote",
        ] {
            assert!(text.contains(needle), "gqa-mqa.md missing {needle}");
        }
        assert!(readme().contains("docs/gqa-mqa.md"));
        assert_eq!(crate::architecture::ARCHITECTURE_CONFIG_SCHEMA_VERSION, 5);
        assert_eq!(crate::brain_ab::ATTENTION_HEAD_AB_SCHEMA_VERSION, 1);
        let bench = crate::bench::find("gqa-mqa").expect("gqa-mqa benchmark");
        assert_eq!(bench.bin, "auralis_gqa_mqa_bench");
    }

    #[test]
    fn local_attention_doc_matches_contract() {
        let text = docs("local-attention.md");
        for needle in [
            "Dense causal attention remains the reference and default",
            "attention_window=0",
            "max(0, i + 1 - W)..=i",
            "compact probability storage",
            "run_attention_window_experiment",
            "auralis_local_attention_bench",
            "finite-difference",
            "probability-cache slots and bytes",
            "does not auto-promote",
        ] {
            assert!(text.contains(needle), "local-attention.md missing {needle}");
        }
        assert!(readme().contains("docs/local-attention.md"));
        assert_eq!(crate::architecture::ARCHITECTURE_CONFIG_SCHEMA_VERSION, 5);
        assert_eq!(crate::kv_cache::KV_CACHE_SCHEMA_VERSION, 3);
        assert_eq!(crate::brain_ab::ATTENTION_WINDOW_AB_SCHEMA_VERSION, 1);
        let bench = crate::bench::find("local-attention").expect("local-attention benchmark");
        assert_eq!(bench.bin, "auralis_local_attention_bench");
    }

    #[test]
    fn code_suite_doc_matches_contract() {
        let text = docs("code-suite.md");
        for needle in [
            "CODE_SUITE_SCHEMA_VERSION = 1",
            "CODE_SUITE_SEED = 131659918",
            "Code comprehension",
            "Compile repair",
            "Unit-test repair",
            "Bug-fix patch",
            "single-expression",
            "rustc --test",
            "protected tests",
            "compile-error",
            "borrow-check",
            "patch-rejected",
            "compile timeout 10 s",
            "auralis_code_suite_bench",
            "16 metrics",
        ] {
            assert!(text.contains(needle), "code-suite.md missing {needle}");
        }
        assert!(readme().contains("docs/code-suite.md"));
        assert_eq!(crate::code_suite::CODE_SUITE_SCHEMA_VERSION, 1);
        let suite = crate::code_suite::suite_definition(
            crate::code_suite::CodeSuiteProfile::Smoke,
            crate::code_suite::CODE_SUITE_SEED,
        );
        suite.validate().unwrap();
        assert_eq!(suite.metrics.len(), 16);
        let bench = crate::bench::find("code-suite").expect("code-suite benchmark");
        assert_eq!(bench.bin, "auralis_code_suite_bench");
        assert_eq!(bench.default_args, "both");
    }

    #[test]
    fn memory_suite_doc_matches_contract() {
        let text = docs("memory-suite.md");
        for needle in [
            "MEMORY_SUITE_SCHEMA_VERSION = 1",
            "MEMORY_SUITE_SEED = 135659918",
            "Exact retrieval",
            "Temporal order",
            "Conflict resolution",
            "Stale resolution",
            "Long-horizon recall",
            "latest-record-wins",
            "memory-off",
            "reject-new",
            "eviction = none",
            "wrong-order",
            "conflict-capture",
            "stale-value",
            "auralis_memory_suite_bench",
            "12 metrics",
        ] {
            assert!(text.contains(needle), "memory-suite.md missing {needle}");
        }
        assert!(readme().contains("docs/memory-suite.md"));
        assert_eq!(crate::memory_suite::MEMORY_SUITE_SCHEMA_VERSION, 1);
        let suite = crate::memory_suite::suite_definition(
            crate::memory_suite::MemorySuiteProfile::Smoke,
            crate::memory_suite::MEMORY_SUITE_SEED,
        );
        suite.validate().unwrap();
        assert_eq!(suite.metrics.len(), 12);
        let bench = crate::bench::find("memory-suite").expect("memory-suite benchmark");
        assert_eq!(bench.bin, "auralis_memory_suite_bench");
        assert_eq!(bench.default_args, "both");
    }

    #[test]
    fn reasoning_suite_doc_matches_contract() {
        let text = docs("reasoning-suite.md");
        for needle in [
            "REASONING_SUITE_SCHEMA_VERSION = 1",
            "REASONING_SUITE_SEED = 130659918",
            "Arithmetic composition",
            "Algorithmic sequence",
            "Variable binding",
            "Finite-state planning",
            "Distractor robustness",
            "train-difficulty",
            "eval-difficulty",
            "eval-length",
            "difficulty is fixed at 2",
            "invalid-format",
            "wrong-value",
            "wrong-length",
            "wrong-binding",
            "wrong-transition",
            "distractor-capture",
            "five task families",
            "all three splits",
            "auralis_reasoning_suite_bench",
            "model-independent",
        ] {
            assert!(text.contains(needle), "reasoning-suite.md missing {needle}");
        }
        assert!(readme().contains("docs/reasoning-suite.md"));
        assert_eq!(crate::reasoning_suite::REASONING_SUITE_SCHEMA_VERSION, 1);
        let smoke = crate::reasoning_suite::suite_definition(
            crate::reasoning_suite::ReasoningProfile::Smoke,
            crate::reasoning_suite::REASONING_SUITE_SEED,
        );
        smoke.validate().unwrap();
        let bench = crate::bench::find("reasoning-suite").expect("reasoning-suite benchmark");
        assert_eq!(bench.bin, "auralis_reasoning_suite_bench");
        assert_eq!(bench.default_args, "both");
    }

    #[test]
    fn eval_registry_doc_matches_contract() {
        let text = docs("eval-registry.md");
        for needle in [
            "EVALUATION_REGISTRY_SCHEMA_VERSION = 1",
            "EvaluationSuite",
            "EvaluationTask",
            "EvaluationMetric",
            "Baseline",
            "SuiteRef",
            "SuiteVersionCollision",
            "TaskVersionCollision",
            "MetricVersionCollision",
            "BaselineVersionCollision",
            "DatasetMetadata",
            "SeedPolicy",
            "exact run seed",
            "ResourceLimits",
            "deprecate_suite",
            "historical v1 baseline remains queryable",
            "auralis_eval_registry_smoke",
            "does not",
        ] {
            assert!(text.contains(needle), "eval-registry.md missing {needle}");
        }
        assert!(readme().contains("docs/eval-registry.md"));
        assert_eq!(crate::eval_registry::EVALUATION_REGISTRY_SCHEMA_VERSION, 1);
        let suite = crate::eval_registry::smoke_suite();
        suite.validate().unwrap();
        assert_ne!(suite.definition_fingerprint().unwrap(), 0);
    }

    #[test]
    fn moe_router_doc_matches_contract() {
        let text = docs("moe-router.md");
        for needle in [
            "MoE remains experimental",
            "MOE_ROUTER_SCHEMA_VERSION = 1",
            "DeterministicTopKRouter",
            "MoeRouter",
            "top_k",
            "dense-fallback",
            "dropped_tokens",
            "dispatch",
            "gather",
            "load coefficient of variation",
            "auralis_moe_router_bench",
            "dense model path remains the reference/default",
        ] {
            assert!(text.contains(needle), "moe-router.md missing {needle}");
        }
        assert!(readme().contains("docs/moe-router.md"));
        assert_eq!(crate::moe::MOE_ROUTER_SCHEMA_VERSION, 1);
        let bench = crate::bench::find("moe-router").expect("moe-router benchmark");
        assert_eq!(bench.bin, "auralis_moe_router_bench");
    }

    #[test]
    fn kv_cache_doc_matches_contract() {
        let text = docs("kv-cache.md");
        for needle in [
            "per-session key/value cache",
            "KV_CACHE_SCHEMA_VERSION",
            "query-head count and KV-head count",
            "prefill_kv_cache",
            "decode_kv_cached",
            "transactional across layers",
            "learned absolute",
            "RoPE",
            "ALiBi",
            "logical_bytes",
            "auralis_kv_cache_bench",
            "uncached",
            "cached",
            "not model state",
        ] {
            assert!(text.contains(needle), "kv-cache.md missing {needle}");
        }
        assert!(readme().contains("docs/kv-cache.md"));
        assert_eq!(crate::kv_cache::KV_CACHE_SCHEMA_VERSION, 3);
        let bench = crate::bench::find("kv-cache").expect("kv-cache benchmark");
        assert_eq!(bench.bin, "auralis_kv_cache_bench");
    }

    #[test]
    fn documented_versions_match_code() {
        let cargo = fs::read_to_string(root().join("Cargo.toml")).unwrap();
        assert!(cargo.contains("version = \"0.1.0\""));
        assert!(cargo.contains("edition = \"2021\""));
        assert_eq!(env!("CARGO_PKG_VERSION"), "0.1.0");
        assert_eq!(crate::run_config::RUN_CONFIG_SCHEMA_VERSION, 2);
        assert_eq!(crate::architecture::ARCHITECTURE_CONFIG_SCHEMA_VERSION, 5);
        assert_eq!(crate::brain_ab::BRAIN_AB_SCHEMA_VERSION, 4);
        assert_eq!(crate::brain_ab::ATTENTION_HEAD_AB_SCHEMA_VERSION, 1);
        assert_eq!(crate::brain_ab::ATTENTION_WINDOW_AB_SCHEMA_VERSION, 1);
        assert_eq!(crate::memory::EXTERNAL_MEMORY_SCHEMA_VERSION, 1);
        assert_eq!(crate::kv_cache::KV_CACHE_SCHEMA_VERSION, 3);
        assert_eq!(crate::moe::MOE_ROUTER_SCHEMA_VERSION, 1);
        assert_eq!(crate::eval_registry::EVALUATION_REGISTRY_SCHEMA_VERSION, 1);
        assert_eq!(crate::reasoning_suite::REASONING_SUITE_SCHEMA_VERSION, 1);
        assert_eq!(crate::memory_suite::MEMORY_SUITE_SCHEMA_VERSION, 1);
        assert_eq!(crate::code_suite::CODE_SUITE_SCHEMA_VERSION, 1);
        assert_eq!(crate::tool_protocol::TOOL_PROTOCOL_SCHEMA_VERSION, 1);
        assert_eq!(
            crate::memory_integration::MEMORY_INTEGRATION_SCHEMA_VERSION,
            1
        );
        assert_eq!(crate::scheduler::SCHEDULER_CONFIG_SCHEMA_VERSION, 1);
        assert_eq!(crate::optim::OPTIMIZER_CONFIG_SCHEMA_VERSION, 1);
        assert_eq!(crate::optim::OPTIMIZER_STATE_SCHEMA_VERSION, 1);
        assert_eq!(crate::manifest::MANIFEST_VERSION, 4);
        assert_eq!(crate::release::RELEASE_MANIFEST_VERSION, 1);
        let versions = docs("versions.md");
        for needle in [
            "`0.1.0`",
            "`2021`",
            "RUN_CONFIG_SCHEMA_VERSION = 2",
            "ARCHITECTURE_CONFIG_SCHEMA_VERSION = 5",
            "BRAIN_AB_SCHEMA_VERSION = 4",
            "ATTENTION_HEAD_AB_SCHEMA_VERSION = 1",
            "ATTENTION_WINDOW_AB_SCHEMA_VERSION = 1",
            "EXTERNAL_MEMORY_SCHEMA_VERSION = 1",
            "MEMORY_INTEGRATION_SCHEMA_VERSION = 1",
            "RECURRENT_CONFIG_SCHEMA_VERSION = 1",
            "KV_CACHE_SCHEMA_VERSION = 3",
            "MOE_ROUTER_SCHEMA_VERSION = 1",
            "EVALUATION_REGISTRY_SCHEMA_VERSION = 1",
            "REASONING_SUITE_SCHEMA_VERSION = 1",
            "MEMORY_SUITE_SCHEMA_VERSION = 1",
            "CODE_SUITE_SCHEMA_VERSION = 1",
            "TOOL_PROTOCOL_SCHEMA_VERSION = 1",
            "SCHEDULER_CONFIG_SCHEMA_VERSION = 1",
            "OPTIMIZER_CONFIG_SCHEMA_VERSION = 1",
            "OPTIMIZER_STATE_SCHEMA_VERSION = 1",
            "MANIFEST_VERSION = 4",
            "RELEASE_MANIFEST_VERSION = 1",
        ] {
            assert!(versions.contains(needle), "docs/versions.md missing {needle}");
        }
    }

    #[test]
    fn tool_protocol_doc_matches_contract() {
        let text = docs("tool-protocol.md");
        for needle in [
            "TOOL_PROTOCOL_SCHEMA_VERSION = 1",
            "ToolDefinition",
            "ToolRequest",
            "ToolResult",
            "ToolError",
            "ValidatedToolCall",
            "invoke_validated",
            "protocol-validation",
            "invalid-tool-response",
            "DeterministicMockExecutor",
            "grants **no authority**",
            "no tool execution through the public protocol path before strict validation",
        ] {
            assert!(text.contains(needle), "tool-protocol.md missing {needle}");
        }
        assert!(readme().contains("docs/tool-protocol.md"));
        assert_eq!(crate::tool_protocol::TOOL_PROTOCOL_SCHEMA_VERSION, 1);
    }

    #[test]
    fn rc_gate_doc_lists_default_artifacts() {
        let text = docs("rc-gate.md");
        assert_eq!(crate::release::RELEASE_MANIFEST_VERSION, 1);
        for art in crate::release::DEFAULT_RELEASE_ARTIFACTS {
            assert!(text.contains(art), "docs/rc-gate.md missing {art}");
        }
        for needle in [
            "nunca crea tags",
            "human_approval",
            "DEFAULT_RELEASE_ARTIFACTS",
            "release-check",
        ] {
            assert!(text.contains(needle), "docs/rc-gate.md missing {needle}");
        }
        let report = crate::release::check_release(root());
        assert!(report.automated_pass);
        assert_eq!(report.human_approval, crate::release::GateStatus::Pending);
        assert!(report
            .gates
            .iter()
            .any(|g| g.id == "model_card" && g.status == crate::release::GateStatus::Pass));
    }

    #[test]
    fn model_card_has_required_sections() {
        let text = docs("model-card.md");
        for needle in [
            "## Intended use",
            "## Out of scope",
            "## Architecture",
            "## Training data",
            "## Evaluation",
            "## Metrics",
            "## Limitations",
            "## Ethics / risks",
            "## Versioning",
            "## Supported vs experimental",
            "no inventa baselines",
        ] {
            assert!(text.contains(needle), "docs/model-card.md missing {needle}");
        }
    }

    #[test]
    fn model_card_supported_lists_every_cli_verb() {
        let card = docs("model-card.md");
        let supported = section_after(&card, "### Supported");
        for line in CLI_LINES {
            let verb = line.trim();
            assert!(supported.contains(verb), "Supported matrix missing {verb}");
        }
        let experimental = section_after(&card, "### Experimental");
        for needle in ["GPU", "#27", "v1.0.0", "multimodal"] {
            assert!(
                experimental.contains(needle),
                "Experimental matrix missing {needle}"
            );
        }
    }

    #[test]
    fn clean_env_doc_matches_pruebas_workflow() {
        let doc = docs("clean-env.md");
        let wf = fs::read_to_string(root().join(".github/workflows/pruebas.yml"))
            .expect("pruebas.yml");
        for needle in CLEAN_ENV {
            assert!(doc.contains(needle), "docs/clean-env.md missing {needle}");
            assert!(wf.contains(needle), "pruebas.yml missing {needle}");
        }
        assert!(doc.contains("cli smoke"));
        assert!(wf.contains("name: cli smoke"));
    }

    #[test]
    fn example_config_loads_and_matches_schema() {
        let cfg = crate::run_config::RunConfig::load(root().join("examples/tiny.cfg"))
            .expect("examples/tiny.cfg");
        assert_eq!(crate::run_config::RUN_CONFIG_SCHEMA_VERSION, 2);
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
        for flag in ["--config", "--model-config", "--scheduler-config", "--diagnostics", "--json", "--csv", "--out", "--verify"] {
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
            "rc-gate.md",
            "release-check",
            "model-card.md",
            "clean-env.md",
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
