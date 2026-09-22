//! Registry of Engine microbenchmarks and their optional in-process runners.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchRunnerKind {
    Engine,
    Matmul,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchSpec {
    pub id: &'static str,
    pub bin: &'static str,
    pub kind: &'static str,
    pub summary: &'static str,
    pub default_args: &'static str,
    pub runner: Option<BenchRunnerKind>,
    pub requires_alloc_profile: bool,
}

pub const BENCHES: &[BenchSpec] = &[
    BenchSpec {
        id: "engine",
        bin: "auralis_engine_bench",
        kind: "throughput",
        summary: "side-by-side train reference vs reuse",
        default_args: "3 20 5",
        runner: Some(BenchRunnerKind::Engine),
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "matmul",
        bin: "auralis_matmul_bench",
        kind: "kernel",
        summary: "forward matmul reference",
        default_args: "5 40 5",
        runner: Some(BenchRunnerKind::Matmul),
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "matmul-bt",
        bin: "auralis_matmul_bt_bench",
        kind: "kernel",
        summary: "B-transpose matmul",
        default_args: "5 40 5",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "matmul-blocked",
        bin: "auralis_matmul_blocked_bench",
        kind: "kernel",
        summary: "blocked matmul candidate",
        default_args: "",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "kernel-grad-b",
        bin: "auralis_kernel_bench",
        kind: "kernel",
        summary: "matmul grad-B kernel",
        default_args: "5 40 5",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "attention",
        bin: "auralis_attention_bench",
        kind: "kernel",
        summary: "attention forward",
        default_args: "5 80 5",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "attention-backward",
        bin: "auralis_attention_backward_bench",
        kind: "kernel",
        summary: "attention backward",
        default_args: "5 80 5",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "external-memory",
        bin: "auralis_memory_bench",
        kind: "brain-memory",
        summary: "external-memory retrieval, capacity and persistence",
        default_args: "128 40",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "memory-model",
        bin: "auralis_memory_model_bench",
        kind: "brain-memory",
        summary: "opt-in model-memory retrieval/fusion task and cost",
        default_args: "100 7",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "recurrent-reasoning",
        bin: "auralis_recurrent_reasoning_bench",
        kind: "brain-reasoning",
        summary: "shared-weight recurrent reasoning quality/cost curve",
        default_args: "",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "kv-cache",
        bin: "auralis_kv_cache_bench",
        kind: "brain-generation",
        summary: "cached vs uncached autoregressive decode latency and memory",
        default_args: "",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "gqa-mqa",
        bin: "auralis_gqa_mqa_bench",
        kind: "brain-attention",
        summary: "MHA vs GQA/MQA quality, decode latency and KV-cache memory",
        default_args: "",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "local-attention",
        bin: "auralis_local_attention_bench",
        kind: "brain-attention",
        summary: "dense vs causal local-window quality, scaling and probability-cache memory",
        default_args: "",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "moe-router",
        bin: "auralis_moe_router_bench",
        kind: "brain-moe",
        summary: "deterministic top-k routing, capacity/fallback and dispatch/gather overhead",
        default_args: "",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "profile-alloc",
        bin: "auralis-profile",
        kind: "memory",
        summary: "engine allocation profile (reference vs reuse)",
        default_args: "2",
        runner: None,
        requires_alloc_profile: true,
    },
    BenchSpec {
        id: "profile-train",
        bin: "auralis_train_profile",
        kind: "memory",
        summary: "train-path allocation profile",
        default_args: "",
        runner: None,
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "profile-eval",
        bin: "auralis_eval_profile",
        kind: "memory",
        summary: "eval-path allocation profile",
        default_args: "",
        runner: None,
        requires_alloc_profile: false,
    },
];


pub const INTERNAL_ONLY_BINS: &[&str] = &[
    "auralis_bench_result_smoke",
    "auralis_workspace_profile",
    "auralis_training_diagnostics_bench",
    "auralis_forward_diagnostics_bench",
    "auralis_matmul_rowslices_vs_blocked_bench",
    "auralis_engine_soak",
    "auralis_engine_step_timing_bench",
];

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

impl BenchSpec {
    pub fn line(&self) -> String {
        format!(
            "bench | id={} bin={} kind={} feature={} args={} | {}",
            self.id,
            self.bin,
            self.kind,
            if self.requires_alloc_profile {
                "alloc-profile"
            } else {
                "-"
            },
            if self.default_args.is_empty() {
                "-"
            } else {
                self.default_args
            },
            self.summary
        )
    }

    pub fn describe(&self) -> String {
        let invoke = if self.runner.is_some() {
            format!("auralis bench run {}", self.id)
        } else {
            format!(
                "cargo run --release{} --bin {} -- {}",
                if self.requires_alloc_profile { " --features alloc-profile" } else { "" },
                self.bin,
                self.default_args
            )
        };
        format!(
            "id={}\nbin={}\nkind={}\nsummary={}\ndefault_args={}\nrequires_alloc_profile={}\nlibrary_runner={}\ninvoke={}\n",
            self.id,
            self.bin,
            self.kind,
            self.summary,
            self.default_args,
            self.requires_alloc_profile,
            self.runner.is_some(),
            invoke
        )
    }

    pub fn json(&self) -> String {
        format!(
            "{{\"id\":\"{}\",\"bin\":\"{}\",\"kind\":\"{}\",\"summary\":\"{}\",\"default_args\":\"{}\",\"requires_alloc_profile\":{}}}",
            json_escape(self.id),
            json_escape(self.bin),
            json_escape(self.kind),
            json_escape(self.summary),
            json_escape(self.default_args),
            self.requires_alloc_profile
        )
    }

    pub fn csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{}\n",
            csv_escape(self.id),
            csv_escape(self.bin),
            csv_escape(self.kind),
            csv_escape(self.summary),
            csv_escape(self.default_args),
            self.requires_alloc_profile
        )
    }
}

pub fn list() -> &'static [BenchSpec] {
    BENCHES
}

pub fn find(id: &str) -> Option<&'static BenchSpec> {
    BENCHES.iter().find(|b| b.id == id || b.bin == id)
}

pub fn list_report() -> String {
    let mut out = format!("bench-list | count={}\n", BENCHES.len());
    for spec in BENCHES {
        out.push_str(&spec.line());
        out.push('\n');
    }
    out
}

pub fn list_json() -> String {
    let mut out = String::from("{\"count\":");
    out.push_str(&BENCHES.len().to_string());
    out.push_str(",\"benches\":[");
    for (i, spec) in BENCHES.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&spec.json());
    }
    out.push_str("]}\n");
    out
}

pub fn csv_header() -> &'static str {
    "id,bin,kind,summary,default_args,requires_alloc_profile\n"
}

pub fn list_csv() -> String {
    let mut out = csv_header().to_string();
    for spec in BENCHES {
        out.push_str(&spec.csv_row());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_ids_are_unique_and_nonempty() {
        let mut seen = Vec::new();
        for spec in BENCHES {
            assert!(!spec.id.is_empty());
            assert!(!spec.bin.is_empty());
            assert!(!seen.contains(&spec.id));
            seen.push(spec.id);
        }
        assert_eq!(seen.len(), BENCHES.len());
        assert!(BENCHES.len() >= 8);
    }

    #[test]
    fn find_accepts_id_or_bin_name() {
        let a = find("engine").unwrap();
        let b = find("auralis_engine_bench").unwrap();
        assert_eq!(a, b);
        assert!(find("does-not-exist").is_none());
    }

    #[test]
    fn first_library_runners_are_explicit() {
        assert_eq!(find("engine").unwrap().runner, Some(BenchRunnerKind::Engine));
        assert_eq!(find("matmul").unwrap().runner, Some(BenchRunnerKind::Matmul));
        assert!(find("attention").unwrap().runner.is_none());
    }

    #[test]
    fn internal_benchmark_allowlist_covers_workspace_profiler() {
        assert!(INTERNAL_ONLY_BINS.contains(&"auralis_workspace_profile"));
        assert!(INTERNAL_ONLY_BINS.contains(&"auralis_engine_step_timing_bench"));
    }

    #[test]
    fn json_export_contains_every_id() {
        let blob = list_json();
        assert!(blob.starts_with("{\"count\":"));
        for spec in BENCHES {
            assert!(blob.contains(&format!("\"id\":\"{}\"", spec.id)));
        }
        assert_eq!(blob.matches("\"id\":").count(), BENCHES.len());
    }

    #[test]
    fn csv_export_has_header_and_one_row_per_bench() {
        let blob = list_csv();
        let lines: Vec<&str> = blob.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines[0], "id,bin,kind,summary,default_args,requires_alloc_profile");
        assert_eq!(lines.len(), BENCHES.len() + 1);
        assert!(lines.iter().any(|l| l.starts_with("engine,")));
    }
}
