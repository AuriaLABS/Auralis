//! Registry of Engine microbenchmarks. This slice only catalogs bins;
//! running them stays on the existing `cargo run --bin` paths.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchSpec {
    pub id: &'static str,
    pub bin: &'static str,
    pub kind: &'static str,
    pub summary: &'static str,
    pub default_args: &'static str,
    pub requires_alloc_profile: bool,
}

pub const BENCHES: &[BenchSpec] = &[
    BenchSpec {
        id: "engine",
        bin: "auralis_engine_bench",
        kind: "throughput",
        summary: "side-by-side train reference vs reuse",
        default_args: "3 20 5",
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "matmul",
        bin: "auralis_matmul_bench",
        kind: "kernel",
        summary: "forward matmul reference",
        default_args: "5 40 5",
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "matmul-bt",
        bin: "auralis_matmul_bt_bench",
        kind: "kernel",
        summary: "B-transpose matmul",
        default_args: "5 40 5",
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "matmul-blocked",
        bin: "auralis_matmul_blocked_bench",
        kind: "kernel",
        summary: "blocked matmul candidate",
        default_args: "",
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "kernel-grad-b",
        bin: "auralis_kernel_bench",
        kind: "kernel",
        summary: "matmul grad-B kernel",
        default_args: "5 40 5",
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "attention",
        bin: "auralis_attention_bench",
        kind: "kernel",
        summary: "attention forward",
        default_args: "5 80 5",
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "attention-backward",
        bin: "auralis_attention_backward_bench",
        kind: "kernel",
        summary: "attention backward",
        default_args: "5 80 5",
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "profile-alloc",
        bin: "auralis-profile",
        kind: "memory",
        summary: "engine allocation profile (reference vs reuse)",
        default_args: "2",
        requires_alloc_profile: true,
    },
    BenchSpec {
        id: "profile-train",
        bin: "auralis_train_profile",
        kind: "memory",
        summary: "train-path allocation profile",
        default_args: "",
        requires_alloc_profile: false,
    },
    BenchSpec {
        id: "profile-eval",
        bin: "auralis_eval_profile",
        kind: "memory",
        summary: "eval-path allocation profile",
        default_args: "",
        requires_alloc_profile: false,
    },
];

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
        format!(
            "id={}\nbin={}\nkind={}\nsummary={}\ndefault_args={}\nrequires_alloc_profile={}\ninvoke=cargo run --release{} --bin {} -- {}\n",
            self.id,
            self.bin,
            self.kind,
            self.summary,
            self.default_args,
            self.requires_alloc_profile,
            if self.requires_alloc_profile {
                " --features alloc-profile"
            } else {
                ""
            },
            self.bin,
            self.default_args
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
}
