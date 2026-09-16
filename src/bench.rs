//! Registry of Engine microbenchmarks. This slice catalogs bins and
//! exports the catalog; running kernels stays on `cargo run --bin`.

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
