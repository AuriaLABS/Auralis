//! Inventory of CI jobs. Catalog only; does not launch workflows.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CiJob {
    pub id: &'static str,
    pub workflow: &'static str,
    pub blocks_merge: bool,
    pub on_pr: bool,
    pub summary: &'static str,
}

pub const JOBS: &[CiJob] = &[
    CiJob {
        id: "core",
        workflow: "genesis.yml",
        blocks_merge: true,
        on_pr: true,
        summary: "historical gate: fmt-ish build + selected tests",
    },
    CiJob {
        id: "unit",
        workflow: "pruebas.yml",
        blocks_merge: true,
        on_pr: true,
        summary: "cargo test --lib",
    },
    CiJob {
        id: "integration",
        workflow: "pruebas.yml",
        blocks_merge: true,
        on_pr: true,
        summary: "genesis/properties/equivalence",
    },
    CiJob {
        id: "fuzz",
        workflow: "pruebas.yml",
        blocks_merge: true,
        on_pr: true,
        summary: "fuzz_parsers",
    },
    CiJob {
        id: "cli",
        workflow: "pruebas.yml",
        blocks_merge: true,
        on_pr: true,
        summary: "check + bpe + train-fresh smoke",
    },
    CiJob {
        id: "profile",
        workflow: "pruebas.yml",
        blocks_merge: true,
        on_pr: true,
        summary: "auralis_eval_profile",
    },
    CiJob {
        id: "benches",
        workflow: "pruebas.yml",
        blocks_merge: false,
        on_pr: false,
        summary: "short matmul/engine benches; main or manual",
    },
];

impl CiJob {
    pub fn json(&self) -> String {
        format!(
            "{{\"id\":\"{}\",\"workflow\":\"{}\",\"blocks_merge\":{},\"on_pr\":{},\"summary\":\"{}\"}}",
            self.id, self.workflow, self.blocks_merge, self.on_pr, self.summary
        )
    }
}

pub fn required_pr_jobs() -> Vec<&'static CiJob> {
    JOBS.iter().filter(|j| j.blocks_merge && j.on_pr).collect()
}

pub fn list_json() -> String {
    let mut out = format!("{{\"count\":{},\"jobs\":[", JOBS.len());
    for (i, job) in JOBS.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&job.json());
    }
    out.push_str("]}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn required_pr_set_is_stable() {
        let ids: Vec<&str> = required_pr_jobs().iter().map(|j| j.id).collect();
        assert_eq!(ids, ["core", "unit", "integration", "fuzz", "cli", "profile"]);
    }

    #[test]
    fn benches_do_not_block_prs() {
        let benches = JOBS.iter().find(|j| j.id == "benches").unwrap();
        assert!(!benches.on_pr);
        assert!(!benches.blocks_merge);
    }

    #[test]
    fn json_lists_every_job() {
        let blob = list_json();
        for job in JOBS {
            assert!(blob.contains(&format!("\"id\":\"{}\"", job.id)));
        }
    }

    #[test]
    fn docs_table_names_every_job() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/ci-pruebas.md");
        let text = fs::read_to_string(&path).expect("docs/ci-pruebas.md");
        for job in JOBS {
            let needle = format!("`{}`", job.id);
            assert!(
                text.contains(&needle),
                "docs/ci-pruebas.md missing {}",
                needle
            );
        }
    }
}
