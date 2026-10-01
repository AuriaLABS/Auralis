//! Model-card gate for local release-check.
//!
//! `CARD_SCHEMA_VERSION = 1`. Presence and required sections only.
//! Does not invent RC metrics, ppl, or SOTA.

use std::fs;
use std::path::Path;

use crate::release::{Gate, GateStatus};

pub const REQUIRED_CARD_SECTIONS: &[&str] = &[
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
];

/// Missing file or a missing required section is fail-closed.
/// A present Genesis card with the sections and `no inventa baselines` passes.
/// This does not promote the card to an RC metrics table.
pub fn model_card_gate(root: &Path) -> Gate {
    let path = root.join("docs/model-card.md");
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(_) => {
            return Gate {
                id: "model_card",
                status: GateStatus::Fail,
                kind: "automated",
                detail: "docs/model-card.md missing".into(),
            };
        }
    };
    for section in REQUIRED_CARD_SECTIONS {
        if !text.contains(section) {
            return Gate {
                id: "model_card",
                status: GateStatus::Fail,
                kind: "automated",
                detail: format!("docs/model-card.md missing {section}"),
            };
        }
    }
    if !text.contains("no inventa baselines") {
        return Gate {
            id: "model_card",
            status: GateStatus::Fail,
            kind: "automated",
            detail: "docs/model-card.md invents baselines".into(),
        };
    }
    Gate {
        id: "model_card",
        status: GateStatus::Pass,
        kind: "automated",
        detail: "CARD_SCHEMA_VERSION = 1 sections present; does not invent RC metrics".into(),
    }
}

pub fn write_stub(root: &Path) {
    let path = root.join("docs/model-card.md");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let mut body = String::from("# stub\n");
    for section in REQUIRED_CARD_SECTIONS {
        body.push_str(section);
        body.push('\n');
    }
    body.push_str("no inventa baselines\n");
    fs::write(path, body).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_card_required_sections_fail_closed_without_inventing_metrics() {
        let root = std::env::temp_dir().join(format!(
            "auralis-release-card-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        for rel in crate::release::DEFAULT_RELEASE_ARTIFACTS {
            let path = root.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, "ok\n").unwrap();
        }
        let missing = crate::release::check_release(&root);
        assert!(!missing.automated_pass);
        assert!(missing
            .gates
            .iter()
            .any(|g| g.id == "model_card" && g.status == GateStatus::Fail));

        write_stub(&root);
        let pass = crate::release::check_release(&root);
        assert!(pass.automated_pass);
        fs::write(root.join("docs/model-card.md"), "## Intended use\n").unwrap();
        let bad = crate::release::check_release(&root);
        assert!(!bad.automated_pass);
        let _ = fs::remove_dir_all(&root);

        let docs = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/rc-gate.md"));
        assert!(docs.contains("`CARD_SCHEMA_VERSION = 1`"));
        assert!(docs.contains("does not invent RC metrics"));
        assert!(docs.contains("Does **not** cut `v1.0.0`"));
    }
}
