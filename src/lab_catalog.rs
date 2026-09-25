//! #58 searchable catalog of prior experimental evidence.
//!
//! Positive, negative and inconclusive records stay distinct. Incomparable
//! metadata is never collapsed into one outcome.

use crate::lab_registry::ExperimentResult;
use crate::lab_review::{ReviewDecision, ReviewReport};

pub const LAB_CATALOG_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceOutcome {
    Positive,
    Negative,
    Inconclusive,
}

impl EvidenceOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
            Self::Inconclusive => "inconclusive",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceEntry {
    pub spec_id: String,
    pub run_id: String,
    pub hypothesis: String,
    pub subsystem: String,
    pub metric: String,
    pub config: String,
    pub commit: String,
    pub budget: String,
    pub outcome: EvidenceOutcome,
}

impl EvidenceEntry {
    pub fn comparable_key(&self) -> String {
        format!(
            "{}|{}|{}|{}",
            self.subsystem, self.metric, self.config, self.commit
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LabCatalog {
    pub schema_version: u32,
    entries: Vec<EvidenceEntry>,
}

impl LabCatalog {
    pub fn new() -> Self {
        Self {
            schema_version: LAB_CATALOG_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }

    pub fn index(
        &mut self,
        spec_id: &str,
        hypothesis: &str,
        subsystem: &str,
        metric: &str,
        budget: &str,
        result: &ExperimentResult,
        review: Option<&ReviewReport>,
    ) {
        let outcome = match review.map(|item| item.decision) {
            Some(ReviewDecision::Pass) => EvidenceOutcome::Positive,
            Some(ReviewDecision::Fail) => EvidenceOutcome::Negative,
            Some(ReviewDecision::Inconclusive) => EvidenceOutcome::Inconclusive,
            None if result.negative => EvidenceOutcome::Negative,
            None => EvidenceOutcome::Positive,
        };
        self.entries.push(EvidenceEntry {
            spec_id: spec_id.into(),
            run_id: result.run_id.clone(),
            hypothesis: hypothesis.into(),
            subsystem: subsystem.into(),
            metric: metric.into(),
            config: result.config.clone(),
            commit: result.commit.clone(),
            budget: budget.into(),
            outcome,
        });
    }

    pub fn search(&self, needle: &str) -> Vec<&EvidenceEntry> {
        let needle = needle.to_ascii_lowercase();
        self.entries
            .iter()
            .filter(|entry| {
                [
                    entry.hypothesis.as_str(),
                    entry.subsystem.as_str(),
                    entry.metric.as_str(),
                    entry.config.as_str(),
                    entry.outcome.as_str(),
                    entry.spec_id.as_str(),
                ]
                .iter()
                .any(|field| field.to_ascii_lowercase().contains(&needle))
            })
            .collect()
    }

    pub fn flag_duplicate(
        &self,
        hypothesis: &str,
        subsystem: &str,
        metric: &str,
        config: &str,
    ) -> Option<&EvidenceEntry> {
        self.entries.iter().find(|entry| {
            entry.hypothesis == hypothesis
                && entry.subsystem == subsystem
                && entry.metric == metric
                && entry.config == config
        })
    }

    pub fn comparable_group(&self, key: &str) -> Vec<&EvidenceEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.comparable_key() == key)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lab_review::ReviewDecision;

    fn result(run_id: &str, config: &str, commit: &str, negative: bool) -> ExperimentResult {
        ExperimentResult {
            run_id: run_id.into(),
            spec_id: "e1".into(),
            commit: commit.into(),
            config: config.into(),
            negative,
            summary: "metric=1".into(),
        }
    }

    fn review(decision: ReviewDecision) -> ReviewReport {
        ReviewReport {
            spec_id: "e1".into(),
            run_id: "r1".into(),
            reviewer_id: "v".into(),
            reviewer_role: "Verifier".into(),
            implementer_id: "i".into(),
            decision,
            reason: "fixture".into(),
        }
    }

    #[test]
    fn search_keeps_negative_and_inconclusive_distinct() {
        let mut cat = LabCatalog::new();
        cat.index(
            "e1",
            "Lion beats Adam",
            "src/optim.rs",
            "eval_loss_milli",
            "1-fixture",
            &result("r-neg", "tiny", "aaa", true),
            Some(&review(ReviewDecision::Fail)),
        );
        cat.index(
            "e2",
            "Lion beats Adam",
            "src/optim.rs",
            "eval_loss_milli",
            "1-fixture",
            &result("r-inc", "other-split", "bbb", false),
            Some(&review(ReviewDecision::Inconclusive)),
        );
        let hits = cat.search("lion");
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|e| e.outcome == EvidenceOutcome::Negative));
        assert!(hits
            .iter()
            .any(|e| e.outcome == EvidenceOutcome::Inconclusive));
        assert_ne!(hits[0].comparable_key(), hits[1].comparable_key());
    }

    #[test]
    fn duplicate_proposal_is_flagged_before_compute() {
        let mut cat = LabCatalog::new();
        cat.index(
            "e1",
            "Lion beats Adam",
            "src/optim.rs",
            "eval_loss_milli",
            "1-fixture",
            &result("r1", "tiny", "aaa", true),
            None,
        );
        let dup = cat.flag_duplicate(
            "Lion beats Adam",
            "src/optim.rs",
            "eval_loss_milli",
            "tiny",
        );
        assert_eq!(dup.unwrap().run_id, "r1");
        assert!(cat
            .flag_duplicate("new idea", "src/optim.rs", "eval_loss_milli", "tiny")
            .is_none());
    }
}
