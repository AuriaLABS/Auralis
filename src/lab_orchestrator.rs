//! #56 multi-agent lab orchestrator.
//!
//! Exclusive scopes cannot be double-claimed. Implementation and review
//! require distinct actors. Handoffs are explicit records, not private chat.

use crate::lab_proposal::ProposalSpec;
use std::collections::BTreeMap;

pub const LAB_ORCHESTRATOR_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabRole {
    Researcher,
    Implementer,
    Verifier,
    Reviewer,
}

impl LabRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Researcher => "researcher",
            Self::Implementer => "implementer",
            Self::Verifier => "verifier",
            Self::Reviewer => "reviewer",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkStatus {
    Queued,
    Claimed,
    HandedOff,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchestratorError {
    ScopeConflict { scope: String, holder: String },
    DuplicateProposal,
    SameActor,
    Unknown(String),
    IncompatibleConclusions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkItem {
    pub id: String,
    pub proposal_id: String,
    pub scope: String,
    pub status: WorkStatus,
    pub actors: BTreeMap<&'static str, String>,
    pub handoff: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LabOrchestrator {
    pub schema_version: u32,
    items: BTreeMap<String, WorkItem>,
    scopes: BTreeMap<String, String>,
    fingerprints: BTreeMap<String, String>,
}

impl LabOrchestrator {
    pub fn new() -> Self {
        Self {
            schema_version: LAB_ORCHESTRATOR_SCHEMA_VERSION,
            ..Self::default()
        }
    }

    pub fn enqueue(&mut self, proposal: &ProposalSpec) -> Result<String, OrchestratorError> {
        let fingerprint = format!(
            "{}|{}|{}",
            proposal.hypothesis.trim(),
            proposal.baseline.trim(),
            proposal.affected_scope.trim()
        );
        if let Some(existing) = self.fingerprints.get(&fingerprint) {
            return Err(if existing == &proposal.id {
                OrchestratorError::DuplicateProposal
            } else {
                OrchestratorError::DuplicateProposal
            });
        }
        let item_id = format!("work-{}", proposal.id);
        self.items.insert(
            item_id.clone(),
            WorkItem {
                id: item_id.clone(),
                proposal_id: proposal.id.clone(),
                scope: proposal.affected_scope.clone(),
                status: WorkStatus::Queued,
                actors: BTreeMap::new(),
                handoff: None,
            },
        );
        self.fingerprints.insert(fingerprint, proposal.id.clone());
        Ok(item_id)
    }

    pub fn claim(
        &mut self,
        item_id: &str,
        role: LabRole,
        actor: &str,
    ) -> Result<(), OrchestratorError> {
        let scope = self
            .items
            .get(item_id)
            .ok_or_else(|| OrchestratorError::Unknown(item_id.into()))?
            .scope
            .clone();
        if let Some(holder) = self.scopes.get(&scope) {
            if holder != item_id {
                return Err(OrchestratorError::ScopeConflict {
                    scope,
                    holder: holder.clone(),
                });
            }
        }
        let item = self
            .items
            .get_mut(item_id)
            .ok_or_else(|| OrchestratorError::Unknown(item_id.into()))?;
        if role == LabRole::Reviewer {
            if let Some(implementer) = item.actors.get(LabRole::Implementer.as_str()) {
                if implementer == actor {
                    return Err(OrchestratorError::SameActor);
                }
            }
        }
        item.actors.insert(role.as_str(), actor.into());
        item.status = WorkStatus::Claimed;
        self.scopes.insert(scope, item_id.into());
        Ok(())
    }

    pub fn handoff(&mut self, item_id: &str, note: &str) -> Result<String, OrchestratorError> {
        let item = self
            .items
            .get_mut(item_id)
            .ok_or_else(|| OrchestratorError::Unknown(item_id.into()))?;
        let record = format!(
            "handoff | item={} scope={} actors={} note={}",
            item.id,
            item.scope,
            item.actors
                .iter()
                .map(|(role, actor)| format!("{role}={actor}"))
                .collect::<Vec<_>>()
                .join(","),
            note
        );
        item.handoff = Some(record.clone());
        item.status = WorkStatus::HandedOff;
        if let Some(scope) = self.scopes.get(&item.scope).cloned() {
            let _ = scope;
        }
        self.scopes.remove(&item.scope);
        Ok(record)
    }

    pub fn item(&self, id: &str) -> Option<&WorkItem> {
        self.items.get(id)
    }

    pub fn aggregate(&self, left: &str, right: &str) -> Result<String, OrchestratorError> {
        if left.trim() != right.trim() {
            return Err(OrchestratorError::IncompatibleConclusions);
        }
        Ok(left.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lab_proposal::{ProposalSpec, ProposalStatus};

    fn proposal(id: &str, scope: &str, hypothesis: &str) -> ProposalSpec {
        ProposalSpec {
            id: id.into(),
            problem: "plateau".into(),
            hypothesis: hypothesis.into(),
            mechanism: "sign update".into(),
            baseline: "adam-default".into(),
            primary_metric: "eval_loss_milli".into(),
            budget: "1-fixture".into(),
            falsification: "metric<2000500".into(),
            affected_scope: scope.into(),
            author_id: "researcher".into(),
            status: ProposalStatus::Accepted,
        }
    }

    #[test]
    fn exclusive_scope_and_duplicate_proposals_conflict() {
        let mut orch = LabOrchestrator::new();
        let a = proposal("p-a", "src/optim.rs", "Lion wins");
        let b = proposal("p-b", "src/optim.rs", "AdamW wins");
        let work_a = orch.enqueue(&a).unwrap();
        let work_b = orch.enqueue(&b).unwrap();
        orch.claim(&work_a, LabRole::Implementer, "agent-1").unwrap();
        let err = orch.claim(&work_b, LabRole::Implementer, "agent-2").unwrap_err();
        assert!(matches!(err, OrchestratorError::ScopeConflict { .. }));
        assert_eq!(
            orch.enqueue(&a).unwrap_err(),
            OrchestratorError::DuplicateProposal
        );
    }

    #[test]
    fn implementer_cannot_self_review_and_handoff_is_explicit() {
        let mut orch = LabOrchestrator::new();
        let work = orch
            .enqueue(&proposal("p-c", "src/lab_runner.rs", "mock eval holds"))
            .unwrap();
        orch.claim(&work, LabRole::Implementer, "agent-1").unwrap();
        assert_eq!(
            orch.claim(&work, LabRole::Reviewer, "agent-1").unwrap_err(),
            OrchestratorError::SameActor
        );
        orch.claim(&work, LabRole::Reviewer, "agent-2").unwrap();
        let note = orch.handoff(&work, "ready for #54").unwrap();
        assert!(note.contains("implementer=agent-1"));
        assert!(note.contains("reviewer=agent-2"));
        assert_eq!(orch.item(&work).unwrap().status, WorkStatus::HandedOff);
    }

    #[test]
    fn incompatible_conclusions_are_not_merged() {
        let orch = LabOrchestrator::new();
        assert_eq!(
            orch.aggregate("pass", "fail").unwrap_err(),
            OrchestratorError::IncompatibleConclusions
        );
        assert_eq!(orch.aggregate("pass", "pass").unwrap(), "pass");
    }
}
