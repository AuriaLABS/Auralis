//! #55 structured research proposals.
//!
//! A proposal cannot become an experiment until baseline, budget and a
//! falsification condition exist. The author cannot approve their own proposal.

use crate::lab_registry::{ExperimentRecord, ExperimentStatus, LabRegistry};

pub const LAB_PROPOSAL_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalStatus {
    Draft,
    Rejected,
    Accepted,
}

impl ProposalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Rejected => "rejected",
            Self::Accepted => "accepted",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProposalError {
    Incomplete(&'static str),
    SameActor,
    NotAccepted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProposalSpec {
    pub id: String,
    pub problem: String,
    pub hypothesis: String,
    pub mechanism: String,
    pub baseline: String,
    pub primary_metric: String,
    pub budget: String,
    pub falsification: String,
    pub affected_scope: String,
    pub author_id: String,
    pub status: ProposalStatus,
}

impl ProposalSpec {
    pub fn validate(&self) -> Result<(), ProposalError> {
        for (field, value) in [
            ("problem", self.problem.as_str()),
            ("hypothesis", self.hypothesis.as_str()),
            ("baseline", self.baseline.as_str()),
            ("primary_metric", self.primary_metric.as_str()),
            ("budget", self.budget.as_str()),
            ("falsification", self.falsification.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ProposalError::Incomplete(field));
            }
        }
        Ok(())
    }
}

pub fn review_proposal(
    proposal: &mut ProposalSpec,
    reviewer_id: &str,
    accept: bool,
) -> Result<ProposalStatus, ProposalError> {
    proposal.validate()?;
    if reviewer_id == proposal.author_id {
        return Err(ProposalError::SameActor);
    }
    proposal.status = if accept {
        ProposalStatus::Accepted
    } else {
        ProposalStatus::Rejected
    };
    Ok(proposal.status)
}

pub fn promote_to_experiment(
    lab: &mut LabRegistry,
    proposal: &ProposalSpec,
) -> Result<String, ProposalError> {
    proposal.validate()?;
    if proposal.status != ProposalStatus::Accepted {
        return Err(ProposalError::NotAccepted);
    }
    let spec_id = format!("exp-{}", proposal.id);
    lab.register_experiment(ExperimentRecord {
        id: spec_id.clone(),
        hypothesis_id: proposal.id.clone(),
        baseline: proposal.baseline.clone(),
        primary_metric: proposal.primary_metric.clone(),
        budget: proposal.budget.clone(),
        success_criterion: proposal.falsification.clone(),
        status: ExperimentStatus::Proposed,
    });
    Ok(spec_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(id: &str, author: &str) -> ProposalSpec {
        ProposalSpec {
            id: id.into(),
            problem: "tiny eval plateaus".into(),
            hypothesis: "Lion beats Adam on the fixture".into(),
            mechanism: "decoupled sign updates".into(),
            baseline: "adam-default".into(),
            primary_metric: "eval_loss_milli".into(),
            budget: "1-fixture".into(),
            falsification: "metric<2000500".into(),
            affected_scope: "src/lab_runner.rs".into(),
            author_id: author.into(),
            status: ProposalStatus::Draft,
        }
    }

    #[test]
    fn incomplete_proposal_cannot_become_experiment() {
        let mut proposal = complete("p-bad", "researcher");
        proposal.falsification.clear();
        assert_eq!(
            proposal.validate(),
            Err(ProposalError::Incomplete("falsification"))
        );
        let mut lab = LabRegistry::new();
        assert_eq!(
            promote_to_experiment(&mut lab, &proposal),
            Err(ProposalError::Incomplete("falsification"))
        );
    }

    #[test]
    fn author_cannot_accept_and_reviewer_can_reject() {
        let mut proposal = complete("p1", "researcher");
        assert_eq!(
            review_proposal(&mut proposal, "researcher", true),
            Err(ProposalError::SameActor)
        );
        assert_eq!(
            review_proposal(&mut proposal, "reviewer", false).unwrap(),
            ProposalStatus::Rejected
        );
        let mut lab = LabRegistry::new();
        assert_eq!(
            promote_to_experiment(&mut lab, &proposal),
            Err(ProposalError::NotAccepted)
        );
    }

    #[test]
    fn accepted_proposal_promotes_to_registry_spec() {
        let mut proposal = complete("p2", "researcher");
        review_proposal(&mut proposal, "reviewer", true).unwrap();
        let mut lab = LabRegistry::new();
        let spec_id = promote_to_experiment(&mut lab, &proposal).unwrap();
        let spec = lab.experiment(&spec_id).unwrap();
        assert_eq!(spec.baseline, "adam-default");
        assert_eq!(spec.success_criterion, "metric<2000500");
        assert_eq!(spec.status, ExperimentStatus::Proposed);
    }
}
