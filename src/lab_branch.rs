//! #57 policy-controlled experimental branches and draft PRs.
//!
//! Automated paths never write `main`. Scope violations fail closed.

use crate::lab_proposal::{ProposalSpec, ProposalStatus};

pub const LAB_BRANCH_SCHEMA_VERSION: u32 = 1;
pub const PROTECTED_BASE: &str = "main";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchPrState {
    Draft,
    Archived,
}

impl BranchPrState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Archived => "archived",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabBranchError {
    NotAccepted,
    ScopeViolation(String),
    ProtectedBase,
    MissingBaseline,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExperimentalPr {
    pub branch: String,
    pub baseline: String,
    pub proposal_id: String,
    pub title: String,
    pub body: String,
    pub allowed_scope: String,
    pub draft: bool,
    pub state: BranchPrState,
}

pub fn branch_name(proposal_id: &str, baseline: &str) -> String {
    format!("exp/{proposal_id}/{}", &baseline[..baseline.len().min(8)])
}

pub fn open_draft(
    proposal: &ProposalSpec,
    baseline: &str,
    changed_path: &str,
) -> Result<ExperimentalPr, LabBranchError> {
    if proposal.status != ProposalStatus::Accepted {
        return Err(LabBranchError::NotAccepted);
    }
    if baseline.trim().is_empty() {
        return Err(LabBranchError::MissingBaseline);
    }
    if changed_path == PROTECTED_BASE || changed_path.starts_with("refs/heads/main") {
        return Err(LabBranchError::ProtectedBase);
    }
    if !changed_path.starts_with(&proposal.affected_scope)
        && changed_path != proposal.affected_scope
    {
        return Err(LabBranchError::ScopeViolation(changed_path.into()));
    }
    let branch = branch_name(&proposal.id, baseline);
    if branch == PROTECTED_BASE {
        return Err(LabBranchError::ProtectedBase);
    }
    Ok(ExperimentalPr {
        branch,
        baseline: baseline.into(),
        proposal_id: proposal.id.clone(),
        title: format!("exp({}): {}", proposal.id, proposal.hypothesis),
        body: format!(
            "proposal={}\nhypothesis={}\nscope={}\nbaseline={}\ncriterion={}\nbudget={}\nstate=draft\n",
            proposal.id,
            proposal.hypothesis,
            proposal.affected_scope,
            baseline,
            proposal.falsification,
            proposal.budget
        ),
        allowed_scope: proposal.affected_scope.clone(),
        draft: true,
        state: BranchPrState::Draft,
    })
}

pub fn archive_negative(pr: &mut ExperimentalPr) {
    pr.state = BranchPrState::Archived;
    pr.draft = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lab_proposal::ProposalStatus;

    fn accepted() -> ProposalSpec {
        ProposalSpec {
            id: "p57".into(),
            problem: "plateau".into(),
            hypothesis: "Lion beats Adam".into(),
            mechanism: "sign update".into(),
            baseline: "adam-default".into(),
            primary_metric: "eval_loss_milli".into(),
            budget: "1-fixture".into(),
            falsification: "metric<2000500".into(),
            affected_scope: "src/lab_runner.rs".into(),
            author_id: "researcher".into(),
            status: ProposalStatus::Accepted,
        }
    }

    #[test]
    fn draft_pr_is_traceable_and_never_main() {
        let pr = open_draft(&accepted(), "bb58339ddeadbeef", "src/lab_runner.rs").unwrap();
        assert!(pr.draft);
        assert_eq!(pr.state, BranchPrState::Draft);
        assert_eq!(pr.branch, "exp/p57/bb58339d");
        assert_ne!(pr.branch, PROTECTED_BASE);
        assert!(pr.body.contains("proposal=p57"));
        assert!(pr.body.contains("baseline=bb58339ddeadbeef"));
        assert!(pr.body.contains("criterion=metric<2000500"));
    }

    #[test]
    fn scope_and_main_writes_fail_closed() {
        assert!(matches!(
            open_draft(&accepted(), "bb58339d", "src/optim.rs"),
            Err(LabBranchError::ScopeViolation(_))
        ));
        assert_eq!(
            open_draft(&accepted(), "bb58339d", "main").unwrap_err(),
            LabBranchError::ProtectedBase
        );
        let mut draft = accepted();
        draft.status = ProposalStatus::Draft;
        assert_eq!(
            open_draft(&draft, "bb58339d", "src/lab_runner.rs").unwrap_err(),
            LabBranchError::NotAccepted
        );
    }

    #[test]
    fn negative_result_archives_without_dropping_ids() {
        let mut pr = open_draft(&accepted(), "bb58339ddeadbeef", "src/lab_runner.rs").unwrap();
        archive_negative(&mut pr);
        assert_eq!(pr.state, BranchPrState::Archived);
        assert_eq!(pr.proposal_id, "p57");
        assert_eq!(pr.baseline, "bb58339ddeadbeef");
    }
}
