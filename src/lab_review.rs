//! #54 independent lab review.
//!
//! Criteria are read from the original #52 spec. The reviewer identity must
//! differ from the implementer. Non-comparable evidence is inconclusive.

use crate::lab_registry::{ExperimentResult, LabError, LabRegistry};

pub const LAB_REVIEW_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReviewDecision {
    Pass,
    Fail,
    Inconclusive,
}

impl ReviewDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Inconclusive => "inconclusive",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabReviewError {
    SameActor,
    Missing(LabError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReviewReport {
    pub spec_id: String,
    pub run_id: String,
    pub reviewer_id: String,
    pub reviewer_role: String,
    pub implementer_id: String,
    pub decision: ReviewDecision,
    pub reason: String,
}

pub fn review_run(
    lab: &LabRegistry,
    spec_id: &str,
    run_id: &str,
    reviewer_id: &str,
    reviewer_role: &str,
    implementer_id: &str,
) -> Result<ReviewReport, LabReviewError> {
    if reviewer_id == implementer_id {
        return Err(LabReviewError::SameActor);
    }
    let spec = lab
        .experiment(spec_id)
        .ok_or_else(|| LabReviewError::Missing(LabError::Unknown(spec_id.into())))?;
    let result = lab
        .results_for(spec_id)
        .into_iter()
        .find(|item| item.run_id == run_id)
        .cloned()
        .ok_or_else(|| LabReviewError::Missing(LabError::Unknown(run_id.into())))?;
    let (decision, reason) = decide(spec.success_criterion.as_str(), spec.baseline.as_str(), &result);
    Ok(ReviewReport {
        spec_id: spec_id.into(),
        run_id: run_id.into(),
        reviewer_id: reviewer_id.into(),
        reviewer_role: reviewer_role.into(),
        implementer_id: implementer_id.into(),
        decision,
        reason,
    })
}

fn decide(
    criterion: &str,
    baseline: &str,
    result: &ExperimentResult,
) -> (ReviewDecision, String) {
    if result.summary.is_empty() || !result.summary.contains("metric=") {
        return (
            ReviewDecision::Inconclusive,
            "result is not comparable to the original spec".into(),
        );
    }
    if !result.summary.contains(&format!("baseline={baseline}")) {
        return (
            ReviewDecision::Inconclusive,
            "result baseline does not match the spec".into(),
        );
    }
    let Some(metric) = parse_metric(&result.summary) else {
        return (
            ReviewDecision::Inconclusive,
            "metric missing from result".into(),
        );
    };
    let Some(limit) = parse_metric_limit(criterion) else {
        return (
            ReviewDecision::Inconclusive,
            "success criterion is not a metric threshold".into(),
        );
    };
    if metric < limit {
        (
            ReviewDecision::Pass,
            format!("metric {metric} held original criterion {criterion}"),
        )
    } else {
        (
            ReviewDecision::Fail,
            format!("metric {metric} missed original criterion {criterion}"),
        )
    }
}

fn parse_metric(summary: &str) -> Option<u64> {
    let start = summary.find("metric=")? + "metric=".len();
    let token = summary[start..].split_whitespace().next()?;
    token.parse().ok()
}

fn parse_metric_limit(criterion: &str) -> Option<u64> {
    let start = criterion.find("metric<")? + "metric<".len();
    criterion[start..].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lab_registry::{ExperimentRecord, ExperimentStatus};
    use crate::lab_runner::{run_fixture, LabFixture, LabRunRequest};

    fn lab_with_eval(seed: u64, run_id: &str) -> LabRegistry {
        let mut lab = LabRegistry::new();
        lab.register_experiment(ExperimentRecord {
            id: "e1".into(),
            hypothesis_id: "h1".into(),
            baseline: "adam-default".into(),
            primary_metric: "eval_loss_milli".into(),
            budget: "1-fixture".into(),
            success_criterion: "metric<2000500".into(),
            status: ExperimentStatus::Proposed,
        });
        run_fixture(
            &mut lab,
            &LabRunRequest {
                spec_id: "e1".into(),
                run_id: run_id.into(),
                commit: "200a1801".into(),
                config: "tiny-mock".into(),
                seed,
                timeout_ms: 10,
                fixture: LabFixture::MockEval,
            },
        )
        .unwrap();
        lab
    }

    #[test]
    fn pass_and_fail_use_original_spec() {
        let pass_lab = lab_with_eval(1, "r-pass");
        let pass = review_run(&pass_lab, "e1", "r-pass", "verifier", "Verifier", "implementer")
            .unwrap();
        assert_eq!(pass.decision, ReviewDecision::Pass);

        let fail_lab = lab_with_eval(800, "r-fail");
        let fail = review_run(&fail_lab, "e1", "r-fail", "verifier", "Verifier", "implementer")
            .unwrap();
        assert_eq!(fail.decision, ReviewDecision::Fail);
        assert!(fail.reason.contains("metric<2000500"));
    }

    #[test]
    fn non_comparable_is_inconclusive_and_same_actor_is_rejected() {
        let mut lab = lab_with_eval(1, "r1");
        lab.record_result(ExperimentResult {
            run_id: "r-empty".into(),
            spec_id: "e1".into(),
            commit: "x".into(),
            config: "y".into(),
            negative: false,
            summary: "".into(),
        })
        .unwrap();
        let report = review_run(&lab, "e1", "r-empty", "verifier", "Verifier", "implementer")
            .unwrap();
        assert_eq!(report.decision, ReviewDecision::Inconclusive);

        let err = review_run(&lab, "e1", "r1", "same", "Verifier", "same").unwrap_err();
        assert_eq!(err, LabReviewError::SameActor);
    }

    #[test]
    fn intentional_false_positive_is_rejected() {
        let mut lab = lab_with_eval(1, "r1");
        lab.record_result(ExperimentResult {
            run_id: "r-fake".into(),
            spec_id: "e1".into(),
            commit: "x".into(),
            config: "y".into(),
            negative: false,
            summary: "fixture=mock-eval seed=1 metric=2000900 baseline=adam-default".into(),
        })
        .unwrap();
        let report = review_run(&lab, "e1", "r-fake", "verifier", "Verifier", "implementer")
            .unwrap();
        assert_eq!(report.decision, ReviewDecision::Fail);
    }
}
