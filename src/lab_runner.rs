//! #53 controlled experiment runner.
//!
//! Consumes #52 specs. Only allowlisted in-process fixtures run here.
//! Timeout and corrupt artifacts are failures, never partial success.

use crate::lab_registry::{
    ExperimentResult, ExperimentStatus, LabError, LabRegistry,
};

pub const LAB_RUNNER_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabFixture {
    MockEval,
    MockTimeout,
    MockCorrupt,
}

impl LabFixture {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MockEval => "mock-eval",
            Self::MockTimeout => "mock-timeout",
            Self::MockCorrupt => "mock-corrupt",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabRunnerError {
    UnknownFixture,
    Incomplete(LabError),
    Timeout,
    CorruptArtifact,
    NotRunning,
}

impl From<LabError> for LabRunnerError {
    fn from(err: LabError) -> Self {
        Self::Incomplete(err)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabRunRequest {
    pub spec_id: String,
    pub run_id: String,
    pub commit: String,
    pub config: String,
    pub seed: u64,
    pub timeout_ms: u64,
    pub fixture: LabFixture,
}

pub fn run_fixture(
    lab: &mut LabRegistry,
    request: &LabRunRequest,
) -> Result<ExperimentResult, LabRunnerError> {
    lab.start(&request.spec_id)?;
    let baseline = {
        let experiment = lab
            .experiment(&request.spec_id)
            .ok_or_else(|| LabRunnerError::Incomplete(LabError::Unknown(request.spec_id.clone())))?;
        if experiment.status != ExperimentStatus::Running {
            return Err(LabRunnerError::NotRunning);
        }
        experiment.baseline.clone()
    };
    if request.timeout_ms == 0 || request.fixture == LabFixture::MockTimeout {
        return Err(LabRunnerError::Timeout);
    }
    if request.fixture == LabFixture::MockCorrupt {
        return Err(LabRunnerError::CorruptArtifact);
    }
    if request.fixture != LabFixture::MockEval {
        return Err(LabRunnerError::UnknownFixture);
    }
    let metric = 2_000_000 + (request.seed % 1_000);
    let result = ExperimentResult {
        run_id: request.run_id.clone(),
        spec_id: request.spec_id.clone(),
        commit: request.commit.clone(),
        config: request.config.clone(),
        negative: metric >= 2_000_500,
        summary: format!(
            "fixture={} seed={} metric={} baseline={}",
            request.fixture.as_str(),
            request.seed,
            metric,
            baseline
        ),
    };
    lab.record_result(result.clone())?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lab_registry::{ExperimentRecord, ExperimentStatus, Hypothesis};

    fn primed() -> LabRegistry {
        let mut lab = LabRegistry::new();
        lab.register_hypothesis(Hypothesis {
            id: "h1".into(),
            statement: "mock eval is deterministic".into(),
            issue: "#53".into(),
            pr: "".into(),
            commit: "local".into(),
        });
        lab.register_experiment(ExperimentRecord {
            id: "e1".into(),
            hypothesis_id: "h1".into(),
            baseline: "adam-default".into(),
            primary_metric: "eval_loss_milli".into(),
            budget: "1-fixture".into(),
            success_criterion: "metric<2000500".into(),
            status: ExperimentStatus::Proposed,
        });
        lab
    }

    fn req(run_id: &str, seed: u64, fixture: LabFixture, timeout_ms: u64) -> LabRunRequest {
        LabRunRequest {
            spec_id: "e1".into(),
            run_id: run_id.into(),
            commit: "6ede82f6".into(),
            config: "tiny-mock".into(),
            seed,
            timeout_ms,
            fixture,
        }
    }

    #[test]
    fn same_spec_and_seed_are_equivalent() {
        let mut a = primed();
        let mut b = primed();
        let left = run_fixture(&mut a, &req("r1", 7, LabFixture::MockEval, 10)).unwrap();
        let right = run_fixture(&mut b, &req("r2", 7, LabFixture::MockEval, 10)).unwrap();
        assert_eq!(left.summary, right.summary);
        assert_ne!(left.run_id, right.run_id);
        assert_eq!(left.commit, "6ede82f6");
        assert_eq!(left.config, "tiny-mock");
    }

    #[test]
    fn timeout_and_corrupt_are_not_partial_success() {
        let mut lab = primed();
        assert_eq!(
            run_fixture(&mut lab, &req("t1", 1, LabFixture::MockTimeout, 10)).unwrap_err(),
            LabRunnerError::Timeout
        );
        assert_eq!(
            run_fixture(&mut lab, &req("c1", 1, LabFixture::MockCorrupt, 10)).unwrap_err(),
            LabRunnerError::CorruptArtifact
        );
        assert!(lab.results_for("e1").is_empty());
    }

    #[test]
    fn incomplete_spec_is_fail_closed() {
        let mut lab = LabRegistry::new();
        lab.register_experiment(ExperimentRecord {
            id: "e-bad".into(),
            hypothesis_id: "h".into(),
            baseline: "".into(),
            primary_metric: "x".into(),
            budget: "1".into(),
            success_criterion: "y".into(),
            status: ExperimentStatus::Proposed,
        });
        let err = run_fixture(
            &mut lab,
            &LabRunRequest {
                spec_id: "e-bad".into(),
                run_id: "r".into(),
                commit: "local".into(),
                config: "x".into(),
                seed: 1,
                timeout_ms: 10,
                fixture: LabFixture::MockEval,
            },
        )
        .unwrap_err();
        assert!(matches!(err, LabRunnerError::Incomplete(_)));
    }
}
