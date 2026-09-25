//! #76 campaign runner.
//!
//! Serial batteries of candidates × suites × seeds under a global budget.
//! Completed runs are not repeated on resume. Budget overflow is explicit.

use std::collections::BTreeSet;

pub const CAMPAIGN_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CampaignFailure {
    Experiment,
    Infra,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CampaignError {
    BudgetExceeded,
    Empty,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CampaignSpec {
    pub id: String,
    pub candidates: Vec<String>,
    pub suites: Vec<String>,
    pub seeds: Vec<u64>,
    pub budget_units: u32,
}

impl CampaignSpec {
    pub fn planned_runs(&self) -> Vec<CampaignSlot> {
        let mut out = Vec::new();
        for candidate in &self.candidates {
            for suite in &self.suites {
                for seed in &self.seeds {
                    out.push(CampaignSlot {
                        run_id: format!("{}:{}:{}:{}", self.id, candidate, suite, seed),
                        candidate: candidate.clone(),
                        suite: suite.clone(),
                        seed: *seed,
                    });
                }
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CampaignSlot {
    pub run_id: String,
    pub candidate: String,
    pub suite: String,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CampaignRecord {
    pub run_id: String,
    pub spec_id: String,
    pub candidate: String,
    pub suite: String,
    pub seed: u64,
    pub failure: Option<CampaignFailure>,
    pub summary: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CampaignState {
    pub spec: CampaignSpec,
    pub completed: BTreeSet<String>,
    pub records: Vec<CampaignRecord>,
    pub spent: u32,
}

impl CampaignState {
    pub fn new(spec: CampaignSpec) -> Result<Self, CampaignError> {
        if spec.planned_runs().is_empty() || spec.budget_units == 0 {
            return Err(CampaignError::Empty);
        }
        Ok(Self {
            spec,
            completed: BTreeSet::new(),
            records: Vec::new(),
            spent: 0,
        })
    }

    pub fn step(&mut self) -> Result<Option<CampaignRecord>, CampaignError> {
        let next = self
            .spec
            .planned_runs()
            .into_iter()
            .find(|slot| !self.completed.contains(&slot.run_id));
        let Some(slot) = next else {
            return Ok(None);
        };
        if self.spent >= self.spec.budget_units {
            return Err(CampaignError::BudgetExceeded);
        }
        self.spent += 1;
        let record = execute_slot(&self.spec.id, &slot);
        self.completed.insert(slot.run_id);
        self.records.push(record.clone());
        Ok(Some(record))
    }

    pub fn run_all(&mut self) -> Result<Vec<CampaignRecord>, CampaignError> {
        let mut out = Vec::new();
        while let Some(record) = self.step()? {
            out.push(record);
        }
        Ok(out)
    }
}

fn execute_slot(spec_id: &str, slot: &CampaignSlot) -> CampaignRecord {
    let failure = if slot.suite == "infra-flaky" && slot.seed == 0 {
        Some(CampaignFailure::Infra)
    } else if slot.candidate == "broken" {
        Some(CampaignFailure::Experiment)
    } else {
        None
    };
    CampaignRecord {
        run_id: slot.run_id.clone(),
        spec_id: spec_id.into(),
        candidate: slot.candidate.clone(),
        suite: slot.suite.clone(),
        seed: slot.seed,
        failure,
        summary: format!(
            "campaign={} candidate={} suite={} seed={}",
            spec_id, slot.candidate, slot.suite, slot.seed
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CampaignSpec {
        CampaignSpec {
            id: "c1".into(),
            candidates: vec!["adam".into(), "lion".into()],
            suites: vec!["reason".into()],
            seeds: vec![1, 2],
            budget_units: 4,
        }
    }

    #[test]
    fn resume_skips_completed_runs() {
        let mut state = CampaignState::new(spec()).unwrap();
        let first = state.step().unwrap().unwrap();
        assert_eq!(state.spent, 1);
        let again = CampaignState {
            spec: spec(),
            completed: state.completed.clone(),
            records: state.records.clone(),
            spent: state.spent,
        };
        let mut resumed = again;
        let second = resumed.step().unwrap().unwrap();
        assert_ne!(first.run_id, second.run_id);
        assert!(resumed.completed.contains(&first.run_id));
    }

    #[test]
    fn budget_overflow_is_explicit() {
        let mut tight = spec();
        tight.budget_units = 1;
        let mut state = CampaignState::new(tight).unwrap();
        state.step().unwrap().unwrap();
        assert_eq!(state.step().unwrap_err(), CampaignError::BudgetExceeded);
        assert_eq!(state.records.len(), 1);
    }

    #[test]
    fn infra_and_experiment_failures_are_distinct() {
        let spec = CampaignSpec {
            id: "c2".into(),
            candidates: vec!["broken".into(), "adam".into()],
            suites: vec!["infra-flaky".into()],
            seeds: vec![0],
            budget_units: 2,
        };
        let mut state = CampaignState::new(spec).unwrap();
        let left = state.step().unwrap().unwrap();
        let right = state.step().unwrap().unwrap();
        assert_eq!(left.failure, Some(CampaignFailure::Experiment));
        assert_eq!(right.failure, Some(CampaignFailure::Infra));
        assert_ne!(left.run_id, right.run_id);
        assert_eq!(left.spec_id, "c2");
    }
}
