//! #115 local campaign scheduler.
//!
//! Persistent queue over #76 slots. Completed RunSpecs are not repeated.
//! Budget overflow stops scheduling explicitly.

use crate::campaign::{CampaignRecord, CampaignSpec, CampaignState};
use std::collections::BTreeMap;

pub const CAMPAIGN_SCHEDULER_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Running,
    Blocked,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Blocked => "blocked",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchedulerError {
    BudgetExceeded,
    Paused,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulerJob {
    pub run_id: String,
    pub spec_id: String,
    pub priority: u32,
    pub status: JobStatus,
    pub depends_on: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CampaignScheduler {
    pub schema_version: u32,
    pub concurrency: u32,
    pub budget_units: u32,
    pub spent: u32,
    pub paused: bool,
    pub jobs: BTreeMap<String, SchedulerJob>,
    pub journal: Vec<String>,
    state: CampaignState,
}

impl CampaignScheduler {
    pub fn new(spec: CampaignSpec, concurrency: u32) -> Result<Self, SchedulerError> {
        let budget_units = spec.budget_units;
        let mut jobs = BTreeMap::new();
        let planned = spec.planned_runs();
        for (idx, slot) in planned.iter().enumerate() {
            jobs.insert(
                slot.run_id.clone(),
                SchedulerJob {
                    run_id: slot.run_id.clone(),
                    spec_id: spec.id.clone(),
                    priority: idx as u32,
                    status: JobStatus::Queued,
                    depends_on: None,
                },
            );
        }
        Ok(Self {
            schema_version: CAMPAIGN_SCHEDULER_SCHEMA_VERSION,
            concurrency: concurrency.max(1),
            budget_units,
            spent: 0,
            paused: false,
            jobs,
            journal: vec!["init".into()],
            state: CampaignState::new(spec).map_err(|_| SchedulerError::BudgetExceeded)?,
        })
    }

    pub fn pause(&mut self) {
        self.paused = true;
        self.journal.push("pause".into());
    }

    pub fn resume(&mut self) {
        self.paused = false;
        self.journal.push("resume".into());
    }

    pub fn persist(&self) -> String {
        let mut lines = vec![format!(
            "schema={} paused={} spent={} budget={}",
            self.schema_version, self.paused, self.spent, self.budget_units
        )];
        for job in self.jobs.values() {
            lines.push(format!(
                "job {} {} {}",
                job.run_id,
                job.status.as_str(),
                job.spec_id
            ));
        }
        lines.join("\n")
    }

    pub fn restore_completed(&mut self, snapshot: &str) {
        for line in snapshot.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.first() == Some(&"job") && parts.get(2) == Some(&"done") {
                if let Some(id) = parts.get(1) {
                    if let Some(job) = self.jobs.get_mut(*id) {
                        job.status = JobStatus::Done;
                        self.state.completed.insert((*id).into());
                    }
                }
            }
        }
        self.journal.push("restore".into());
    }

    pub fn tick(&mut self) -> Result<Option<CampaignRecord>, SchedulerError> {
        if self.paused {
            return Err(SchedulerError::Paused);
        }
        let running = self
            .jobs
            .values()
            .filter(|job| job.status == JobStatus::Running)
            .count() as u32;
        if running >= self.concurrency {
            self.journal.push("blocked-concurrency".into());
            return Ok(None);
        }
        let next_id = self
            .jobs
            .values()
            .filter(|job| job.status == JobStatus::Queued)
            .min_by_key(|job| job.priority)
            .map(|job| job.run_id.clone());
        let Some(run_id) = next_id else {
            return Ok(None);
        };
        if self.state.completed.contains(&run_id) {
            if let Some(job) = self.jobs.get_mut(&run_id) {
                job.status = JobStatus::Done;
            }
            self.journal.push(format!("dedupe {run_id}"));
            return self.tick();
        }
        if self.spent >= self.budget_units {
            self.journal.push("budget-stop".into());
            return Err(SchedulerError::BudgetExceeded);
        }
        if let Some(job) = self.jobs.get_mut(&run_id) {
            job.status = JobStatus::Running;
        }
        match self.state.step() {
            Ok(Some(record)) => {
                self.spent += 1;
                let status = if record.failure.is_some() {
                    JobStatus::Failed
                } else {
                    JobStatus::Done
                };
                if let Some(job) = self.jobs.get_mut(&record.run_id) {
                    job.status = status;
                }
                self.journal.push(format!("finish {} {}", record.run_id, status.as_str()));
                Ok(Some(record))
            }
            Ok(None) => Ok(None),
            Err(_) => Err(SchedulerError::BudgetExceeded),
        }
    }

    pub fn reclaim_orphans(&mut self) {
        for job in self.jobs.values_mut() {
            if job.status == JobStatus::Running {
                job.status = JobStatus::Queued;
                self.journal.push(format!("reclaim {}", job.run_id));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(budget: u32) -> CampaignSpec {
        CampaignSpec {
            id: "camp".into(),
            candidates: vec!["adam".into(), "lion".into()],
            suites: vec!["reason".into()],
            seeds: vec![1],
            budget_units: budget,
        }
    }

    #[test]
    fn persist_resume_does_not_repeat_done_runs() {
        let mut sched = CampaignScheduler::new(spec(2), 1).unwrap();
        let first = sched.tick().unwrap().unwrap();
        let snapshot = sched.persist();
        let mut restored = CampaignScheduler::new(spec(2), 1).unwrap();
        restored.restore_completed(&snapshot);
        let second = restored.tick().unwrap().unwrap();
        assert_ne!(first.run_id, second.run_id);
        assert!(snapshot.contains(&first.run_id));
        assert!(restored.journal.iter().any(|line| line == "restore"));
    }

    #[test]
    fn budget_stop_is_explicit_and_pause_blocks() {
        let mut sched = CampaignScheduler::new(spec(1), 1).unwrap();
        sched.tick().unwrap().unwrap();
        assert_eq!(sched.tick().unwrap_err(), SchedulerError::BudgetExceeded);
        sched.pause();
        assert_eq!(sched.tick().unwrap_err(), SchedulerError::Paused);
        assert!(sched.journal.iter().any(|line| line == "budget-stop"));
    }

    #[test]
    fn orphans_return_to_queue() {
        let mut sched = CampaignScheduler::new(spec(2), 1).unwrap();
        let id = sched.jobs.keys().next().unwrap().clone();
        sched.jobs.get_mut(&id).unwrap().status = JobStatus::Running;
        sched.reclaim_orphans();
        assert_eq!(sched.jobs[&id].status, JobStatus::Queued);
    }
}
