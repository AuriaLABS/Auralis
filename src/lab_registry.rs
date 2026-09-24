//! #52 lab registry: hypotheses, experiments and negative results.
//!
//! An experiment cannot enter Running without a baseline and a success
//! criterion. Negative results persist the same way as positive ones.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub const LAB_REGISTRY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExperimentStatus {
    Proposed,
    Running,
    Completed,
    Rejected,
    Inconclusive,
}

impl ExperimentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Rejected => "rejected",
            Self::Inconclusive => "inconclusive",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        match name {
            "proposed" => Some(Self::Proposed),
            "running" => Some(Self::Running),
            "completed" => Some(Self::Completed),
            "rejected" => Some(Self::Rejected),
            "inconclusive" => Some(Self::Inconclusive),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabError {
    MissingBaseline,
    MissingSuccessCriterion,
    Unknown(String),
    IncompatibleSchema(u32),
    Malformed(&'static str),
}

impl fmt::Display for LabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBaseline => write!(f, "experiment has no baseline"),
            Self::MissingSuccessCriterion => write!(f, "experiment has no success criterion"),
            Self::Unknown(id) => write!(f, "unknown lab record {id}"),
            Self::IncompatibleSchema(v) => write!(f, "incompatible lab registry schema {v}"),
            Self::Malformed(msg) => write!(f, "malformed lab registry: {msg}"),
        }
    }
}

impl Error for LabError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hypothesis {
    pub id: String,
    pub statement: String,
    pub issue: String,
    pub pr: String,
    pub commit: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExperimentRecord {
    pub id: String,
    pub hypothesis_id: String,
    pub baseline: String,
    pub primary_metric: String,
    pub budget: String,
    pub success_criterion: String,
    pub status: ExperimentStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExperimentResult {
    pub run_id: String,
    pub spec_id: String,
    pub commit: String,
    pub config: String,
    pub negative: bool,
    pub summary: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabDecision {
    pub id: String,
    pub spec_id: String,
    pub conclusion: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LabRegistry {
    pub schema_version: u32,
    hypotheses: BTreeMap<String, Hypothesis>,
    experiments: BTreeMap<String, ExperimentRecord>,
    results: Vec<ExperimentResult>,
    decisions: Vec<LabDecision>,
}

impl LabRegistry {
    pub fn new() -> Self {
        Self {
            schema_version: LAB_REGISTRY_SCHEMA_VERSION,
            ..Self::default()
        }
    }

    pub fn register_hypothesis(&mut self, hypothesis: Hypothesis) {
        self.hypotheses.insert(hypothesis.id.clone(), hypothesis);
    }

    pub fn register_experiment(&mut self, experiment: ExperimentRecord) {
        self.experiments.insert(experiment.id.clone(), experiment);
    }

    pub fn experiment(&self, id: &str) -> Option<&ExperimentRecord> {
        self.experiments.get(id)
    }

    pub fn start(&mut self, spec_id: &str) -> Result<(), LabError> {
        let experiment = self
            .experiments
            .get_mut(spec_id)
            .ok_or_else(|| LabError::Unknown(spec_id.into()))?;
        if experiment.baseline.trim().is_empty() {
            return Err(LabError::MissingBaseline);
        }
        if experiment.success_criterion.trim().is_empty() {
            return Err(LabError::MissingSuccessCriterion);
        }
        experiment.status = ExperimentStatus::Running;
        Ok(())
    }

    pub fn record_result(&mut self, result: ExperimentResult) -> Result<(), LabError> {
        if !self.experiments.contains_key(&result.spec_id) {
            return Err(LabError::Unknown(result.spec_id.clone()));
        }
        self.results.push(result);
        Ok(())
    }

    pub fn decide(&mut self, decision: LabDecision) -> Result<(), LabError> {
        if !self.experiments.contains_key(&decision.spec_id) {
            return Err(LabError::Unknown(decision.spec_id.clone()));
        }
        self.decisions.push(decision);
        Ok(())
    }

    pub fn results_for(&self, spec_id: &str) -> Vec<&ExperimentResult> {
        self.results.iter().filter(|r| r.spec_id == spec_id).collect()
    }

    pub fn negatives(&self) -> Vec<&ExperimentResult> {
        self.results.iter().filter(|r| r.negative).collect()
    }

    pub fn canonical(&self) -> String {
        let mut out = format!("auralis_lab={}\n", self.schema_version);
        for h in self.hypotheses.values() {
            out.push_str(&format!(
                "H|{}|{}|{}|{}|{}\n",
                escape(&h.id),
                escape(&h.statement),
                escape(&h.issue),
                escape(&h.pr),
                escape(&h.commit)
            ));
        }
        for e in self.experiments.values() {
            out.push_str(&format!(
                "E|{}|{}|{}|{}|{}|{}|{}\n",
                escape(&e.id),
                escape(&e.hypothesis_id),
                escape(&e.baseline),
                escape(&e.primary_metric),
                escape(&e.budget),
                escape(&e.success_criterion),
                e.status.as_str()
            ));
        }
        for r in &self.results {
            out.push_str(&format!(
                "R|{}|{}|{}|{}|{}|{}\n",
                escape(&r.run_id),
                escape(&r.spec_id),
                escape(&r.commit),
                escape(&r.config),
                r.negative as u8,
                escape(&r.summary)
            ));
        }
        for d in &self.decisions {
            out.push_str(&format!(
                "D|{}|{}|{}\n",
                escape(&d.id),
                escape(&d.spec_id),
                escape(&d.conclusion)
            ));
        }
        out
    }

    pub fn load(blob: &str) -> Result<Self, LabError> {
        let mut lines = blob.lines();
        let schema = lines
            .next()
            .and_then(|l| l.strip_prefix("auralis_lab="))
            .and_then(|v| v.parse().ok())
            .ok_or(LabError::Malformed("missing schema"))?;
        if schema != LAB_REGISTRY_SCHEMA_VERSION {
            return Err(LabError::IncompatibleSchema(schema));
        }
        let mut registry = LabRegistry::new();
        for line in lines {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split('|');
            match parts.next() {
                Some("H") => {
                    let id = unescape(parts.next().unwrap_or(""));
                    registry.register_hypothesis(Hypothesis {
                        id,
                        statement: unescape(parts.next().unwrap_or("")),
                        issue: unescape(parts.next().unwrap_or("")),
                        pr: unescape(parts.next().unwrap_or("")),
                        commit: unescape(parts.next().unwrap_or("")),
                    });
                }
                Some("E") => {
                    let id = unescape(parts.next().unwrap_or(""));
                    registry.register_experiment(ExperimentRecord {
                        id,
                        hypothesis_id: unescape(parts.next().unwrap_or("")),
                        baseline: unescape(parts.next().unwrap_or("")),
                        primary_metric: unescape(parts.next().unwrap_or("")),
                        budget: unescape(parts.next().unwrap_or("")),
                        success_criterion: unescape(parts.next().unwrap_or("")),
                        status: ExperimentStatus::parse(parts.next().unwrap_or(""))
                            .ok_or(LabError::Malformed("bad status"))?,
                    });
                }
                Some("R") => {
                    registry.record_result(ExperimentResult {
                        run_id: unescape(parts.next().unwrap_or("")),
                        spec_id: unescape(parts.next().unwrap_or("")),
                        commit: unescape(parts.next().unwrap_or("")),
                        config: unescape(parts.next().unwrap_or("")),
                        negative: parts.next() == Some("1"),
                        summary: unescape(parts.next().unwrap_or("")),
                    })?;
                }
                Some("D") => {
                    registry.decide(LabDecision {
                        id: unescape(parts.next().unwrap_or("")),
                        spec_id: unescape(parts.next().unwrap_or("")),
                        conclusion: unescape(parts.next().unwrap_or("")),
                    })?;
                }
                _ => return Err(LabError::Malformed("unknown record")),
            }
        }
        Ok(registry)
    }
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\n', "\\n").replace('|', "\\|")
}

fn unescape(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('|') => out.push('|'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spec(id: &str, baseline: &str, criterion: &str) -> ExperimentRecord {
        ExperimentRecord {
            id: id.into(),
            hypothesis_id: "h1".into(),
            baseline: baseline.into(),
            primary_metric: "eval_loss".into(),
            budget: "4-steps".into(),
            success_criterion: criterion.into(),
            status: ExperimentStatus::Proposed,
        }
    }

    #[test]
    fn cannot_start_without_baseline_or_criterion() {
        let mut lab = LabRegistry::new();
        lab.register_hypothesis(Hypothesis {
            id: "h1".into(),
            statement: "Lion beats Adam on tiny eval".into(),
            issue: "#52".into(),
            pr: "".into(),
            commit: "local".into(),
        });
        lab.register_experiment(sample_spec("e-empty", "", "loss<2"));
        assert_eq!(lab.start("e-empty"), Err(LabError::MissingBaseline));
        lab.register_experiment(sample_spec("e-no-gate", "adam-default", ""));
        assert_eq!(lab.start("e-no-gate"), Err(LabError::MissingSuccessCriterion));
        lab.register_experiment(sample_spec("e-ok", "adam-default", "loss<2"));
        lab.start("e-ok").unwrap();
        assert_eq!(lab.experiments["e-ok"].status, ExperimentStatus::Running);
    }

    #[test]
    fn negative_results_persist_and_runs_differ_by_id() {
        let mut lab = LabRegistry::new();
        lab.register_experiment(sample_spec("e1", "adam", "loss<2"));
        lab.record_result(ExperimentResult {
            run_id: "r1".into(),
            spec_id: "e1".into(),
            commit: "aaa".into(),
            config: "tiny".into(),
            negative: true,
            summary: "no improvement".into(),
        })
        .unwrap();
        lab.record_result(ExperimentResult {
            run_id: "r2".into(),
            spec_id: "e1".into(),
            commit: "aaa".into(),
            config: "tiny".into(),
            negative: false,
            summary: "repeat".into(),
        })
        .unwrap();
        assert_eq!(lab.results_for("e1").len(), 2);
        assert_eq!(lab.negatives().len(), 1);
        assert_ne!(lab.results[0].run_id, lab.results[1].run_id);
        let loaded = LabRegistry::load(&lab.canonical()).unwrap();
        assert_eq!(loaded.canonical(), lab.canonical());
        assert_eq!(loaded.negatives().len(), 1);
    }

    #[test]
    fn incompatible_schema_is_rejected() {
        let err = LabRegistry::load("auralis_lab=9\n").unwrap_err();
        assert_eq!(err, LabError::IncompatibleSchema(9));
    }
}
