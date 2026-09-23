use crate::code_suite::{
    evaluate_submissions, generate_suite as generate_code_suite,
    intentional_regression_submissions, oracle_submissions, suite_definition as code_suite,
    CodeExecLimits, CodeReport, CodeSuiteProfile, CODE_SUITE_SCHEMA_VERSION, CODE_SUITE_SEED,
};
use crate::eval_registry::{
    DatasetMetadata, EvaluationMetric, EvaluationSuite, EvaluationTask, MetricDirection,
    ResourceLimits, SeedPolicy, SemVer, TaskKind,
};
use crate::experiment::fingerprint_bytes;
use crate::memory_suite::{
    generate_suite as generate_memory_suite, intentional_regression_predictions,
    oracle_predictions as memory_oracle, score_predictions as score_memory,
    suite_definition as memory_suite, MemoryReport, MemorySuiteProfile,
    MEMORY_SUITE_SCHEMA_VERSION, MEMORY_SUITE_SEED,
};
use crate::reasoning_suite::{
    generate_suite as generate_reasoning_suite, intentional_regression_predictions as reasoning_regression,
    oracle_predictions as reasoning_oracle, score_predictions as score_reasoning,
    suite_definition as reasoning_suite, ReasoningProfile, ReasoningReport,
    REASONING_SUITE_SCHEMA_VERSION, REASONING_SUITE_SEED,
};
use std::error::Error;
use std::fmt;

pub const CAPABILITY_SUITE_SCHEMA_VERSION: u32 = 1;
pub const CAPABILITY_SUITE_SEED: u64 = 75_659_918;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityProfile {
    Smoke,
    Full,
}

impl CapabilityProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Full => "full",
        }
    }

    pub fn reasoning(self) -> ReasoningProfile {
        match self {
            Self::Smoke => ReasoningProfile::Smoke,
            Self::Full => ReasoningProfile::Full,
        }
    }

    pub fn memory(self) -> MemorySuiteProfile {
        match self {
            Self::Smoke => MemorySuiteProfile::Smoke,
            Self::Full => MemorySuiteProfile::Full,
        }
    }

    pub fn code(self) -> CodeSuiteProfile {
        match self {
            Self::Smoke => CodeSuiteProfile::Smoke,
            Self::Full => CodeSuiteProfile::Full,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityKind {
    Reasoning,
    Code,
    Memory,
    Agent,
}

impl CapabilityKind {
    pub const ALL: [Self; 4] = [Self::Reasoning, Self::Code, Self::Memory, Self::Agent];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::Code => "code",
            Self::Memory => "memory",
            Self::Agent => "agent",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityStatus {
    Executed,
    Blocked,
}

impl CapabilityStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Executed => "executed",
            Self::Blocked => "blocked",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityScore {
    pub kind: CapabilityKind,
    pub status: CapabilityStatus,
    pub suite_id: String,
    pub suite_version: String,
    pub schema_version: u32,
    pub suite_fingerprint: u64,
    pub cases: usize,
    pub exact_ratio: f64,
    pub note: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityBatteryReport {
    pub profile: CapabilityProfile,
    pub seed: u64,
    pub capabilities: Vec<CapabilityScore>,
}

impl CapabilityBatteryReport {
    pub fn score(&self, kind: CapabilityKind) -> Option<&CapabilityScore> {
        self.capabilities.iter().find(|score| score.kind == kind)
    }

    pub fn executed_exact_ratios(&self) -> Vec<(CapabilityKind, f64)> {
        self.capabilities
            .iter()
            .filter(|score| score.status == CapabilityStatus::Executed)
            .map(|score| (score.kind, score.exact_ratio))
            .collect()
    }

    /// Descriptive only. Never a gate.
    pub fn descriptive_mean_executed(&self) -> f64 {
        let ratios = self.executed_exact_ratios();
        if ratios.is_empty() {
            0.0
        } else {
            ratios.iter().map(|(_, ratio)| *ratio).sum::<f64>() / ratios.len() as f64
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapabilitySuiteError {
    Child(&'static str),
    AgentNotIntegrated,
}

impl fmt::Display for CapabilitySuiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Child(name) => write!(f, "child suite failed: {name}"),
            Self::AgentNotIntegrated => {
                write!(f, "agent capability is reserved until #47 traces/replay exist")
            }
        }
    }
}

impl Error for CapabilitySuiteError {}

pub fn suite_definition(profile: CapabilityProfile) -> EvaluationSuite {
    let reasoning = reasoning_suite(profile.reasoning(), REASONING_SUITE_SEED);
    let code = code_suite(profile.code(), CODE_SUITE_SEED);
    let memory = memory_suite(profile.memory(), MEMORY_SUITE_SEED);
    let sample_count = reasoning.dataset.sample_count
        + code.dataset.sample_count
        + memory.dataset.sample_count;
    let fingerprint = fingerprint_bytes(
        format!(
            "capability|reasoning={:016x}|code={:016x}|memory={:016x}|agent=blocked-47\n",
            reasoning.dataset.fixture_fingerprint,
            code.dataset.fixture_fingerprint,
            memory.dataset.fixture_fingerprint
        )
        .as_bytes(),
    );

    EvaluationSuite {
        id: format!("capability-battery-{}", profile.as_str()),
        version: SemVer::new(1, 0, 0),
        description: format!(
            "modular #75 battery over reasoning/code/memory {}; agent reserved until #47",
            profile.as_str()
        ),
        dataset: DatasetMetadata {
            id: "auralis-capability-battery".to_string(),
            revision: "1".to_string(),
            split: profile.as_str().to_string(),
            population: "composed child suites; agent profile blocked".to_string(),
            fixture_fingerprint: fingerprint,
            sample_count,
        },
        tasks: vec![
            child_task("reasoning", TaskKind::Reasoning, &reasoning),
            child_task("code", TaskKind::Code, &code),
            child_task("memory", TaskKind::Memory, &memory),
            EvaluationTask {
                id: "capability.agent.reserved".to_string(),
                version: SemVer::new(1, 0, 0),
                kind: TaskKind::Agent,
                description: "reserved until #47 trace/replay/agent suite exists".to_string(),
                fixture_fingerprint: fingerprint_bytes(b"agent-reserved-issue-47"),
                fixture_count: 1,
            },
        ],
        metrics: battery_metrics(),
        seed_policy: SeedPolicy {
            seeds: vec![
                CAPABILITY_SUITE_SEED,
                REASONING_SUITE_SEED,
                CODE_SUITE_SEED,
                MEMORY_SUITE_SEED,
            ],
        },
        limits: ResourceLimits {
            max_examples: sample_count,
            max_steps: 1,
            max_tokens: reasoning.limits.max_tokens
                + code.limits.max_tokens
                + memory.limits.max_tokens,
            max_wall_ms: reasoning.limits.max_wall_ms
                + code.limits.max_wall_ms
                + memory.limits.max_wall_ms,
        },
    }
}

fn child_task(id: &str, kind: TaskKind, suite: &EvaluationSuite) -> EvaluationTask {
    EvaluationTask {
        id: format!("capability.{id}"),
        version: suite.version,
        kind,
        description: suite.description.clone(),
        fixture_fingerprint: suite.dataset.fixture_fingerprint,
        fixture_count: suite.dataset.sample_count,
    }
}

fn battery_metrics() -> Vec<EvaluationMetric> {
    [
        ("reasoning.exact-pass", "ratio", MetricDirection::HigherIsBetter),
        ("code.exact-pass", "ratio", MetricDirection::HigherIsBetter),
        ("memory.exact-pass", "ratio", MetricDirection::HigherIsBetter),
        ("agent.status", "enum", MetricDirection::ExactTarget),
        ("executed-capabilities", "count", MetricDirection::ExactTarget),
        ("descriptive-mean-executed", "ratio", MetricDirection::HigherIsBetter),
    ]
    .into_iter()
    .map(|(id, unit, direction)| EvaluationMetric {
        id: id.to_string(),
        version: SemVer::new(1, 0, 0),
        unit: unit.to_string(),
        direction,
        description: format!("#75 battery metric {id}"),
    })
    .collect()
}

pub fn evaluate_oracle(
    profile: CapabilityProfile,
) -> Result<CapabilityBatteryReport, CapabilitySuiteError> {
    evaluate_with(profile, CapabilityMode::Oracle)
}

pub fn evaluate_isolated_regression(
    profile: CapabilityProfile,
    broken: CapabilityKind,
) -> Result<CapabilityBatteryReport, CapabilitySuiteError> {
    if broken == CapabilityKind::Agent {
        return Err(CapabilitySuiteError::AgentNotIntegrated);
    }
    evaluate_with(profile, CapabilityMode::Regress(broken))
}

#[derive(Clone, Copy)]
enum CapabilityMode {
    Oracle,
    Regress(CapabilityKind),
}

fn evaluate_with(
    profile: CapabilityProfile,
    mode: CapabilityMode,
) -> Result<CapabilityBatteryReport, CapabilitySuiteError> {
    let reasoning = run_reasoning(profile, matches!(mode, CapabilityMode::Regress(CapabilityKind::Reasoning)))?;
    let code = run_code(profile, matches!(mode, CapabilityMode::Regress(CapabilityKind::Code)))?;
    let memory = run_memory(profile, matches!(mode, CapabilityMode::Regress(CapabilityKind::Memory)))?;

    Ok(CapabilityBatteryReport {
        profile,
        seed: CAPABILITY_SUITE_SEED,
        capabilities: vec![
            reasoning,
            code,
            memory,
            CapabilityScore {
                kind: CapabilityKind::Agent,
                status: CapabilityStatus::Blocked,
                suite_id: "agent-reserved".to_string(),
                suite_version: "0.0.0".to_string(),
                schema_version: 0,
                suite_fingerprint: 0,
                cases: 0,
                exact_ratio: 0.0,
                note: "#47 traces/replay not integrated; agent is not scored".to_string(),
            },
        ],
    })
}

fn run_reasoning(
    profile: CapabilityProfile,
    regress: bool,
) -> Result<CapabilityScore, CapabilitySuiteError> {
    let suite = reasoning_suite(profile.reasoning(), REASONING_SUITE_SEED);
    suite.validate().map_err(|_| CapabilitySuiteError::Child("reasoning"))?;
    let cases = generate_reasoning_suite(profile.reasoning(), REASONING_SUITE_SEED);
    let predictions = if regress {
        reasoning_regression(&cases)
    } else {
        reasoning_oracle(&cases)
    };
    let report: ReasoningReport =
        score_reasoning(&cases, &predictions).map_err(|_| CapabilitySuiteError::Child("reasoning"))?;
    Ok(child_score(
        CapabilityKind::Reasoning,
        &suite,
        REASONING_SUITE_SCHEMA_VERSION,
        cases.len(),
        report.exact_match_ratio(),
    ))
}

fn run_code(
    profile: CapabilityProfile,
    regress: bool,
) -> Result<CapabilityScore, CapabilitySuiteError> {
    let suite = code_suite(profile.code(), CODE_SUITE_SEED);
    suite.validate().map_err(|_| CapabilitySuiteError::Child("code"))?;
    let cases = generate_code_suite(profile.code(), CODE_SUITE_SEED);
    let limits = CodeExecLimits::for_profile(profile.code());
    let submissions = if regress {
        intentional_regression_submissions(&cases)
    } else {
        oracle_submissions(&cases)
    };
    let report: CodeReport = evaluate_submissions(&cases, &submissions, limits)
        .map_err(|_| CapabilitySuiteError::Child("code"))?;
    Ok(child_score(
        CapabilityKind::Code,
        &suite,
        CODE_SUITE_SCHEMA_VERSION,
        cases.len(),
        report.exact_ratio(),
    ))
}

fn run_memory(
    profile: CapabilityProfile,
    regress: bool,
) -> Result<CapabilityScore, CapabilitySuiteError> {
    let suite = memory_suite(profile.memory(), MEMORY_SUITE_SEED);
    suite.validate().map_err(|_| CapabilitySuiteError::Child("memory"))?;
    let cases = generate_memory_suite(profile.memory(), MEMORY_SUITE_SEED);
    let predictions = if regress {
        intentional_regression_predictions(&cases)
    } else {
        memory_oracle(&cases).map_err(|_| CapabilitySuiteError::Child("memory"))?
    };
    let report: MemoryReport =
        score_memory(&cases, &predictions).map_err(|_| CapabilitySuiteError::Child("memory"))?;
    Ok(child_score(
        CapabilityKind::Memory,
        &suite,
        MEMORY_SUITE_SCHEMA_VERSION,
        cases.len(),
        report.exact_match_ratio(),
    ))
}

fn child_score(
    kind: CapabilityKind,
    suite: &EvaluationSuite,
    schema_version: u32,
    cases: usize,
    exact_ratio: f64,
) -> CapabilityScore {
    CapabilityScore {
        kind,
        status: CapabilityStatus::Executed,
        suite_id: suite.id.clone(),
        suite_version: suite.version.to_string(),
        schema_version,
        suite_fingerprint: suite.dataset.fixture_fingerprint,
        cases,
        exact_ratio,
        note: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn suite_definition_is_valid_and_keeps_agent_reserved() {
        let suite = suite_definition(CapabilityProfile::Smoke);
        suite.validate().unwrap();
        assert_eq!(suite.tasks.len(), 4);
        assert_eq!(suite.metrics.len(), 6);
        assert_eq!(suite.tasks[3].kind, TaskKind::Agent);
        assert_eq!(suite.tasks[3].fixture_count, 1);
        assert!(suite.seed_policy.seeds.contains(&CAPABILITY_SUITE_SEED));
    }

    #[test]
    fn smoke_oracle_executes_three_capabilities_and_blocks_agent() {
        let report = evaluate_oracle(CapabilityProfile::Smoke).unwrap();
        let reasoning = report.score(CapabilityKind::Reasoning).unwrap();
        let code = report.score(CapabilityKind::Code).unwrap();
        let memory = report.score(CapabilityKind::Memory).unwrap();
        let agent = report.score(CapabilityKind::Agent).unwrap();
        assert_eq!(reasoning.status, CapabilityStatus::Executed);
        assert_eq!(code.status, CapabilityStatus::Executed);
        assert_eq!(memory.status, CapabilityStatus::Executed);
        assert_eq!(agent.status, CapabilityStatus::Blocked);
        assert_eq!(reasoning.exact_ratio, 1.0);
        assert_eq!(code.exact_ratio, 1.0);
        assert_eq!(memory.exact_ratio, 1.0);
        assert_eq!(agent.exact_ratio, 0.0);
        assert_eq!(report.executed_exact_ratios().len(), 3);
    }

    #[test]
    fn isolated_reasoning_regression_does_not_hide_behind_other_capabilities() {
        let report =
            evaluate_isolated_regression(CapabilityProfile::Smoke, CapabilityKind::Reasoning)
                .unwrap();
        assert!(
            report.score(CapabilityKind::Reasoning).unwrap().exact_ratio < 1.0
        );
        assert_eq!(report.score(CapabilityKind::Code).unwrap().exact_ratio, 1.0);
        assert_eq!(
            report.score(CapabilityKind::Memory).unwrap().exact_ratio,
            1.0
        );
        assert_eq!(
            report.score(CapabilityKind::Agent).unwrap().status,
            CapabilityStatus::Blocked
        );
    }

    #[test]
    fn agent_regression_is_fail_closed() {
        let err = evaluate_isolated_regression(CapabilityProfile::Smoke, CapabilityKind::Agent)
            .unwrap_err();
        assert_eq!(err, CapabilitySuiteError::AgentNotIntegrated);
    }

    #[test]
    fn docs_and_bench_match_contract() {
        let text = fs::read_to_string(root().join("docs/capability-suite.md")).unwrap();
        for needle in [
            "CAPABILITY_SUITE_SCHEMA_VERSION = 1",
            "CAPABILITY_SUITE_SEED = 75659918",
            "evaluate_isolated_regression",
            "descriptive-mean-executed",
            "auralis_capability_suite_bench",
            "reserved / blocked until #47",
        ] {
            assert!(text.contains(needle), "capability-suite.md missing {needle}");
        }
        let readme = fs::read_to_string(root().join("README.md")).unwrap();
        assert!(readme.contains("docs/capability-suite.md"));
        let versions = fs::read_to_string(root().join("docs/versions.md")).unwrap();
        assert!(versions.contains("CAPABILITY_SUITE_SCHEMA_VERSION = 1"));
        let registry = fs::read_to_string(root().join("docs/bench-registry.md")).unwrap();
        assert!(registry.contains("auralis_capability_suite_bench"));
        let bench = crate::bench::find("capability-suite").expect("capability-suite benchmark");
        assert_eq!(bench.bin, "auralis_capability_suite_bench");
        assert_eq!(bench.default_args, "smoke");
    }
}
