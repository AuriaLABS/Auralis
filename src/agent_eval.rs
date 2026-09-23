//! #47 agent evaluation: versioned traces, mock replay, objective suite.
//!
//! Replay never performs real tool effects. Recorded results or in-process
//! mocks are the only backends. Secrets are redacted before a trace is kept.

use crate::eval_registry::{
    DatasetMetadata, EvaluationMetric, EvaluationSuite, EvaluationTask, MetricDirection,
    ResourceLimits, SeedPolicy, SemVer, TaskKind,
};
use crate::experiment::fingerprint_bytes;
use std::error::Error;
use std::fmt;

pub const AGENT_EVAL_SCHEMA_VERSION: u32 = 1;
pub const AGENT_EVAL_SEED: u64 = 47_659_918;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentEvalProfile {
    Smoke,
    Full,
}

impl AgentEvalProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Full => "full",
        }
    }

    pub fn case_count(self) -> usize {
        match self {
            Self::Smoke => 4,
            Self::Full => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentTaskKind {
    LookupThenAnswer,
    PermissionDenied,
    ToolFailure,
    ReasoningFailure,
}

impl AgentTaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LookupThenAnswer => "lookup-then-answer",
            Self::PermissionDenied => "permission-denied",
            Self::ToolFailure => "tool-failure",
            Self::ReasoningFailure => "reasoning-failure",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureClass {
    None,
    Tool,
    Reasoning,
    Permission,
}

impl FailureClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Tool => "tool",
            Self::Reasoning => "reasoning",
            Self::Permission => "permission",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceKind {
    Message,
    Plan,
    ToolCall,
    ToolResult,
    Error,
    Timing,
}

impl TraceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::Plan => "plan",
            Self::ToolCall => "tool-call",
            Self::ToolResult => "tool-result",
            Self::Error => "error",
            Self::Timing => "timing",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEvent {
    pub step_id: String,
    pub call_id: String,
    pub kind: TraceKind,
    pub payload: String,
    pub elapsed_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionTrace {
    pub session_id: String,
    pub schema_version: u32,
    pub commit: String,
    pub config: String,
    pub model: String,
    pub checkpoint: String,
    pub events: Vec<TraceEvent>,
}

impl SessionTrace {
    pub fn canonical(&self) -> String {
        let mut out = format!(
            "session={}|schema={}|commit={}|config={}|model={}|checkpoint={}\n",
            trace_escape(&self.session_id),
            self.schema_version,
            trace_escape(&self.commit),
            trace_escape(&self.config),
            trace_escape(&self.model),
            trace_escape(&self.checkpoint)
        );
        for event in &self.events {
            out.push_str(&format!(
                "{}|{}|{}|{}|{}\n",
                trace_escape(&event.step_id),
                trace_escape(&event.call_id),
                event.kind.as_str(),
                trace_escape(&event.payload),
                event.elapsed_ms
            ));
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentCase {
    pub id: String,
    pub kind: AgentTaskKind,
    pub goal: String,
    pub expected_answer: String,
    pub expected_failure: FailureClass,
    pub allowed_tools: Vec<&'static str>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentCaseScore {
    pub exact: bool,
    pub steps: usize,
    pub tool_calls: usize,
    pub latency_ms: u64,
    pub failure: FailureClass,
    pub redacted: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentReport {
    pub cases: usize,
    pub exact: usize,
    pub tool_failures: usize,
    pub reasoning_failures: usize,
    pub permission_failures: usize,
    pub steps: usize,
    pub tool_calls: usize,
    pub latency_ms: u64,
}

impl AgentReport {
    pub fn exact_ratio(&self) -> f64 {
        if self.cases == 0 {
            0.0
        } else {
            self.exact as f64 / self.cases as f64
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentEvalError {
    ReplayEffectsDisabled,
    SecretInTrace,
    UnknownCase,
}

impl fmt::Display for AgentEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReplayEffectsDisabled => write!(f, "replay refuses real tool effects"),
            Self::SecretInTrace => write!(f, "trace contained an unredacted secret"),
            Self::UnknownCase => write!(f, "unknown agent case"),
        }
    }
}

impl Error for AgentEvalError {}

pub fn redact(value: &str) -> String {
    let mut out = value.to_string();
    for marker in ["sk-live-", "password=", "token="] {
        while let Some(start) = out.find(marker) {
            let tail = &out[start..];
            let end = tail
                .char_indices()
                .find_map(|(offset, ch)| {
                    if offset > 0 && ch.is_whitespace() {
                        Some(start + offset)
                    } else {
                        None
                    }
                })
                .unwrap_or(out.len());
            out.replace_range(start..end, "[REDACTED]");
        }
    }
    out.replace("SECRET", "[REDACTED]")
}

fn trace_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn goal_key(goal: &str, fallback: &str) -> String {
    goal.split_whitespace()
        .find(|w| w.starts_with("city-") || w.starts_with("missing-") || w.starts_with("record-"))
        .unwrap_or(fallback)
        .to_string()
}

pub fn generate_suite(profile: AgentEvalProfile, seed: u64) -> Vec<AgentCase> {
    let kinds = [
        AgentTaskKind::LookupThenAnswer,
        AgentTaskKind::PermissionDenied,
        AgentTaskKind::ToolFailure,
        AgentTaskKind::ReasoningFailure,
    ];
    (0..profile.case_count())
        .map(|i| case_for(kinds[i % kinds.len()], seed, i as u64))
        .collect()
}

fn case_for(kind: AgentTaskKind, seed: u64, index: u64) -> AgentCase {
    let id = format!("agent-{seed}-{index}-{}", kind.as_str());
    match kind {
        AgentTaskKind::LookupThenAnswer => AgentCase {
            id,
            kind,
            goal: format!("lookup city-{index} population"),
            expected_answer: format!("{}", 1000 + index),
            expected_failure: FailureClass::None,
            allowed_tools: vec!["lookup"],
        },
        AgentTaskKind::PermissionDenied => AgentCase {
            id,
            kind,
            goal: format!("delete record-{index}"),
            expected_answer: "denied".to_string(),
            expected_failure: FailureClass::Permission,
            allowed_tools: vec!["lookup"],
        },
        AgentTaskKind::ToolFailure => AgentCase {
            id,
            kind,
            goal: format!("lookup missing-{index}"),
            expected_answer: "tool-error".to_string(),
            expected_failure: FailureClass::Tool,
            allowed_tools: vec!["lookup"],
        },
        AgentTaskKind::ReasoningFailure => AgentCase {
            id,
            kind,
            goal: format!("sum lookup city-{index} without using the number"),
            expected_answer: format!("{}", 1000 + index),
            expected_failure: FailureClass::Reasoning,
            allowed_tools: vec!["lookup"],
        },
    }
}

pub fn mock_lookup(key: &str) -> Result<String, FailureClass> {
    if key.starts_with("missing-") {
        return Err(FailureClass::Tool);
    }
    if key.starts_with("city-") {
        let idx = key.trim_start_matches("city-").parse::<u64>().unwrap_or(0);
        return Ok(format!("{}", 1000 + idx));
    }
    Err(FailureClass::Tool)
}

pub fn replay_case(
    case: &AgentCase,
    allow_real_effects: bool,
) -> Result<(SessionTrace, AgentCaseScore), AgentEvalError> {
    if allow_real_effects {
        return Err(AgentEvalError::ReplayEffectsDisabled);
    }
    let mut events = Vec::new();
    events.push(TraceEvent {
        step_id: format!("{}-s0", case.id),
        call_id: format!("{}-c0", case.id),
        kind: TraceKind::Message,
        payload: redact(&case.goal),
        elapsed_ms: 1,
    });
    events.push(TraceEvent {
        step_id: format!("{}-s1", case.id),
        call_id: format!("{}-c1", case.id),
        kind: TraceKind::Plan,
        payload: format!("use tools={:?}", case.allowed_tools),
        elapsed_ms: 1,
    });

    let (answer, failure, tool_calls) = match case.kind {
        AgentTaskKind::LookupThenAnswer => {
            let key = goal_key(&case.goal, "city-0");
            events.push(tool_call(&case.id, "lookup", &key, 2));
            let value = mock_lookup(&key).unwrap();
            events.push(tool_result(&case.id, &value, 3));
            (value, FailureClass::None, 1)
        }
        AgentTaskKind::PermissionDenied => {
            events.push(TraceEvent {
                step_id: format!("{}-s2", case.id),
                call_id: format!("{}-c2", case.id),
                kind: TraceKind::Error,
                payload: "permission-denied:delete".to_string(),
                elapsed_ms: 1,
            });
            ("denied".to_string(), FailureClass::Permission, 0)
        }
        AgentTaskKind::ToolFailure => {
            let key = goal_key(&case.goal, "missing-0");
            events.push(tool_call(&case.id, "lookup", &key, 2));
            events.push(TraceEvent {
                step_id: format!("{}-s3", case.id),
                call_id: format!("{}-c3", case.id),
                kind: TraceKind::Error,
                payload: "tool-error".to_string(),
                elapsed_ms: 1,
            });
            ("tool-error".to_string(), FailureClass::Tool, 1)
        }
        AgentTaskKind::ReasoningFailure => {
            let key = goal_key(&case.goal, "city-0");
            events.push(tool_call(&case.id, "lookup", &key, 2));
            let value = mock_lookup(&key).unwrap_or_else(|_| "0".into());
            events.push(tool_result(&case.id, &value, 3));
            ("0".to_string(), FailureClass::Reasoning, 1)
        }
    };

    events.push(TraceEvent {
        step_id: format!("{}-sN", case.id),
        call_id: format!("{}-cN", case.id),
        kind: TraceKind::Timing,
        payload: format!("steps={}", events.len() + 1),
        elapsed_ms: events.len() as u64,
    });

    let trace = SessionTrace {
        session_id: case.id.clone(),
        schema_version: AGENT_EVAL_SCHEMA_VERSION,
        commit: "local".to_string(),
        config: "agent-eval-1".to_string(),
        model: "mock".to_string(),
        checkpoint: "none".to_string(),
        events,
    };
    let blob = trace.canonical();
    if blob.contains("SECRET") || blob.contains("sk-live-") || blob.contains("password=") {
        return Err(AgentEvalError::SecretInTrace);
    }
    let exact = failure == case.expected_failure
        && (failure != FailureClass::None || answer == case.expected_answer);
    let steps = trace.events.len();
    let latency_ms: u64 = trace.events.iter().map(|e| e.elapsed_ms).sum();
    Ok((
        trace,
        AgentCaseScore {
            exact,
            steps,
            tool_calls,
            latency_ms,
            failure,
            redacted: true,
        },
    ))
}

fn tool_call(case_id: &str, tool: &str, input: &str, step: u32) -> TraceEvent {
    TraceEvent {
        step_id: format!("{case_id}-s{step}"),
        call_id: format!("{case_id}-c{step}"),
        kind: TraceKind::ToolCall,
        payload: redact(&format!("{tool}({input})")),
        elapsed_ms: 1,
    }
}

fn tool_result(case_id: &str, value: &str, step: u32) -> TraceEvent {
    TraceEvent {
        step_id: format!("{case_id}-s{step}"),
        call_id: format!("{case_id}-c{step}"),
        kind: TraceKind::ToolResult,
        payload: redact(value),
        elapsed_ms: 1,
    }
}

pub fn score_suite(
    cases: &[AgentCase],
    regress: bool,
) -> Result<(Vec<SessionTrace>, AgentReport), AgentEvalError> {
    let mut traces = Vec::new();
    let mut exact = 0;
    let mut tool_failures = 0;
    let mut reasoning_failures = 0;
    let mut permission_failures = 0;
    let mut steps = 0;
    let mut tool_calls = 0;
    let mut latency_ms = 0;
    for case in cases {
        let (trace, mut score) = replay_case(case, false)?;
        if regress {
            score.exact = false;
            if score.failure == FailureClass::None {
                score.failure = FailureClass::Reasoning;
            }
        }
        if score.exact {
            exact += 1;
        }
        match score.failure {
            FailureClass::Tool => tool_failures += 1,
            FailureClass::Reasoning => reasoning_failures += 1,
            FailureClass::Permission => permission_failures += 1,
            FailureClass::None => {}
        }
        steps += score.steps;
        tool_calls += score.tool_calls;
        latency_ms += score.latency_ms;
        traces.push(trace);
    }
    Ok((
        traces,
        AgentReport {
            cases: cases.len(),
            exact,
            tool_failures,
            reasoning_failures,
            permission_failures,
            steps,
            tool_calls,
            latency_ms,
        },
    ))
}

pub fn suite_definition(profile: AgentEvalProfile, seed: u64) -> EvaluationSuite {
    let cases = generate_suite(profile, seed);
    let blob = cases
        .iter()
        .map(|c| format!("{}|{}|{}", c.id, c.kind.as_str(), c.expected_answer))
        .collect::<Vec<_>>()
        .join("\n");
    EvaluationSuite {
        id: format!("agent-eval-{}", profile.as_str()),
        version: SemVer::new(1, 0, 0),
        description: format!(
            "#47 mock-replay agent suite {}; traces redact secrets and refuse real effects",
            profile.as_str()
        ),
        dataset: DatasetMetadata {
            id: "auralis-agent-eval".to_string(),
            revision: "1".to_string(),
            split: profile.as_str().to_string(),
            population: "deterministic mock multi-step agent tasks".to_string(),
            fixture_fingerprint: fingerprint_bytes(blob.as_bytes()),
            sample_count: cases.len(),
        },
        tasks: vec![EvaluationTask {
            id: "agent.eval.replay".to_string(),
            version: SemVer::new(1, 0, 0),
            kind: TaskKind::Agent,
            description: "replay recorded/mocked tool results without real effects".to_string(),
            fixture_fingerprint: fingerprint_bytes(blob.as_bytes()),
            fixture_count: cases.len(),
        }],
        metrics: [
            ("exact-pass", "ratio", MetricDirection::HigherIsBetter),
            ("tool-failures", "count", MetricDirection::ExactTarget),
            ("reasoning-failures", "count", MetricDirection::ExactTarget),
            ("permission-failures", "count", MetricDirection::ExactTarget),
            ("steps", "count", MetricDirection::LowerIsBetter),
            ("tool-calls", "count", MetricDirection::LowerIsBetter),
            ("latency-ms", "ms", MetricDirection::LowerIsBetter),
        ]
        .into_iter()
        .map(|(id, unit, direction)| EvaluationMetric {
            id: id.to_string(),
            version: SemVer::new(1, 0, 0),
            unit: unit.to_string(),
            direction,
            description: format!("#47 agent metric {id}"),
        })
        .collect(),
        seed_policy: SeedPolicy { seeds: vec![seed] },
        limits: ResourceLimits {
            max_examples: cases.len(),
            max_steps: 8,
            max_tokens: 256,
            max_wall_ms: 1_000,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_oracle_is_exact_and_classifies_failures() {
        let cases = generate_suite(AgentEvalProfile::Smoke, AGENT_EVAL_SEED);
        assert_eq!(cases.len(), 4);
        let (traces, report) = score_suite(&cases, false).unwrap();
        assert_eq!(traces.len(), 4);
        assert_eq!(report.exact_ratio(), 1.0);
        assert_eq!(report.tool_failures, 1);
        assert_eq!(report.reasoning_failures, 1);
        assert_eq!(report.permission_failures, 1);
        assert!(report.tool_calls >= 2);
    }

    #[test]
    fn replay_refuses_real_effects() {
        let cases = generate_suite(AgentEvalProfile::Smoke, AGENT_EVAL_SEED);
        let err = replay_case(&cases[0], true).unwrap_err();
        assert_eq!(err, AgentEvalError::ReplayEffectsDisabled);
    }

    #[test]
    fn secrets_are_redacted_without_leaking_values() {
        let redacted = redact(
            "token=abc123 password=hunter2 sk-live-supersecret SECRET visible",
        );
        assert_eq!(
            redacted,
            "[REDACTED] [REDACTED] [REDACTED] [REDACTED] visible"
        );
        for leaked in ["abc123", "hunter2", "supersecret", "SECRET"] {
            assert!(!redacted.contains(leaked), "secret leaked: {leaked}");
        }
    }

    #[test]
    fn canonical_trace_escapes_record_delimiters() {
        let trace = SessionTrace {
            session_id: "s|1".to_string(),
            schema_version: AGENT_EVAL_SCHEMA_VERSION,
            commit: "c\n2".to_string(),
            config: "cfg".to_string(),
            model: "mock".to_string(),
            checkpoint: "none".to_string(),
            events: vec![TraceEvent {
                step_id: "step|1".to_string(),
                call_id: "call\n1".to_string(),
                kind: TraceKind::Message,
                payload: "line1\nline2|tail".to_string(),
                elapsed_ms: 1,
            }],
        };
        let canonical = trace.canonical();
        assert!(canonical.contains("session=s\\|1"));
        assert!(canonical.contains("commit=c\\n2"));
        assert!(canonical.contains("step\\|1|call\\n1|message|line1\\nline2\\|tail|1"));
        assert_eq!(canonical.lines().count(), 2);
    }

    #[test]
    fn regression_drops_exact_pass() {
        let cases = generate_suite(AgentEvalProfile::Smoke, AGENT_EVAL_SEED);
        let (_, report) = score_suite(&cases, true).unwrap();
        assert_eq!(report.exact_ratio(), 0.0);
    }

    #[test]
    fn suite_definition_validates() {
        let suite = suite_definition(AgentEvalProfile::Smoke, AGENT_EVAL_SEED);
        suite.validate().unwrap();
        assert_eq!(suite.tasks[0].kind, TaskKind::Agent);
        assert_eq!(suite.metrics.len(), 7);
    }

    #[test]
    fn traces_are_deterministic() {
        let cases = generate_suite(AgentEvalProfile::Smoke, AGENT_EVAL_SEED);
        let (a, _) = score_suite(&cases, false).unwrap();
        let (b, _) = score_suite(&cases, false).unwrap();
        assert_eq!(a[0].canonical(), b[0].canonical());
    }
}
