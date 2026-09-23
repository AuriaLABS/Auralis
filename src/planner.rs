//! #46 explicit deterministic planner and minimal plan runner.
//!
//! Planning never executes tools. The runner only consumes already structured
//! plan steps through #44 validation, #45 authorization and #49 local tools.
//! Full cancellation/rollback/retry state-machine semantics remain #114.

use crate::agent_permissions::AuthorizedToolRuntime;
use crate::tool_protocol::{
    ToolArgument, ToolErrorSeverity, ToolRequest, ToolResponse, ToolValue,
};
use crate::tool_registry::{ReferenceToolExecutor, ToolRegistry};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const PLAN_SCHEMA_VERSION: u32 = 1;
pub const MAX_PLAN_STEPS: usize = 32;
pub const MAX_PLAN_TOOL_CALLS: usize = 24;
pub const MAX_PLAN_TIMEOUT_MS: u64 = 30_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanBudget {
    pub max_steps: usize,
    pub max_tool_calls: usize,
    pub max_timeout_ms: u64,
}

impl Default for PlanBudget {
    fn default() -> Self {
        Self {
            max_steps: 12,
            max_tool_calls: 8,
            max_timeout_ms: 5_000,
        }
    }
}

impl PlanBudget {
    pub fn validate(self) -> Result<Self, PlannerError> {
        if self.max_steps == 0 || self.max_steps > MAX_PLAN_STEPS {
            return Err(PlannerError::InvalidPlan("max_steps out of range".into()));
        }
        if self.max_tool_calls == 0 || self.max_tool_calls > MAX_PLAN_TOOL_CALLS {
            return Err(PlannerError::InvalidPlan(
                "max_tool_calls out of range".into(),
            ));
        }
        if self.max_timeout_ms == 0 || self.max_timeout_ms > MAX_PLAN_TIMEOUT_MS {
            return Err(PlannerError::InvalidPlan(
                "max_timeout_ms out of range".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanStepStatus {
    Pending,
    Done,
    Failed,
}

impl PlanStepStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanAction {
    Tool(ToolRequest),
    FinishFromStep(String),
    FinishLiteral(String),
}

impl PlanAction {
    fn kind(&self) -> &'static str {
        match self {
            Self::Tool(_) => "tool",
            Self::FinishFromStep(_) => "finish-from-step",
            Self::FinishLiteral(_) => "finish-literal",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanStep {
    pub id: String,
    pub depends_on: Vec<String>,
    pub completion: String,
    pub status: PlanStepStatus,
    pub action: PlanAction,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub schema_version: u32,
    pub plan_id: String,
    pub revision: u32,
    pub goal: String,
    pub budget: PlanBudget,
    pub steps: Vec<PlanStep>,
}

impl Plan {
    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.schema_version != PLAN_SCHEMA_VERSION {
            return Err(PlannerError::InvalidPlan(format!(
                "plan schema {} unsupported",
                self.schema_version
            )));
        }
        self.budget.validate()?;
        if self.plan_id.is_empty() || self.plan_id.len() > 96 {
            return Err(PlannerError::InvalidPlan("invalid plan_id".into()));
        }
        if self.goal.trim().is_empty() || self.goal.len() > 1024 {
            return Err(PlannerError::InvalidPlan("invalid goal".into()));
        }
        if self.steps.is_empty() || self.steps.len() > self.budget.max_steps {
            return Err(PlannerError::InvalidPlan(
                "step count exceeds plan budget".into(),
            ));
        }

        let mut seen = BTreeSet::new();
        let mut tool_calls = 0usize;
        for step in &self.steps {
            if step.id.is_empty() || step.id.len() > 96 || !seen.insert(step.id.clone()) {
                return Err(PlannerError::InvalidPlan(format!(
                    "invalid/duplicate step id {}",
                    step.id
                )));
            }
            if step.completion.trim().is_empty() {
                return Err(PlannerError::InvalidPlan(format!(
                    "step {} has empty completion criterion",
                    step.id
                )));
            }
            for dep in &step.depends_on {
                if !seen.contains(dep) {
                    return Err(PlannerError::InvalidPlan(format!(
                        "step {} dependency {} must reference an earlier step",
                        step.id, dep
                    )));
                }
            }
            match &step.action {
                PlanAction::Tool(request) => {
                    request
                        .validate_envelope()
                        .map_err(|e| PlannerError::InvalidPlan(e.to_string()))?;
                    tool_calls += 1;
                }
                PlanAction::FinishFromStep(source) => {
                    if !seen.contains(source) {
                        return Err(PlannerError::InvalidPlan(format!(
                            "finish source {source} must reference an earlier step"
                        )));
                    }
                }
                PlanAction::FinishLiteral(value) if value.len() > 4096 => {
                    return Err(PlannerError::InvalidPlan(
                        "finish literal exceeds limit".into(),
                    ))
                }
                _ => {}
            }
        }
        if tool_calls > self.budget.max_tool_calls {
            return Err(PlannerError::InvalidPlan(
                "tool calls exceed plan budget".into(),
            ));
        }
        Ok(())
    }

    pub fn canonical(&self) -> Result<String, PlannerError> {
        self.validate()?;
        let mut out = format!(
            "auralis_plan={}\nplan_id={}\nrevision={}\ngoal={}\nmax_steps={}\nmax_tool_calls={}\nmax_timeout_ms={}\nstep_count={}\n",
            self.schema_version,
            esc(&self.plan_id),
            self.revision,
            esc(&self.goal),
            self.budget.max_steps,
            self.budget.max_tool_calls,
            self.budget.max_timeout_ms,
            self.steps.len()
        );
        for (i, step) in self.steps.iter().enumerate() {
            out.push_str(&format!(
                "step.{i}.id={}\nstep.{i}.status={}\nstep.{i}.depends={}\nstep.{i}.completion={}\nstep.{i}.action={}\n",
                esc(&step.id),
                step.status.as_str(),
                step.depends_on.iter().map(|v| esc(v)).collect::<Vec<_>>().join(","),
                esc(&step.completion),
                step.action.kind(),
            ));
            match &step.action {
                PlanAction::Tool(request) => {
                    out.push_str(&format!(
                        "step.{i}.tool_request={}\n",
                        esc(&request.encode())
                    ));
                }
                PlanAction::FinishFromStep(source) => {
                    out.push_str(&format!("step.{i}.source={}\n", esc(source)));
                }
                PlanAction::FinishLiteral(value) => {
                    out.push_str(&format!("step.{i}.value={}\n", esc(value)));
                }
            }
        }
        Ok(out)
    }

    pub fn fingerprint(&self) -> Result<u64, PlannerError> {
        Ok(fingerprint(self.canonical()?.as_bytes()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannerTask {
    Lookup { key: String },
    RecoverThenEcho { text: String },
}

impl PlannerTask {
    pub fn goal(&self) -> String {
        match self {
            Self::Lookup { key } => format!("lookup {key} and return the value"),
            Self::RecoverThenEcho { text } => {
                format!("recover from a controlled transient error and return {text}")
            }
        }
    }

    fn id_suffix(&self) -> String {
        match self {
            Self::Lookup { key } => format!("lookup-{}", safe_id(key)),
            Self::RecoverThenEcho { text } => format!("recover-{}", safe_id(text)),
        }
    }
}

pub trait Planner {
    fn plan(&self, task: &PlannerTask, budget: PlanBudget) -> Result<Plan, PlannerError>;
    fn replan(
        &self,
        task: &PlannerTask,
        plan: &Plan,
        failed_step: &str,
        response: &ToolResponse,
    ) -> Result<Option<Plan>, PlannerError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReferencePlanner;

impl Planner for ReferencePlanner {
    fn plan(&self, task: &PlannerTask, budget: PlanBudget) -> Result<Plan, PlannerError> {
        budget.validate()?;
        let plan_id = format!("plan-{}", task.id_suffix());
        let steps = match task {
            PlannerTask::Lookup { key } => vec![
                tool_step(
                    "s1",
                    vec![],
                    "lookup returns a controlled value",
                    string_request(&plan_id, "s1", "lookup", "key", key),
                ),
                finish_from("s2", vec!["s1"], "answer equals lookup result", "s1"),
            ],
            PlannerTask::RecoverThenEcho { .. } => vec![tool_step(
                "s1",
                vec![],
                "controlled error is observed and triggers replan",
                string_request(&plan_id, "s1", "ref.error", "code", "busy"),
            )],
        };
        let plan = Plan {
            schema_version: PLAN_SCHEMA_VERSION,
            plan_id,
            revision: 1,
            goal: task.goal(),
            budget,
            steps,
        };
        plan.validate()?;
        Ok(plan)
    }

    fn replan(
        &self,
        task: &PlannerTask,
        plan: &Plan,
        failed_step: &str,
        response: &ToolResponse,
    ) -> Result<Option<Plan>, PlannerError> {
        let ToolResponse::Error(error) = response else {
            return Ok(None);
        };
        if error.severity != ToolErrorSeverity::Recoverable {
            return Ok(None);
        }
        match task {
            PlannerTask::RecoverThenEcho { text } if failed_step == "s1" => {
                let mut revised = plan.clone();
                revised.revision = revised.revision.saturating_add(1);
                revised.steps.push(tool_step(
                    "s2",
                    vec![],
                    "fallback echo returns requested text",
                    string_request(&revised.plan_id, "s2", "ref.echo", "text", text),
                ));
                revised
                    .steps
                    .push(finish_from("s3", vec!["s2"], "answer equals fallback result", "s2"));
                revised.validate()?;
                Ok(Some(revised))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannerError {
    InvalidPlan(String),
    Budget(String),
    Dependency(String),
    Tool(String),
    NoReplan,
    NoAnswer,
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(v) => write!(f, "invalid plan: {v}"),
            Self::Budget(v) => write!(f, "plan budget exceeded: {v}"),
            Self::Dependency(v) => write!(f, "plan dependency error: {v}"),
            Self::Tool(v) => write!(f, "plan tool error: {v}"),
            Self::NoReplan => write!(f, "recoverable failure had no valid replan"),
            Self::NoAnswer => write!(f, "plan ended without final answer"),
        }
    }
}

impl std::error::Error for PlannerError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanRunReport {
    pub success: bool,
    pub answer: Option<String>,
    pub plan_revision: u32,
    pub steps_executed: usize,
    pub tool_calls: usize,
    pub recoverable_errors: usize,
    pub replans: usize,
    pub declared_timeout_ms: u64,
    pub plan_fingerprint: u64,
    pub error: Option<String>,
}

impl PlanRunReport {
    pub fn cost_units(&self) -> usize {
        self.steps_executed + self.tool_calls.saturating_mul(4) + self.replans.saturating_mul(2)
    }
}

pub fn run_plan(
    task: &PlannerTask,
    planner: &impl Planner,
    mut plan: Plan,
    registry: &ToolRegistry,
    runtime: &mut AuthorizedToolRuntime,
    executor: &mut ReferenceToolExecutor,
) -> PlanRunReport {
    let initial_fingerprint = plan.fingerprint().unwrap_or(0);
    let mut results = BTreeMap::<String, String>::new();
    let mut cursor = 0usize;
    let mut steps_executed = 0usize;
    let mut tool_calls = 0usize;
    let mut recoverable_errors = 0usize;
    let mut replans = 0usize;
    let mut declared_timeout_ms = 0u64;

    loop {
        if let Err(error) = plan.validate() {
            return failed_report(
                &plan,
                steps_executed,
                tool_calls,
                recoverable_errors,
                replans,
                declared_timeout_ms,
                initial_fingerprint,
                error.to_string(),
            );
        }
        if cursor >= plan.steps.len() {
            return failed_report(
                &plan,
                steps_executed,
                tool_calls,
                recoverable_errors,
                replans,
                declared_timeout_ms,
                initial_fingerprint,
                PlannerError::NoAnswer.to_string(),
            );
        }
        if steps_executed >= plan.budget.max_steps {
            return failed_report(
                &plan,
                steps_executed,
                tool_calls,
                recoverable_errors,
                replans,
                declared_timeout_ms,
                initial_fingerprint,
                PlannerError::Budget("step budget".into()).to_string(),
            );
        }

        let step = plan.steps[cursor].clone();
        if step.status != PlanStepStatus::Pending {
            cursor += 1;
            continue;
        }
        for dep in &step.depends_on {
            let Some(dep_step) = plan.steps.iter().find(|candidate| &candidate.id == dep) else {
                return failed_report(
                    &plan,
                    steps_executed,
                    tool_calls,
                    recoverable_errors,
                    replans,
                    declared_timeout_ms,
                    initial_fingerprint,
                    PlannerError::Dependency(format!("missing {dep}")).to_string(),
                );
            };
            if dep_step.status != PlanStepStatus::Done {
                return failed_report(
                    &plan,
                    steps_executed,
                    tool_calls,
                    recoverable_errors,
                    replans,
                    declared_timeout_ms,
                    initial_fingerprint,
                    PlannerError::Dependency(format!(
                        "{} depends on incomplete {}",
                        step.id, dep
                    ))
                    .to_string(),
                );
            }
        }

        steps_executed += 1;
        match step.action {
            PlanAction::Tool(request) => {
                if tool_calls >= plan.budget.max_tool_calls {
                    return failed_report(
                        &plan,
                        steps_executed,
                        tool_calls,
                        recoverable_errors,
                        replans,
                        declared_timeout_ms,
                        initial_fingerprint,
                        PlannerError::Budget("tool-call budget".into()).to_string(),
                    );
                }
                let Some(definition) = registry.resolve(&request.tool_name, request.tool_version)
                else {
                    return failed_report(
                        &plan,
                        steps_executed,
                        tool_calls,
                        recoverable_errors,
                        replans,
                        declared_timeout_ms,
                        initial_fingerprint,
                        PlannerError::Tool("unknown planned tool".into()).to_string(),
                    );
                };
                declared_timeout_ms = declared_timeout_ms.saturating_add(definition.timeout_ms);
                if declared_timeout_ms > plan.budget.max_timeout_ms {
                    return failed_report(
                        &plan,
                        steps_executed,
                        tool_calls,
                        recoverable_errors,
                        replans,
                        declared_timeout_ms,
                        initial_fingerprint,
                        PlannerError::Budget("timeout budget".into()).to_string(),
                    );
                }

                tool_calls += 1;
                let response = runtime.invoke(&request, false, executor);
                match &response {
                    ToolResponse::Result(result) => {
                        results.insert(step.id.clone(), result.content.clone());
                        mark_status(&mut plan, &step.id, PlanStepStatus::Done);
                        cursor += 1;
                    }
                    ToolResponse::Error(error) => {
                        mark_status(&mut plan, &step.id, PlanStepStatus::Failed);
                        if error.severity == ToolErrorSeverity::Recoverable {
                            recoverable_errors += 1;
                            match planner.replan(task, &plan, &step.id, &response) {
                                Ok(Some(revised)) => {
                                    plan = revised;
                                    replans += 1;
                                    cursor += 1;
                                }
                                Ok(None) => {
                                    return failed_report(
                                        &plan,
                                        steps_executed,
                                        tool_calls,
                                        recoverable_errors,
                                        replans,
                                        declared_timeout_ms,
                                        initial_fingerprint,
                                        PlannerError::NoReplan.to_string(),
                                    )
                                }
                                Err(err) => {
                                    return failed_report(
                                        &plan,
                                        steps_executed,
                                        tool_calls,
                                        recoverable_errors,
                                        replans,
                                        declared_timeout_ms,
                                        initial_fingerprint,
                                        err.to_string(),
                                    )
                                }
                            }
                        } else {
                            return failed_report(
                                &plan,
                                steps_executed,
                                tool_calls,
                                recoverable_errors,
                                replans,
                                declared_timeout_ms,
                                initial_fingerprint,
                                PlannerError::Tool(format!("fatal {}", error.code)).to_string(),
                            );
                        }
                    }
                }
            }
            PlanAction::FinishFromStep(source) => {
                let Some(answer) = results.get(&source).cloned() else {
                    return failed_report(
                        &plan,
                        steps_executed,
                        tool_calls,
                        recoverable_errors,
                        replans,
                        declared_timeout_ms,
                        initial_fingerprint,
                        PlannerError::Dependency(format!("no result for {source}")).to_string(),
                    );
                };
                mark_status(&mut plan, &step.id, PlanStepStatus::Done);
                return PlanRunReport {
                    success: true,
                    answer: Some(answer),
                    plan_revision: plan.revision,
                    steps_executed,
                    tool_calls,
                    recoverable_errors,
                    replans,
                    declared_timeout_ms,
                    plan_fingerprint: plan.fingerprint().unwrap_or(initial_fingerprint),
                    error: None,
                };
            }
            PlanAction::FinishLiteral(answer) => {
                mark_status(&mut plan, &step.id, PlanStepStatus::Done);
                return PlanRunReport {
                    success: true,
                    answer: Some(answer),
                    plan_revision: plan.revision,
                    steps_executed,
                    tool_calls,
                    recoverable_errors,
                    replans,
                    declared_timeout_ms,
                    plan_fingerprint: plan.fingerprint().unwrap_or(initial_fingerprint),
                    error: None,
                };
            }
        }
    }
}

pub fn run_direct(
    task: &PlannerTask,
    registry: &ToolRegistry,
    runtime: &mut AuthorizedToolRuntime,
    executor: &mut ReferenceToolExecutor,
) -> PlanRunReport {
    let budget = PlanBudget::default();
    let request = match task {
        PlannerTask::Lookup { key } => string_request("direct", "s1", "lookup", "key", key),
        PlannerTask::RecoverThenEcho { .. } => {
            string_request("direct", "s1", "ref.error", "code", "busy")
        }
    };
    let Some(definition) = registry.resolve(&request.tool_name, request.tool_version) else {
        return PlanRunReport {
            success: false,
            answer: None,
            plan_revision: 0,
            steps_executed: 1,
            tool_calls: 0,
            recoverable_errors: 0,
            replans: 0,
            declared_timeout_ms: 0,
            plan_fingerprint: 0,
            error: Some("unknown direct tool".into()),
        };
    };
    let response = runtime.invoke(&request, false, executor);
    match response {
        ToolResponse::Result(result) => PlanRunReport {
            success: true,
            answer: Some(result.content),
            plan_revision: 0,
            steps_executed: 1,
            tool_calls: 1,
            recoverable_errors: 0,
            replans: 0,
            declared_timeout_ms: definition.timeout_ms.min(budget.max_timeout_ms),
            plan_fingerprint: 0,
            error: None,
        },
        ToolResponse::Error(error) => PlanRunReport {
            success: false,
            answer: None,
            plan_revision: 0,
            steps_executed: 1,
            tool_calls: 1,
            recoverable_errors: usize::from(error.severity == ToolErrorSeverity::Recoverable),
            replans: 0,
            declared_timeout_ms: definition.timeout_ms.min(budget.max_timeout_ms),
            plan_fingerprint: 0,
            error: Some(error.code),
        },
    }
}

fn tool_step(
    id: &str,
    depends_on: Vec<&str>,
    completion: &str,
    request: ToolRequest,
) -> PlanStep {
    PlanStep {
        id: id.into(),
        depends_on: depends_on.into_iter().map(str::to_string).collect(),
        completion: completion.into(),
        status: PlanStepStatus::Pending,
        action: PlanAction::Tool(request),
    }
}

fn finish_from(id: &str, depends_on: Vec<&str>, completion: &str, source: &str) -> PlanStep {
    PlanStep {
        id: id.into(),
        depends_on: depends_on.into_iter().map(str::to_string).collect(),
        completion: completion.into(),
        status: PlanStepStatus::Pending,
        action: PlanAction::FinishFromStep(source.into()),
    }
}

fn string_request(
    prefix: &str,
    step: &str,
    tool: &str,
    arg: &str,
    value: &str,
) -> ToolRequest {
    ToolRequest::new(
        format!("{prefix}:{step}"),
        tool,
        1,
        vec![ToolArgument {
            name: arg.into(),
            value: ToolValue::String(value.into()),
        }],
    )
}

fn mark_status(plan: &mut Plan, id: &str, status: PlanStepStatus) {
    if let Some(step) = plan.steps.iter_mut().find(|candidate| candidate.id == id) {
        step.status = status;
    }
}

fn failed_report(
    plan: &Plan,
    steps_executed: usize,
    tool_calls: usize,
    recoverable_errors: usize,
    replans: usize,
    declared_timeout_ms: u64,
    fingerprint: u64,
    error: String,
) -> PlanRunReport {
    PlanRunReport {
        success: false,
        answer: None,
        plan_revision: plan.revision,
        steps_executed,
        tool_calls,
        recoverable_errors,
        replans,
        declared_timeout_ms,
        plan_fingerprint: fingerprint,
        error: Some(error),
    }
}

fn safe_id(value: &str) -> String {
    let out = value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>();
    out.chars().take(40).collect()
}

fn esc(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace(',', "\\,")
        .replace('=', "\\=")
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &byte in bytes {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_permissions::PermissionPolicy;

    fn runtime(
        registry: &ToolRegistry,
    ) -> (AuthorizedToolRuntime, ReferenceToolExecutor) {
        let policy = PermissionPolicy::default()
            .allow_tool("lookup", 1)
            .allow_tool("ref.error", 1)
            .allow_tool("ref.echo", 1);
        (
            AuthorizedToolRuntime::new(registry.catalog(), policy).unwrap(),
            registry.reference_executor(),
        )
    }

    #[test]
    fn plan_is_deterministic_inspectable_and_budgeted() {
        let planner = ReferencePlanner;
        let task = PlannerTask::Lookup {
            key: "city-2".into(),
        };
        let a = planner.plan(&task, PlanBudget::default()).unwrap();
        let b = planner.plan(&task, PlanBudget::default()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.fingerprint().unwrap(), b.fingerprint().unwrap());
        let text = a.canonical().unwrap();
        assert!(text.contains("auralis_plan=1"));
        assert!(text.contains("step.0.action=tool"));
        assert!(text.contains("step.1.action=finish-from-step"));
    }

    #[test]
    fn lookup_plan_replays_exactly_with_authorized_reference_tool() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let planner = ReferencePlanner;
        let task = PlannerTask::Lookup {
            key: "city-2".into(),
        };

        let plan = planner.plan(&task, PlanBudget::default()).unwrap();
        let (mut runtime_a, mut executor_a) = runtime(&registry);
        let a = run_plan(
            &task,
            &planner,
            plan.clone(),
            &registry,
            &mut runtime_a,
            &mut executor_a,
        );
        let (mut runtime_b, mut executor_b) = runtime(&registry);
        let b = run_plan(
            &task,
            &planner,
            plan,
            &registry,
            &mut runtime_b,
            &mut executor_b,
        );

        assert!(a.success && b.success);
        assert_eq!(a.answer.as_deref(), Some("1002"));
        assert_eq!(a.answer, b.answer);
        assert_eq!(a.steps_executed, b.steps_executed);
        assert_eq!(a.tool_calls, b.tool_calls);
        assert_eq!(a.replans, 0);
        assert_eq!(runtime_a.audit().len(), 1);
        assert_eq!(executor_a.invocations(), 1);
    }

    #[test]
    fn recoverable_error_stops_then_replans_without_silent_continuation() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let planner = ReferencePlanner;
        let task = PlannerTask::RecoverThenEcho {
            text: "recovered".into(),
        };
        let plan = planner.plan(&task, PlanBudget::default()).unwrap();
        let (mut runtime, mut executor) = runtime(&registry);
        let report = run_plan(
            &task,
            &planner,
            plan,
            &registry,
            &mut runtime,
            &mut executor,
        );

        assert!(report.success);
        assert_eq!(report.answer.as_deref(), Some("recovered"));
        assert_eq!(report.recoverable_errors, 1);
        assert_eq!(report.replans, 1);
        assert_eq!(report.tool_calls, 2);
        assert_eq!(report.plan_revision, 2);
        assert_eq!(runtime.audit().len(), 2);
    }

    #[test]
    fn direct_strategy_does_not_replan_recoverable_error() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let task = PlannerTask::RecoverThenEcho {
            text: "recovered".into(),
        };
        let (mut runtime, mut executor) = runtime(&registry);
        let direct = run_direct(&task, &registry, &mut runtime, &mut executor);
        assert!(!direct.success);
        assert_eq!(direct.recoverable_errors, 1);
        assert_eq!(direct.replans, 0);
        assert_eq!(direct.tool_calls, 1);
    }

    #[test]
    fn tool_and_step_budgets_fail_closed() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let planner = ReferencePlanner;
        let task = PlannerTask::Lookup {
            key: "city-1".into(),
        };
        let budget = PlanBudget {
            max_steps: 2,
            max_tool_calls: 1,
            max_timeout_ms: 50,
        };
        let plan = planner.plan(&task, budget).unwrap();
        let (mut runtime, mut executor) = runtime(&registry);
        let report = run_plan(
            &task,
            &planner,
            plan,
            &registry,
            &mut runtime,
            &mut executor,
        );
        assert!(!report.success);
        assert!(report.error.unwrap().contains("timeout budget"));
        assert_eq!(executor.invocations(), 0);
    }

    #[test]
    fn dependency_failure_is_not_skipped() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let planner = ReferencePlanner;
        let task = PlannerTask::Lookup {
            key: "city-1".into(),
        };
        let mut plan = planner.plan(&task, PlanBudget::default()).unwrap();
        plan.steps[0].status = PlanStepStatus::Failed;
        let (mut runtime, mut executor) = runtime(&registry);
        let report = run_plan(
            &task,
            &planner,
            plan,
            &registry,
            &mut runtime,
            &mut executor,
        );
        assert!(!report.success);
        assert!(report.error.unwrap().contains("incomplete"));
        assert_eq!(executor.invocations(), 0);
    }
}
