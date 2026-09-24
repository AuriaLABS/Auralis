//! #114 plan executor: explicit state machine, abort and logical rollback.
//!
//! Planning stays in #46. This machine records every admit/reject/invoke/
//! confirm/fail/retry/replan/skip/abort/rollback decision so a replay of the
//! decision log reconstructs the same sequence. Confirmed mutative actions
//! are not repeated unless the policy explicitly allows it.

use crate::agent_eval::{SessionTrace, TraceEvent, TraceKind, AGENT_EVAL_SCHEMA_VERSION};
use crate::agent_permissions::AuthorizedToolRuntime;
use crate::planner::{
    Plan, PlanAction, PlanBudget, PlanStepStatus, Planner, PlannerError, PlannerTask,
};
use crate::tool_protocol::{ToolErrorSeverity, ToolRequest, ToolResponse};
use crate::tool_registry::{ReferenceToolExecutor, ToolRegistry};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const EXECUTOR_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutorStepState {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
    Cancelled,
}

impl ExecutorStepState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Done | Self::Failed | Self::Skipped | Self::Cancelled
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MachineStatus {
    Idle,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl MachineStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionKind {
    Admit,
    RejectDependency,
    RejectBudget,
    RejectMutationReplay,
    InvokeTool,
    Confirm,
    Fail,
    Retry,
    Replan,
    Skip,
    Abort,
    Rollback,
    Finish,
}

impl DecisionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Admit => "admit",
            Self::RejectDependency => "reject-dependency",
            Self::RejectBudget => "reject-budget",
            Self::RejectMutationReplay => "reject-mutation-replay",
            Self::InvokeTool => "invoke-tool",
            Self::Confirm => "confirm",
            Self::Fail => "fail",
            Self::Retry => "retry",
            Self::Replan => "replan",
            Self::Skip => "skip",
            Self::Abort => "abort",
            Self::Rollback => "rollback",
            Self::Finish => "finish",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionRecord {
    pub seq: u32,
    pub step_id: String,
    pub from: ExecutorStepState,
    pub to: ExecutorStepState,
    pub kind: DecisionKind,
    pub detail: String,
}

impl DecisionRecord {
    pub fn canonical(&self) -> String {
        format!(
            "{:04}|{}|{}|{}|{}|{}\n",
            self.seq,
            esc(&self.step_id),
            self.from.as_str(),
            self.to.as_str(),
            self.kind.as_str(),
            esc(&self.detail)
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepLedgerEntry {
    pub step_id: String,
    pub action_fingerprint: u64,
    pub mutative: bool,
    pub confirmed: bool,
    pub invocations: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorPolicy {
    pub max_retries_per_step: u32,
    pub allow_mutative_replay: bool,
    pub abort_after_steps: Option<usize>,
    pub rollback_before_confirm: bool,
    pub mutative_tools: BTreeSet<String>,
}

impl Default for ExecutorPolicy {
    fn default() -> Self {
        Self {
            max_retries_per_step: 1,
            allow_mutative_replay: false,
            abort_after_steps: None,
            rollback_before_confirm: false,
            mutative_tools: BTreeSet::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorReport {
    pub success: bool,
    pub status: MachineStatus,
    pub answer: Option<String>,
    pub plan_revision: u32,
    pub steps_executed: usize,
    pub tool_calls: usize,
    pub retries: usize,
    pub recoverable_errors: usize,
    pub replans: usize,
    pub rollbacks: usize,
    pub skipped: usize,
    pub cancelled: usize,
    pub declared_timeout_ms: u64,
    pub decisions: Vec<DecisionRecord>,
    pub error: Option<String>,
}

impl ExecutorReport {
    pub fn decision_log(&self) -> String {
        let mut out = format!(
            "auralis_executor={}\nstatus={}\nsuccess={}\nrevision={}\n",
            EXECUTOR_SCHEMA_VERSION,
            self.status.as_str(),
            self.success,
            self.plan_revision
        );
        for record in &self.decisions {
            out.push_str(&record.canonical());
        }
        out
    }

    pub fn decision_fingerprint(&self) -> u64 {
        fingerprint(self.decision_log().as_bytes())
    }

    pub fn to_trace(&self, session_id: &str) -> SessionTrace {
        let events = self
            .decisions
            .iter()
            .map(|record| TraceEvent {
                step_id: record.step_id.clone(),
                call_id: format!("d{}", record.seq),
                kind: match record.kind {
                    DecisionKind::InvokeTool => TraceKind::ToolCall,
                    DecisionKind::Confirm | DecisionKind::Finish => TraceKind::ToolResult,
                    DecisionKind::Fail
                    | DecisionKind::RejectDependency
                    | DecisionKind::RejectBudget
                    | DecisionKind::RejectMutationReplay => TraceKind::Error,
                    DecisionKind::Replan | DecisionKind::Admit => TraceKind::Plan,
                    _ => TraceKind::Timing,
                },
                payload: format!(
                    "{}:{}->{}:{}",
                    record.kind.as_str(),
                    record.from.as_str(),
                    record.to.as_str(),
                    record.detail
                ),
                elapsed_ms: 1,
            })
            .collect();
        SessionTrace {
            session_id: session_id.into(),
            schema_version: AGENT_EVAL_SCHEMA_VERSION,
            commit: "local".into(),
            config: format!("executor-{}", EXECUTOR_SCHEMA_VERSION),
            model: "mock".into(),
            checkpoint: "none".into(),
            events,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorError {
    InvalidPlan(String),
    Budget(String),
    Dependency(String),
    MutationReplay(String),
    Aborted,
    NoAnswer,
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(v) => write!(f, "invalid plan: {v}"),
            Self::Budget(v) => write!(f, "executor budget exceeded: {v}"),
            Self::Dependency(v) => write!(f, "executor dependency error: {v}"),
            Self::MutationReplay(v) => write!(f, "mutative replay refused: {v}"),
            Self::Aborted => write!(f, "executor aborted"),
            Self::NoAnswer => write!(f, "executor ended without final answer"),
        }
    }
}

impl std::error::Error for ExecutorError {}

pub fn execute_plan(
    task: &PlannerTask,
    planner: &impl Planner,
    plan: Plan,
    registry: &ToolRegistry,
    runtime: &mut AuthorizedToolRuntime,
    executor: &mut ReferenceToolExecutor,
    policy: ExecutorPolicy,
) -> ExecutorReport {
    PlanMachine::new(plan, policy).run(task, planner, registry, runtime, executor)
}

struct PlanMachine {
    plan: Plan,
    states: BTreeMap<String, ExecutorStepState>,
    ledger: BTreeMap<String, StepLedgerEntry>,
    decisions: Vec<DecisionRecord>,
    retries: BTreeMap<String, u32>,
    policy: ExecutorPolicy,
    status: MachineStatus,
    abort_requested: bool,
    steps_executed: usize,
    tool_calls: usize,
    retry_count: usize,
    recoverable_errors: usize,
    replans: usize,
    rollbacks: usize,
    declared_timeout_ms: u64,
    results: BTreeMap<String, String>,
}

impl PlanMachine {
    fn new(plan: Plan, policy: ExecutorPolicy) -> Self {
        let states = plan
            .steps
            .iter()
            .map(|step| (step.id.clone(), ExecutorStepState::Pending))
            .collect();
        Self {
            plan,
            states,
            ledger: BTreeMap::new(),
            decisions: Vec::new(),
            retries: BTreeMap::new(),
            policy,
            status: MachineStatus::Idle,
            abort_requested: false,
            steps_executed: 0,
            tool_calls: 0,
            retry_count: 0,
            recoverable_errors: 0,
            replans: 0,
            rollbacks: 0,
            declared_timeout_ms: 0,
            results: BTreeMap::new(),
        }
    }

    fn run(
        mut self,
        task: &PlannerTask,
        planner: &impl Planner,
        registry: &ToolRegistry,
        runtime: &mut AuthorizedToolRuntime,
        executor: &mut ReferenceToolExecutor,
    ) -> ExecutorReport {
        if let Err(error) = self.plan.validate() {
            return self.fail(ExecutorError::InvalidPlan(error.to_string()));
        }
        self.status = MachineStatus::Running;
        let mut cursor = 0usize;
        loop {
            if self.abort_requested {
                return self.abort_remaining();
            }
            if let Some(limit) = self.policy.abort_after_steps {
                if self.steps_executed >= limit {
                    self.abort_requested = true;
                    self.record(
                        "machine",
                        ExecutorStepState::Running,
                        ExecutorStepState::Cancelled,
                        DecisionKind::Abort,
                        "abort-after-steps",
                    );
                    return self.abort_remaining();
                }
            }
            if cursor >= self.plan.steps.len() {
                return self.fail(ExecutorError::NoAnswer);
            }
            let step_id = self.plan.steps[cursor].id.clone();
            let state = self.state_of(&step_id);
            if state.is_terminal() {
                cursor += 1;
                continue;
            }
            if self.steps_executed >= self.plan.budget.max_steps {
                self.record(
                    &step_id,
                    state,
                    ExecutorStepState::Failed,
                    DecisionKind::RejectBudget,
                    "step budget",
                );
                return self.fail(ExecutorError::Budget("step budget".into()));
            }

            if let Err(error) = self.validate_dependencies(&step_id) {
                self.skip_dependents(&step_id);
                return self.fail(error);
            }

            self.record(
                &step_id,
                state,
                ExecutorStepState::Running,
                DecisionKind::Admit,
                "dependencies-ok",
            );
            self.set_state(&step_id, ExecutorStepState::Running);
            self.steps_executed += 1;

            let action = self.plan.steps[cursor].action.clone();
            match action {
                PlanAction::Tool(request) => {
                    if let Some(report) = self.run_tool(
                        task,
                        planner,
                        registry,
                        runtime,
                        executor,
                        &step_id,
                        request,
                        &mut cursor,
                    ) {
                        return report;
                    }
                }
                PlanAction::FinishFromStep(source) => {
                    let Some(answer) = self.results.get(&source).cloned() else {
                        self.record(
                            &step_id,
                            ExecutorStepState::Running,
                            ExecutorStepState::Failed,
                            DecisionKind::Fail,
                            "missing finish source",
                        );
                        self.set_state(&step_id, ExecutorStepState::Failed);
                        return self.fail(ExecutorError::NoAnswer);
                    };
                    self.record(
                        &step_id,
                        ExecutorStepState::Running,
                        ExecutorStepState::Done,
                        DecisionKind::Finish,
                        &answer,
                    );
                    self.set_state(&step_id, ExecutorStepState::Done);
                    return self.succeed(Some(answer));
                }
                PlanAction::FinishLiteral(value) => {
                    self.record(
                        &step_id,
                        ExecutorStepState::Running,
                        ExecutorStepState::Done,
                        DecisionKind::Finish,
                        &value,
                    );
                    self.set_state(&step_id, ExecutorStepState::Done);
                    return self.succeed(Some(value));
                }
            }
        }
    }

    fn run_tool(
        &mut self,
        task: &PlannerTask,
        planner: &impl Planner,
        registry: &ToolRegistry,
        runtime: &mut AuthorizedToolRuntime,
        executor: &mut ReferenceToolExecutor,
        step_id: &str,
        request: ToolRequest,
        cursor: &mut usize,
    ) -> Option<ExecutorReport> {
        if self.tool_calls >= self.plan.budget.max_tool_calls {
            self.record(
                step_id,
                ExecutorStepState::Running,
                ExecutorStepState::Failed,
                DecisionKind::RejectBudget,
                "tool-call budget",
            );
            self.set_state(step_id, ExecutorStepState::Failed);
            self.skip_dependents(step_id);
            return Some(self.fail(ExecutorError::Budget("tool-call budget".into())));
        }
        let Some(definition) = registry.resolve(&request.tool_name, request.tool_version) else {
            self.record(
                step_id,
                ExecutorStepState::Running,
                ExecutorStepState::Failed,
                DecisionKind::Fail,
                "unknown planned tool",
            );
            self.set_state(step_id, ExecutorStepState::Failed);
            self.skip_dependents(step_id);
            return Some(self.fail(ExecutorError::InvalidPlan("unknown planned tool".into())));
        };
        self.declared_timeout_ms = self.declared_timeout_ms.saturating_add(definition.timeout_ms);
        if self.declared_timeout_ms > self.plan.budget.max_timeout_ms {
            self.record(
                step_id,
                ExecutorStepState::Running,
                ExecutorStepState::Failed,
                DecisionKind::RejectBudget,
                "timeout budget",
            );
            self.set_state(step_id, ExecutorStepState::Failed);
            self.skip_dependents(step_id);
            return Some(self.fail(ExecutorError::Budget("timeout budget".into())));
        }

        let mutative =
            definition.mutative || self.policy.mutative_tools.contains(&request.tool_name);
        let action_fingerprint = action_fingerprint(&request);
        if mutative && !self.policy.allow_mutative_replay {
            if self.ledger.values().any(|entry| {
                entry.confirmed && entry.mutative && entry.action_fingerprint == action_fingerprint
            }) {
                self.record(
                    step_id,
                    ExecutorStepState::Running,
                    ExecutorStepState::Failed,
                    DecisionKind::RejectMutationReplay,
                    &request.tool_name,
                );
                self.set_state(step_id, ExecutorStepState::Failed);
                self.skip_dependents(step_id);
                return Some(self.fail(ExecutorError::MutationReplay(request.tool_name.clone())));
            }
        }

        self.tool_calls += 1;
        self.record(
            step_id,
            ExecutorStepState::Running,
            ExecutorStepState::Running,
            DecisionKind::InvokeTool,
            &request.tool_name,
        );
        let ledger = self
            .ledger
            .entry(step_id.to_string())
            .or_insert(StepLedgerEntry {
                step_id: step_id.to_string(),
                action_fingerprint,
                mutative,
                confirmed: false,
                invocations: 0,
            });
        ledger.invocations = ledger.invocations.saturating_add(1);

        if self.policy.rollback_before_confirm {
            self.rollbacks += 1;
            self.record(
                step_id,
                ExecutorStepState::Running,
                ExecutorStepState::Cancelled,
                DecisionKind::Rollback,
                "unconfirmed",
            );
            self.set_state(step_id, ExecutorStepState::Cancelled);
            self.abort_requested = true;
            return Some(self.abort_remaining());
        }

        let response = runtime.invoke(&request, false, executor);
        match response {
            ToolResponse::Result(result) => {
                self.results
                    .insert(step_id.to_string(), result.content.clone());
                if let Some(entry) = self.ledger.get_mut(step_id) {
                    entry.confirmed = true;
                }
                self.record(
                    step_id,
                    ExecutorStepState::Running,
                    ExecutorStepState::Done,
                    DecisionKind::Confirm,
                    &result.content,
                );
                self.set_state(step_id, ExecutorStepState::Done);
                *cursor += 1;
                None
            }
            ToolResponse::Error(error) => {
                self.record(
                    step_id,
                    ExecutorStepState::Running,
                    ExecutorStepState::Failed,
                    DecisionKind::Fail,
                    &error.code,
                );
                self.set_state(step_id, ExecutorStepState::Failed);
                if error.severity == ToolErrorSeverity::Recoverable {
                    self.recoverable_errors += 1;
                    let used = *self.retries.get(step_id).unwrap_or(&0);
                    if used < self.policy.max_retries_per_step {
                        self.retries.insert(step_id.to_string(), used + 1);
                        self.retry_count += 1;
                        self.record(
                            step_id,
                            ExecutorStepState::Failed,
                            ExecutorStepState::Pending,
                            DecisionKind::Retry,
                            &error.code,
                        );
                        self.set_state(step_id, ExecutorStepState::Pending);
                        self.steps_executed = self.steps_executed.saturating_sub(1);
                        return None;
                    }
                    match planner.replan(task, &self.plan, step_id, &ToolResponse::Error(error)) {
                        Ok(Some(revised)) => {
                            if let Err(err) = revised.validate() {
                                return Some(self.fail(ExecutorError::InvalidPlan(err.to_string())));
                            }
                            self.plan = revised;
                            self.replans += 1;
                            self.sync_new_steps();
                            self.record(
                                step_id,
                                ExecutorStepState::Failed,
                                ExecutorStepState::Failed,
                                DecisionKind::Replan,
                                &format!("revision={}", self.plan.revision),
                            );
                            *cursor += 1;
                            None
                        }
                        Ok(None) | Err(_) => {
                            self.skip_dependents(step_id);
                            Some(self.fail(ExecutorError::InvalidPlan(
                                PlannerError::NoReplan.to_string(),
                            )))
                        }
                    }
                } else {
                    self.skip_dependents(step_id);
                    Some(self.fail(ExecutorError::InvalidPlan(error.message)))
                }
            }
        }
    }

    fn validate_dependencies(&mut self, step_id: &str) -> Result<(), ExecutorError> {
        let step = self
            .plan
            .steps
            .iter()
            .find(|candidate| candidate.id == step_id)
            .ok_or_else(|| ExecutorError::InvalidPlan(format!("missing {step_id}")))?;
        for dep in &step.depends_on {
            match self.states.get(dep).copied() {
                Some(ExecutorStepState::Done) => {}
                Some(state) => {
                    self.record(
                        step_id,
                        ExecutorStepState::Pending,
                        ExecutorStepState::Failed,
                        DecisionKind::RejectDependency,
                        &format!("{dep}:{}", state.as_str()),
                    );
                    self.set_state(step_id, ExecutorStepState::Failed);
                    return Err(ExecutorError::Dependency(format!(
                        "{step_id} depends on incomplete {dep}"
                    )));
                }
                None => {
                    return Err(ExecutorError::Dependency(format!("missing {dep}")));
                }
            }
        }
        Ok(())
    }

    fn skip_dependents(&mut self, failed_id: &str) {
        let dependents: Vec<String> = self
            .plan
            .steps
            .iter()
            .filter(|step| step.depends_on.iter().any(|dep| dep == failed_id))
            .map(|step| step.id.clone())
            .collect();
        for id in dependents {
            if self.state_of(&id) == ExecutorStepState::Pending {
                self.record(
                    &id,
                    ExecutorStepState::Pending,
                    ExecutorStepState::Skipped,
                    DecisionKind::Skip,
                    failed_id,
                );
                self.set_state(&id, ExecutorStepState::Skipped);
            }
        }
    }

    fn abort_remaining(&mut self) -> ExecutorReport {
        let pending: Vec<String> = self
            .states
            .iter()
            .filter(|(_, state)| {
                **state == ExecutorStepState::Pending || **state == ExecutorStepState::Running
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in pending {
            let from = self.state_of(&id);
            if from == ExecutorStepState::Running {
                self.rollbacks += 1;
                self.record(
                    &id,
                    from,
                    ExecutorStepState::Cancelled,
                    DecisionKind::Rollback,
                    "abort",
                );
            } else {
                self.record(
                    &id,
                    from,
                    ExecutorStepState::Cancelled,
                    DecisionKind::Abort,
                    "abort",
                );
            }
            self.set_state(&id, ExecutorStepState::Cancelled);
        }
        self.status = MachineStatus::Cancelled;
        self.finish_report(false, None, Some(ExecutorError::Aborted.to_string()))
    }

    fn sync_new_steps(&mut self) {
        for step in &self.plan.steps {
            self.states
                .entry(step.id.clone())
                .or_insert(ExecutorStepState::Pending);
        }
    }

    fn state_of(&self, id: &str) -> ExecutorStepState {
        self.states
            .get(id)
            .copied()
            .unwrap_or(ExecutorStepState::Pending)
    }

    fn set_state(&mut self, id: &str, state: ExecutorStepState) {
        self.states.insert(id.to_string(), state);
        if let Some(step) = self
            .plan
            .steps
            .iter_mut()
            .find(|candidate| candidate.id == id)
        {
            step.status = match state {
                ExecutorStepState::Done => PlanStepStatus::Done,
                ExecutorStepState::Failed => PlanStepStatus::Failed,
                _ => PlanStepStatus::Pending,
            };
        }
    }

    fn record(
        &mut self,
        step_id: &str,
        from: ExecutorStepState,
        to: ExecutorStepState,
        kind: DecisionKind,
        detail: &str,
    ) {
        let seq = self.decisions.len() as u32;
        self.decisions.push(DecisionRecord {
            seq,
            step_id: step_id.into(),
            from,
            to,
            kind,
            detail: detail.into(),
        });
    }

    fn succeed(mut self, answer: Option<String>) -> ExecutorReport {
        self.status = MachineStatus::Succeeded;
        self.finish_report(true, answer, None)
    }

    fn fail(mut self, error: ExecutorError) -> ExecutorReport {
        self.status = MachineStatus::Failed;
        self.finish_report(false, None, Some(error.to_string()))
    }

    fn finish_report(
        self,
        success: bool,
        answer: Option<String>,
        error: Option<String>,
    ) -> ExecutorReport {
        let skipped = self
            .states
            .values()
            .filter(|state| **state == ExecutorStepState::Skipped)
            .count();
        let cancelled = self
            .states
            .values()
            .filter(|state| **state == ExecutorStepState::Cancelled)
            .count();
        ExecutorReport {
            success,
            status: self.status,
            answer,
            plan_revision: self.plan.revision,
            steps_executed: self.steps_executed,
            tool_calls: self.tool_calls,
            retries: self.retry_count,
            recoverable_errors: self.recoverable_errors,
            replans: self.replans,
            rollbacks: self.rollbacks,
            skipped,
            cancelled,
            declared_timeout_ms: self.declared_timeout_ms,
            decisions: self.decisions,
            error,
        }
    }
}

fn action_fingerprint(request: &ToolRequest) -> u64 {
    let mut blob = format!("{}@{}", request.tool_name, request.tool_version);
    for argument in &request.arguments {
        blob.push('|');
        blob.push_str(&argument.name);
        blob.push('=');
        blob.push_str(&format!("{:?}", argument.value));
    }
    fingerprint(blob.as_bytes())
}

fn esc(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_permissions::PermissionPolicy;
    use crate::planner::{PlanAction, PlanStep, ReferencePlanner};
    use crate::tool_protocol::{ToolArgument, ToolValue};

    fn runtime(registry: &ToolRegistry) -> (AuthorizedToolRuntime, ReferenceToolExecutor) {
        let policy = PermissionPolicy::default()
            .allow_tool("lookup", 1)
            .allow_tool("ref.error", 1)
            .allow_tool("ref.echo", 1);
        (
            AuthorizedToolRuntime::new(registry.catalog(), policy).unwrap(),
            registry.reference_executor(),
        )
    }

    fn run(task: &PlannerTask, policy: ExecutorPolicy) -> ExecutorReport {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let planner = ReferencePlanner;
        let plan = planner.plan(task, PlanBudget::default()).unwrap();
        let (mut runtime, mut executor) = runtime(&registry);
        execute_plan(
            task,
            &planner,
            plan,
            &registry,
            &mut runtime,
            &mut executor,
            policy,
        )
    }

    #[test]
    fn lookup_machine_confirms_and_replays_the_same_decision_log() {
        let task = PlannerTask::Lookup {
            key: "city-2".into(),
        };
        let a = run(&task, ExecutorPolicy::default());
        let b = run(&task, ExecutorPolicy::default());
        assert!(a.success && b.success);
        assert_eq!(a.answer.as_deref(), Some("1002"));
        assert_eq!(a.decision_log(), b.decision_log());
        assert_eq!(a.decision_fingerprint(), b.decision_fingerprint());
        assert!(a.decisions.iter().any(|d| d.kind == DecisionKind::Admit));
        assert!(a.decisions.iter().any(|d| d.kind == DecisionKind::Confirm));
        assert!(a.decisions.iter().any(|d| d.kind == DecisionKind::Finish));
        let trace = a.to_trace("exec-lookup");
        assert_eq!(trace.events.len(), a.decisions.len());
        assert_eq!(trace.canonical(), b.to_trace("exec-lookup").canonical());
    }

    #[test]
    fn failed_step_skips_dependents_instead_of_continuing() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let planner = ReferencePlanner;
        let task = PlannerTask::RecoverThenEcho {
            text: "recovered".into(),
        };
        let mut plan = planner.plan(&task, PlanBudget::default()).unwrap();
        plan.steps.push(PlanStep {
            id: "blocked".into(),
            depends_on: vec!["s1".into()],
            completion: "must not run after s1 fails without replan".into(),
            status: PlanStepStatus::Pending,
            action: PlanAction::FinishLiteral("leaked".into()),
        });
        let (mut runtime, mut executor) = runtime(&registry);
        let mut policy = ExecutorPolicy::default();
        policy.max_retries_per_step = 0;
        let report = execute_plan(
            &task,
            &NoReplan,
            plan,
            &registry,
            &mut runtime,
            &mut executor,
            policy,
        );
        assert!(!report.success);
        assert_eq!(report.skipped, 1);
        assert!(report.decisions.iter().any(|d| d.kind == DecisionKind::Skip));
        assert_ne!(report.answer.as_deref(), Some("leaked"));
    }

    #[test]
    fn recoverable_error_retries_then_replans() {
        let task = PlannerTask::RecoverThenEcho {
            text: "recovered".into(),
        };
        let report = run(&task, ExecutorPolicy::default());
        assert!(report.success);
        assert_eq!(report.answer.as_deref(), Some("recovered"));
        assert_eq!(report.retries, 1);
        assert_eq!(report.replans, 1);
        assert!(report.decisions.iter().any(|d| d.kind == DecisionKind::Retry));
        assert!(report.decisions.iter().any(|d| d.kind == DecisionKind::Replan));
    }

    #[test]
    fn abort_leaves_pending_steps_cancelled() {
        let task = PlannerTask::Lookup {
            key: "city-2".into(),
        };
        let mut policy = ExecutorPolicy::default();
        policy.abort_after_steps = Some(1);
        let report = run(&task, policy);
        assert!(!report.success);
        assert_eq!(report.status, MachineStatus::Cancelled);
        assert!(report.cancelled >= 1);
        assert!(report.decisions.iter().any(|d| d.kind == DecisionKind::Abort));
        assert!(report.answer.is_none());
    }

    #[test]
    fn unconfirmed_step_rolls_back_instead_of_confirming() {
        let task = PlannerTask::Lookup {
            key: "city-2".into(),
        };
        let mut policy = ExecutorPolicy::default();
        policy.rollback_before_confirm = true;
        let report = run(&task, policy);
        assert!(!report.success);
        assert_eq!(report.rollbacks, 1);
        assert_eq!(report.status, MachineStatus::Cancelled);
        assert!(report
            .decisions
            .iter()
            .any(|d| d.kind == DecisionKind::Rollback));
        assert!(!report.decisions.iter().any(|d| d.kind == DecisionKind::Confirm));
    }

    #[test]
    fn confirmed_mutative_action_is_not_repeated() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let planner = ReferencePlanner;
        let task = PlannerTask::Lookup {
            key: "city-2".into(),
        };
        let mut plan = planner.plan(&task, PlanBudget::default()).unwrap();
        let request = match &plan.steps[0].action {
            PlanAction::Tool(request) => request.clone(),
            _ => panic!("expected tool step"),
        };
        plan.steps.insert(
            1,
            PlanStep {
                id: "s1b".into(),
                depends_on: vec!["s1".into()],
                completion: "must refuse confirmed mutative replay".into(),
                status: PlanStepStatus::Pending,
                action: PlanAction::Tool(ToolRequest::new(
                    "plan-lookup-city-2:s1b",
                    request.tool_name.clone(),
                    request.tool_version,
                    request.arguments.clone(),
                )),
            },
        );
        let (mut runtime, mut executor) = runtime(&registry);
        let mut policy = ExecutorPolicy::default();
        policy.mutative_tools.insert("lookup".into());
        let report = execute_plan(
            &task,
            &planner,
            plan,
            &registry,
            &mut runtime,
            &mut executor,
            policy,
        );
        assert!(!report.success);
        assert!(report
            .decisions
            .iter()
            .any(|d| d.kind == DecisionKind::RejectMutationReplay));
    }

    #[test]
    fn hard_timeout_budget_fails_closed() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let planner = ReferencePlanner;
        let task = PlannerTask::Lookup {
            key: "city-2".into(),
        };
        let mut budget = PlanBudget::default();
        budget.max_timeout_ms = 50;
        let plan = planner.plan(&task, budget).unwrap();
        let (mut runtime, mut executor) = runtime(&registry);
        let report = execute_plan(
            &task,
            &planner,
            plan,
            &registry,
            &mut runtime,
            &mut executor,
            ExecutorPolicy::default(),
        );
        assert!(!report.success);
        assert!(report
            .decisions
            .iter()
            .any(|d| d.kind == DecisionKind::RejectBudget));
    }

    struct NoReplan;

    impl Planner for NoReplan {
        fn plan(&self, _task: &PlannerTask, _budget: PlanBudget) -> Result<Plan, PlannerError> {
            Err(PlannerError::NoReplan)
        }

        fn replan(
            &self,
            _task: &PlannerTask,
            _plan: &Plan,
            _failed_step: &str,
            _response: &ToolResponse,
        ) -> Result<Option<Plan>, PlannerError> {
            Ok(None)
        }
    }

    #[test]
    fn schema_constant_is_version_one() {
        assert_eq!(EXECUTOR_SCHEMA_VERSION, 1);
        let _ = ToolArgument {
            name: "unused".into(),
            value: ToolValue::String("x".into()),
        };
    }
}
