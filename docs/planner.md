# Explicit planner

Issue #46 introduces a versioned, inspectable plan between reasoning and tool execution.

The planner does not execute tools. The minimal runner consumes structured steps only through the existing Agent contracts:

1. #44 typed request/schema validation;
2. #45 deny-by-default authorization and budgets;
3. #49 registry resolution and deterministic local reference executor;
4. #47 remains the trace/evaluation layer.

Full cancel/rollback/retry state-machine semantics remain #114.

## Plan schema

PLAN_SCHEMA_VERSION = 1

A plan records:

- stable plan ID and revision;
- goal;
- hard step/tool/declared-timeout budgets;
- ordered steps;
- explicit dependencies;
- per-step completion criterion;
- step status;
- structured action.

Current actions are typed tool request, finish from a previous step result, and literal finish.

Dependencies must reference an earlier step. Unknown/malformed tools fail before execution. A dependency that is not done blocks the dependent step; the runner never silently skips a failed dependency.

Plan::canonical() and Plan::fingerprint() make the same deterministic plan inspectable/replayable in tests and traces.

## Replanning

ReferencePlanner::replan is invoked only after a recoverable tool error.

The controlled recovery fixture starts with ref.error@1. The failed step is recorded as failed, the plan revision increases, and a new ref.echo@1 fallback plus finish step are appended. A fatal error or a recoverable error without a valid replan stops the run.

No mutative tool is used by this reference planner.

## Hard budgets

PlanBudget caps steps, tool calls, and the sum of declared tool timeout budgets.

Budget exhaustion fails closed before the next tool invocation. These limits are separate from and additional to #45 per-call permission/resource checks.

## Direct vs planner suite

auralis_planner_bench executes two deterministic local cases under the same registry and allowlist:

1. normal lookup;
2. recoverable tool failure.

The direct strategy does not replan. The planner strategy may replan once when the fixture explicitly allows it.

Reported metrics include exact task success, steps, tool calls, recoverable errors, replans, deterministic cost_units (steps + 4*tool_calls + 2*replans), plan fingerprint, authorization audit count, and executor invocation count.

The comparison is descriptive. It does not make the planner the default agent strategy and does not claim general superiority from two fixtures.

## Boundaries

#46 does not add autonomous indefinite loops, shell/network/filesystem tools, sessions or persistence (#48), the full executor state machine/rollback (#114), local HTTP API (#50), or model self-editing.

The planner is an explicit architecture boundary that later Agent work can reuse.
