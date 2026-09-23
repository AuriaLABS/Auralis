# Explicit multi-step planner

Issue #46 separates planning from execution.

PLAN_SCHEMA_VERSION = 1

Limits:
- MAX_PLAN_STEPS = 32
- MAX_PLAN_TOOL_CALLS = 24
- MAX_PLAN_TIMEOUT_MS = 30000

## Contract

A Plan is inspectable and reproducible: goal, ordered steps, earlier-only dependencies, per-step completion criterion, status, revision and a hard PlanBudget.

ReferencePlanner::plan never invokes a tool. run_plan consumes structured steps through:

1. #44 typed request/schema validation;
2. #45 deny-by-default authorization and per-call budgets;
3. #49 registry resolution plus deterministic local reference tools;
4. #47 remains the trace/evaluation layer.

Plan::canonical() and Plan::fingerprint() provide a stable representation for inspection and replay in tests/traces.

## Replanning

Recoverable tool errors do not continue silently.

The controlled recovery fixture starts with ref.error@1. The failed step is marked failed, plan revision increments, and ReferencePlanner::replan appends a ref.echo@1 fallback plus finish step.

Fatal errors and recoverable errors without a valid replan fail closed. Dependents of a failed step remain blocked.

## Hard budgets

PlanBudget caps:
- executed steps;
- tool calls;
- sum of declared tool timeout budgets.

Budget exhaustion fails closed before the next tool invocation. These are additional to #45 permission/resource checks.

## Controlled direct-vs-planner comparison

auralis_planner_bench runs the same two local tasks under the same registry and allowlist:

- lookup: both direct and planner can complete;
- recover: direct executes the first tool once and stops on the recoverable error; planner performs one explicit replan and can complete via the deterministic fallback.

Reported metrics:
- exact task success;
- steps;
- tool calls;
- recoverable errors;
- replans;
- deterministic cost_units = steps + 4*tool_calls + 2*replans;
- plan fingerprint;
- authorization audit events;
- reference-executor invocations.

The comparison does not make the planner the default and does not claim general superiority from two fixtures.

## Boundaries

#46 does not add:
- autonomy without a budget;
- self-editing;
- shell/network/filesystem tools;
- sessions or persistence (#48);
- cancellation/rollback/retry state machine (#114);
- local HTTP API (#50).

The minimal runner is deliberately narrower than #114.
