# Explicit multi-step planner

Issue #46 separates planning from execution.

```text
PLAN_SCHEMA_VERSION = 1
MAX_PLAN_STEPS = 32
MAX_PLAN_TOOL_CALLS = 24
MAX_PLAN_TIMEOUT_MS = 30000
```

## Contract

A `Plan` is inspectable and reproducible: goal, ordered steps, earlier-only dependencies, per-step completion criterion, status, revision and a hard `PlanBudget`. `ReferencePlanner::plan` never invokes a tool. `run_plan` consumes already structured steps through #44 validation, #45 authorization and #49 local reference tools.

Recoverable tool errors do not continue silently. Dependents of a failed step stay blocked. A replan is explicit, increments `revision`, and is itself budgeted. Fatal errors and missing replans fail closed.

## Comparison fixture

The controlled suite compares the same `PlannerTask` under two strategies:

- planned: `run_plan` may recover `ref.error@1 busy` by appending `ref.echo`;
- direct: `run_direct` executes the first tool once and stops.

The comparison does **not** claim that explicit planning is generally superior. It only shows a recoverable-error fixture where the planner recovers and the direct path does not.

## Boundaries

- no autonomy without a budget;
- no self-editing of agent code;
- no shell/network/filesystem tools;
- cancellation/rollback/retry state machine remains #114;
- #47 owns traces and replay of sessions.
