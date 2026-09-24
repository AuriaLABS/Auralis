# Plan executor state machine

Issue #114 separates the #46 planner from an explicit execution machine.

EXECUTOR_SCHEMA_VERSION = 1

The machine never invents tool calls. It only admits already structured plan steps through #44 validation, #45 authorization and #49 local reference tools.

## States

Per-step states: pending, running, done, failed, skipped, cancelled.

Machine states: idle, running, succeeded, failed, cancelled.

A step is admitted only after every listed dependency is done. A failed step skips dependents instead of continuing them silently.

## Decisions

Every transition is a DecisionRecord:

- admit
- reject-dependency
- reject-budget
- reject-mutation-replay
- invoke-tool
- confirm
- fail
- retry
- replan
- skip
- abort
- rollback
- finish

`ExecutorReport::decision_log()` is the replayable sequence of decisions. Two identical mock runs produce the same log and the same #47 SessionTrace.

## Budgets

Hard caps inherited from PlanBudget:

- executed steps
- tool calls
- sum of declared tool timeout budgets

Budget rejection happens before the next invocation.

## Abort and rollback

`abort_after_steps` requests a safe abort between steps. Pending and running steps become cancelled. An unconfirmed running step is rolled back logically: it is not marked done and it is not confirmed in the ledger.

Cancellation leaves a consistent machine: no later dependent is admitted, and the report answer stays empty.

## Mutation policy

A confirmed mutative action fingerprint is not invoked again unless `allow_mutative_replay` is explicit. Default policy refuses the replay and fails closed.

Reference tools stay non-mutative unless a test policy marks them mutative. No shell, network or filesystem effects.

## Boundaries

#114 does not add:

- autonomy without a budget
- self-editing
- real I/O tools
- sessions or persistence (#48)
- local HTTP API (#50)
- a claim that the machine is the default agent runtime
