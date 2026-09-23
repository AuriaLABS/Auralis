# Agent permissions and sandbox boundary

Issue #45 adds the authorization layer between validated #44 tool requests and execution. The runtime is deliberately local/mock-only in this phase: it does not add shell, network, browser or third-party effects.

## Policy contract

The current contract is `PERMISSION_POLICY_SCHEMA_VERSION = 1`.

`PermissionPolicy::default()` is **deny-by-default**:

- no exact tools are allowlisted;
- no capabilities are allowlisted;
- mutative tools are denied;
- timeout, result-size and argument-size budgets are bounded.

A policy may allow an exact `name@version`, a capability, or both. The policy can be replaced at runtime with `set_policy()`; no model recompilation is required.

## Execution order

`AuthorizedToolRuntime::invoke` has a strict order:

1. resolve the exact tool definition from the runtime catalog;
2. validate the #44 `ToolRequest` against that definition;
3. evaluate allowlists, resource budgets and mutation policy;
4. append an `AuthorizationDecision` audit record;
5. only when the decision is `allow`, call #44 `invoke_validated`.

Unknown tools, malformed arguments and denied policies never reach the executor. Tests assert the deterministic mock executor invocation count remains zero for those paths.

The runtime catalog in #45 is intentionally fixed at construction. Dynamic register/unregister/discovery belongs to #49.

## Read-only and mutative tools

The #44 `ToolDefinition::mutative` flag is consumed by #45.

Schema 1 supports three policies:

- `Deny` — mutative calls never execute;
- `RequireConfirmation` — the invocation must carry an explicit confirmation bit;
- `Allow` — an otherwise-authorized mutative call may execute.

A confirmation bit does not bypass tool/capability allowlists or resource budgets. Every mutative attempt is audited whether it is allowed or denied.

## Resource budgets

Authorization checks declared and actual request costs before execution:

- definition `timeout_ms <= policy.max_timeout_ms`;
- definition `max_result_bytes <= policy.max_result_bytes`;
- actual encoded argument-name/value bytes `<= policy.max_argument_bytes`.

The #44 response boundary still validates actual result size after execution, so authorization budgets complement rather than replace protocol validation.

## Auditing and replay support

Each decision records:

- monotonic sequence;
- stable `call_id`;
- tool name/version and capability;
- mutative/confirmation flags;
- allow/deny outcome;
- machine-readable reason;
- permission-policy fingerprint;
- declared timeout/result budget and actual argument bytes.

Argument **values are never copied into the authorization audit**.

`decision_for_call()` and `explain_call()` resolve the latest decision by stable `call_id`, providing the deterministic explanation surface that #47 traces/replay can consume later without coupling replay to an executor.

## Denial taxonomy

Important reasons include:

- `unknown-tool`;
- `protocol-validation`;
- `not-allowlisted`;
- `mutation-denied`;
- `confirmation-required`;
- `timeout-budget`;
- `result-budget`;
- `argument-budget`.

Policy denials return a structured fatal `authorization-denied` tool error. Protocol-invalid calls preserve the distinct fatal `protocol-validation` error.

## Sandbox/reference execution

Tests use #44 `DeterministicMockExecutor`. This is the #45 sandbox/reference environment: calls can exercise validation, authorization, denial, confirmation, audit and response handling with no external side effects.

Real operating-system isolation, shell execution and network access are not introduced by this issue.

## Boundaries

- #44 owns typed request/result validation.
- #45 owns authorization, budgets and audit decisions.
- #46 owns planning, retries and step/tool-call budgets.
- #47 owns complete traces, replay and agentic evaluation.
- #49 owns dynamic tool registry/discovery.

The invariant is: **validate, authorize, audit, then execute**. No denial path performs partial tool execution.
