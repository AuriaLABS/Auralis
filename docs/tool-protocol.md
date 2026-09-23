# Tool protocol

Issue #44 defines the model-independent contract between an Agent tool request and a replaceable runtime. It does **not** grant permissions and does not execute shell, network, browser or third-party actions.

## Versioned envelopes

The current contract is `TOOL_PROTOCOL_SCHEMA_VERSION = 1`.

Four payloads are explicit and machine-readable:

- `ToolDefinition` — tool identity/version, capability, mutative flag, timeout, result-size limit and typed arguments;
- `ToolRequest` — stable `call_id`, exact tool name/version and typed argument values;
- `ToolResult` — correlated successful content;
- `ToolError` — correlated structured error with `recoverable` or `fatal` severity.

Canonical serialization uses strict `key=value` records plus percent escaping. Unknown fields, duplicate fields, malformed escapes and unsupported definition schemas fail closed.

## Typed validation

Schema 1 supports:

- bounded UTF-8 strings;
- bounded signed integers;
- booleans;
- finite declared enums represented by string values.

Definitions reject duplicate argument names, invalid bounds, empty enums, invalid identifiers, zero versions, invalid timeout and oversized result limits.

Requests are validated against one exact definition before execution. Validation rejects:

- wrong tool name/version;
- missing required arguments;
- unknown or duplicate arguments;
- wrong argument types;
- string/integer bounds violations;
- enum values outside the declared set.

## No validation bypass

The public execution path is `invoke_validated`.

A runtime implements `ToolExecutor::execute(ValidatedToolCall)`. The constructor for `ValidatedToolCall` is private to the protocol module, so callers cannot construct an executable call from an unchecked `ToolRequest`.

If request validation fails, the executor invocation count remains zero and a fatal `protocol-validation` error is returned.

Runtime responses are validated again before they leave the protocol boundary. Wrong correlation IDs, oversized results or malformed structured errors are converted to fatal `invalid-tool-response` errors.

## Correlation and errors

`call_id` is stable across request/result/error. It is intentionally protocol-level state so #47 traces/replay can correlate calls without depending on runtime implementation details.

`ToolErrorSeverity::Recoverable` means a later planner may retry/replan. `Fatal` means the protocol/runtime considers the call non-recoverable. #44 only carries this classification; retry policy belongs to #46.

## Runtime replacement and mocks

`DeterministicMockExecutor` is the reference no-side-effect backend. It can:

- echo one validated argument;
- return a fixed result;
- return a fixed recoverable/fatal error.

Replacing this mock with another `ToolExecutor` does not require model changes.

## Boundaries with later Agent work

- #45 owns allow/deny policy, sandboxing and authorization.
- #46 owns plans, budgets, retries and replanning.
- #47 owns traces, deterministic replay and agentic evaluation.
- #49 owns registry/discovery and reference tools.

The protocol itself grants **no authority**. A `mutative=true` definition is metadata for #45; it is not permission to execute.

## Security/resource limits

Schema 1 caps argument count, identifier lengths, string argument length, timeout declaration, response bytes and total serialized protocol text. These are protocol parsing/resource bounds, not OS sandbox guarantees.

The core invariant is: **no tool execution through the public protocol path before strict validation**.
