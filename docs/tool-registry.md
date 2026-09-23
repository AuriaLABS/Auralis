# Tool registry and reference tools

Issue #49 adds a model-decoupled, versioned registry on top of the #44 protocol.

```text
TOOL_REGISTRY_SCHEMA_VERSION = 1
MAX_REGISTRY_TOOLS = 256
```

## Contract

`ToolRegistry` supports dynamic `register_definition`, `unregister`, `list` and `resolve(name, version)`. Every definition is validated through #44 before registration. Duplicate `name@version` identities are rejected, listing order is deterministic, and discovery exposes exact capability and `mutative` metadata for #45 policy decisions.

The registry **grants no authority**. It does not expose a direct public execution shortcut. Callers pass `ToolRegistry::catalog()` into #45 `AuthorizedToolRuntime` and use `reference_executor()` only after the request has passed protocol validation, permission/budget checks and audit.

Dynamic definitions are discovery-only unless some executor explicitly implements them. They are never silently mapped onto a built-in reference behavior.

## Reference tools

The built-ins are local, deterministic fixtures:

| Name | Capability | Behaviour |
|---|---|---|
| `lookup@1` | agent.lookup | #47-compatible `city-N` lookup; `missing-N` returns recoverable `not-found` |
| `ref.echo@1` | fixture.read | echo validated `text` |
| `ref.add@1` | math.read | add two bounded integers |
| `ref.fixture@1` | fixture.read | controlled alpha/beta lookup |
| `ref.error@1` | fixture.read | injected recoverable error |
| `ref.latency@1` | fixture.read | deterministic tick label; it does not sleep |

No shell, network, browser, credentials, arbitrary filesystem or generated-code execution is implemented.

## Security ordering

The supported reference execution path is:

```text
registry discovery
  -> #44 request/schema validation
  -> #45 authorization + resource budgets
  -> #45 audit
  -> ReferenceToolExecutor
```

An unknown tool cannot reach the executor. Deny-by-default remains owned by #45. The registry cannot turn a `mutative` definition into authority by registration alone.

## #47 compatibility

#47 agent-eval advertises the tool name `lookup`. Both smoke and full fixtures resolve `lookup@1` from the reference registry. Its `city-N` and `missing-N` behavior matches the deterministic mock-eval contract without external effects.

## Boundaries

- #44 owns schemas and typed request/result validation.
- #45 owns permissions, budgets and authorization audit.
- #49 owns discovery/registration and safe local reference executors.
- #46 owns planning/replanning.
- #47 owns traces, replay and agentic evaluation.

Registry state is not model/checkpoint state and is not permission state.
