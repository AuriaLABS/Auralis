# Tool registry and reference tools

Issue #49 adds a model-decoupled registry on top of the #44 protocol.

```text
TOOL_REGISTRY_SCHEMA_VERSION = 1
MAX_REGISTRY_TOOLS = 256
```

## Contract

- register / unregister / list are covered by tests;
- unknown `name@version` fails before an executor runs;
- definitions validate through #44;
- `capability` and `mutative` are listed for #45 policies;
- reference tools are local and deterministic.

## Reference tools

| Name | Capability | Behaviour |
|---|---|---|
| `ref.echo` | fixture.read | echo `text` |
| `ref.add` | math.read | add two integers |
| `ref.fixture` | fixture.read | controlled key lookup |
| `ref.error` | fixture.read | injected recoverable error |
| `ref.latency` | fixture.read | deterministic tick label |

No shell, network, browser, credentials or arbitrary filesystem.

## Out of scope

Planner (#46), real OS sandbox, dynamic third-party tools.
