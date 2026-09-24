# Tool provider contract (#113)

`TOOL_PROVIDER_SCHEMA_VERSION = 1`

Providers sit behind the registry and #45. Planner/model core do **not**
import a provider.

- `ToolProvider`: list / describe / invoke / negotiate;
- mock provider reference;
- reusable `run_provider_contract`;
- timeout and cancel propagate;
- two providers do not share invocation state;
- invoking a provider does not grant permissions.

No implicit runtime allowlist, no planner/model edits required to add a provider.
