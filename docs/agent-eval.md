# Agent traces, replay and evaluation suite

Issue #47 adds a versioned observability/evaluation layer for multi-step agent tasks.

```text
AGENT_EVAL_SCHEMA_VERSION = 1
AGENT_EVAL_SEED = 47659918
COMMON_BASELINE_ID = agent-common-baseline-1
replay = mock / recorded results only
real tool effects = refused
```

## Common baseline

`CommonBaseline` pins the schema numbers of #44 protocol, #45 permissions,
#46 planner, #114 executor, #49 registry, #48 session and #50 local API
next to this eval schema. A replay trace stores that canonical string in
`config`. Direct and Planned strategies are compared on the same mock
suite; they must agree on exact-pass and failure class. Bumping any of
those schemas changes the fingerprint.

## Trace

A `SessionTrace` records stable session/step/call IDs plus:

- messages;
- plan;
- tool calls;
- tool results;
- errors;
- timings.

Commit, config, model and checkpoint identity travel with the trace. Payloads run through `redact` before storage. Tokens, passwords and `SECRET` markers become `[REDACTED]`.

## Replay

`replay_case(..., allow_real_effects=true)` is fail-closed. The default path uses in-process mocks. No network, filesystem mutation or process spawn.

## Tasks

Smoke: 4 cases. Full: 8 cases. Families:

- lookup-then-answer;
- permission-denied;
- tool-failure;
- reasoning-failure.

Exact-pass means the expected failure class is observed. Tool failures and reasoning failures are counted separately.

Intentional regression forces exact-pass to 0.

## Running

```bash
cargo run --release --bin auralis_agent_eval_bench -- smoke
```

## Non-goals

Schema 1 still does not execute #44 tools, #45 sandbox or #46 planner
inside replay. It only pins those schemas on the common baseline so a
future change can be compared. No external analytics or user telemetry.
