# Integrated capability battery

Issue #75 composes the already versioned reasoning (#130), code (#131) and memory (#135) suites under the #74 registry, plus the #47 agent evaluation child implemented in `src/agent_eval.rs`.

The battery does **not** collapse those capabilities into a single acceptance score. A regression in one module must remain visible while the others stay exact.

## Contract

```text
CAPABILITY_SUITE_SCHEMA_VERSION = 2
suite version = 2.0.0
CAPABILITY_SUITE_SEED = 75659918
capabilities = reasoning, code, memory, agent
agent = #47 mock-replay suite integrated via agent_eval
```

The schema-2 composition is a new exact suite definition (`2.0.0`, dataset revision `2`) because the agent fixtures/tasks/metrics change the registry identity.

Profiles:

- `smoke`: child smoke profiles only;
- `full`: child full profiles.

CI runs `smoke`. Full remains an offline battery.

## Modules

- reasoning: synthetic composition, sequences, binding, planning, distractors;
- code: confined Rust expression patches, `rustc --test`, protected tests;
- memory: external-memory retrieval, conflict, stale data, horizon and capacity;
- agent: #47 `agent_eval` versioned traces + deterministic mock replay with explicit tool/reasoning/permission failure classes.

## Isolation gate

`evaluate_isolated_regression(profile, kind)` runs the oracle on every executed capability except `kind`. The broken module must drop below exact-pass 1.0 while the others remain 1.0. Agent can also be selected as the isolated broken capability; its intentional regression must fall to 0 while reasoning/code/memory remain exact.

## Metrics

Six registry metrics:

- reasoning/code/memory/agent exact-pass;
- executed-capability count;
- descriptive-mean-executed (published, never a gate).

## Running

```bash
cargo run --release --bin auralis_capability_suite_bench -- smoke
cargo run --release --bin auralis_capability_suite_bench -- full
cargo run --release --bin auralis_capability_suite_bench -- both
```

## Non-goals

#75 schema 2 does not:

- invent a global quality number as acceptance;
- replace the child suite gates;
