# Integrated capability battery

Issue #75 composes the already versioned reasoning (#130), code (#131) and memory (#135) suites under the #74 registry.

The battery does **not** collapse those capabilities into a single acceptance score. A regression in one module must remain visible while the others stay exact.

## Contract

```text
CAPABILITY_SUITE_SCHEMA_VERSION = 1
CAPABILITY_SUITE_SEED = 75659918
capabilities = reasoning, code, memory, agent
agent = reserved / blocked until #47
```

Profiles:

- `smoke`: child smoke profiles only;
- `full`: child full profiles.

CI runs `smoke`. Full remains an offline battery.

## Modules

- reasoning: synthetic composition, sequences, binding, planning, distractors;
- code: confined Rust expression patches, `rustc --test`, protected tests;
- memory: external-memory retrieval, conflict, stale data, horizon and capacity;
- agent: **not scored**. The task exists as a reserved registry entry so the battery cannot pretend the agentic profile is green.

## Isolation gate

`evaluate_isolated_regression(profile, kind)` runs the oracle on every executed capability except `kind`. The broken module must drop below exact-pass 1.0 while the others remain 1.0. Agent cannot be selected as a broken kind: that path is fail-closed until #47.

## Metrics

Six registry metrics:

- reasoning/code/memory exact-pass;
- agent status;
- executed-capability count;
- descriptive-mean-executed (published, never a gate).

## Running

```bash
cargo run --release --bin auralis_capability_suite_bench -- smoke
cargo run --release --bin auralis_capability_suite_bench -- full
cargo run --release --bin auralis_capability_suite_bench -- both
```

## Non-goals

#75 schema 1 does not:

- invent a global quality number as acceptance;
- execute agentic tools or traces;
- replace the child suite gates;
- wait for #47 before reporting the three objective modules that already exist.
