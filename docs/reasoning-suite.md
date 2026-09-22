# Deterministic reasoning suite

Issue #130 adds an objective synthetic reasoning suite on top of the versioned evaluation registry from #74. The suite is model-independent: it defines cases, exact answers, parsers, scoring and failure categories without requiring a judge model or free-form human interpretation.

## Contract

The suite contract is:

```text
REASONING_SUITE_SCHEMA_VERSION = 1
REASONING_SUITE_SEED = 130659918
```

Two profiles are available:

- `smoke`: 2 cases per task-kind/split bucket;
- `full`: 16 cases per task-kind/split bucket.

Each profile contains five task families across three explicit splits.

## Task families

### Arithmetic composition

A case starts from an integer and applies a deterministic ordered sequence of operations. Difficulty changes the operator structure: easy cases use a smaller operator set, while difficulty 3+ guarantees a higher-factor multiplication before the remaining composition.

The answer is one exact integer.

### Algorithmic sequence

A case exposes a deterministic sequence prefix and asks for exactly the next two integers. Difficulty 1–2 uses zero acceleration; difficulty 3+ always uses non-zero second-order acceleration.

The answer is an ordered comma-separated integer sequence. Wrong sequence length is reported separately from wrong values.

### Variable binding

A case executes assignments and rebindings left-to-right and asks for one final variable value. Easy cases use single-source rebindings; difficulty 3+ uses weighted multi-source expressions.

A numerically valid but incorrect answer is classified as `wrong-binding`.

### Finite-state planning

The finite state-space size grows deterministically with case length. States are `S0..S<n>` for that case.

- action `A`: +1 modulo the case state count;
- action `B`: +2 modulo the case state count;
- difficulty 3+ adds action `C`: +3 modulo the case state count, increasing the branching factor.

The requested target offset also grows with case length, so the `eval-length` split has a strictly longer shortest-plan horizon than the training split.

The requested answer is the lexicographically first shortest plan, with action ordering `A` before `B`, plus final state and step count:

```text
plan=<AB...>;state=S<n>;steps=<n>
```

The oracle uses deterministic breadth-first search. Any structurally valid but incorrect plan/state/step triple is `wrong-transition`.

### Distractor robustness

The prompt contains relevant arithmetic facts plus a deterministic list of plausible but unrelated values. The number of distractor notes grows with case length, and difficulty 3+ moves distractors much closer to the correct value.

Returning any listed distractor is classified as `distractor-capture`, separately from a generic wrong value.

## Difficulty and length splits

The correctness rule never changes across splits.

`train-difficulty`:
- difficulty 1–2;
- short lengths.

`eval-difficulty`:
- difficulty 3–5;
- lengths remain in the short regime.

`eval-length`:
- difficulty stays moderate;
- lengths are strictly beyond the training-length range.

This separates harder composition from length generalization instead of mixing them into one score.

## Objective structured scoring

The scorer accepts only the task-specific canonical structure.

Examples:

- integer: `17`;
- sequence: `4,7`;
- plan: `plan=AB;state=S3;steps=2`.

Text such as `The answer is 17` is intentionally rejected as `invalid-format`. The suite does not reward textual similarity or partial prose matches.

Failure taxonomy:

- `invalid-format`;
- `wrong-value`;
- `wrong-length`;
- `wrong-binding`;
- `wrong-transition`;
- `distractor-capture`.

A report keeps total/correct counts plus per-kind and per-split totals, so an improvement in one family cannot hide a regression in another.

## Registry identity

`suite_definition(profile, seed)` builds an #74 `EvaluationSuite` with:

- profile-specific suite id;
- exact fixture fingerprint;
- one versioned task entry per reasoning family;
- namespaced metrics;
- exact seed policy;
- declared resource limits.

Smoke and full profiles have distinct exact suite references and are not automatically comparable as the same suite.

## Regression fixture

`intentional_regression_predictions` deliberately produces:

- wrong arithmetic values;
- truncated sequences;
- incorrect bindings;
- invalid finite-state transitions;
- captured distractors.

The full suite must detect all five semantic failure categories. This is a regression-sensitivity gate, not a model-quality result.

## Running the profiles

```bash
cargo run --release --bin auralis_reasoning_suite_bench -- smoke
cargo run --release --bin auralis_reasoning_suite_bench -- full
cargo run --release --bin auralis_reasoning_suite_bench -- both
```

The benchmark publishes:

- case/task/metric counts;
- suite and fixture fingerprints;
- generation/scoring latency;
- training maximum length;
- difficulty-eval minimum difficulty;
- length-eval minimum length;
- oracle exact-match;
- failure taxonomy for the intentional regression.

Timing is descriptive. Correctness, determinism, split separation and regression detection are the gates.

## Non-goals

#130 does not:

- train Auralis on these fixtures;
- use an LLM-as-judge;
- assign a single global “intelligence score”;
- mix reasoning with code/memory/agent tasks;
- auto-promote a model or architecture;
- depend on external benchmark services.

A model runner can consume these exact cases later while preserving the suite reference and scorer.
