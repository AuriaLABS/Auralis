# Executable code evaluation suite

Issue #131 adds an objective Rust code-evaluation suite on top of the #74 evaluation registry.

The suite does **not** score textual similarity. A submission must apply cleanly, compile with `rustc --test`, and pass protected executable tests.

## Contract

```text
CODE_SUITE_SCHEMA_VERSION = 1
CODE_SUITE_SEED = 131659918
language = Rust 2021
patch dialect = one declared expression in main.rs
compiler = rustc --test
```

Profiles:

- `smoke`: 1 case per family, 4 cases total;
- `full`: 4 cases per family, 16 cases total.

## Task families

### Code comprehension

The fixture contains a small deterministic function and an incorrect `ANSWER` expression. The patch must derive the executable result implied by the code; the protected test checks it.

### Compile repair

The base fixture contains one declared expression hole. The patched fixture must compile and pass its protected behavior test.

### Unit-test repair

The base implementation compiles but fails protected unit tests. The patch repairs the implementation expression without modifying tests.

### Bug-fix patch

The base implementation contains a concrete boundary regression. The expression patch must fix the bug while preserving all protected regression tests.

## Structured patch application

A submission contains:

- path;
- exact expected old expression;
- replacement expression.

The only accepted path is `main.rs`, confined through `sec_path::confine`. The expected-old field must equal the case's declared patch target and that target must occur exactly once.

The replacement dialect is intentionally narrow: one ASCII expression, no statements, braces, attributes, string literals, path separators such as `::`, comments, `unsafe`, imports or include/environment macros. This makes the protected test suffix immutable: a candidate cannot pass by deleting or commenting out tests.

Patch validation failures are scored objectively as `patch-rejected` or `sandbox-violation`.

## Deterministic execution sandbox

Each execution gets a fresh temporary directory containing only the hardened fixture source and compiler output. The evaluator:

1. prepends `#![forbid(unsafe_code)]`;
2. invokes the discovered local `rustc` directly with `--edition=2021 --test`;
3. does not use Cargo dependencies or external services;
4. executes only the resulting protected test binary;
5. kills compile/test processes when their configured timeout expires;
6. removes the sandbox directory after scoring.

This is a constrained evaluation harness for the suite's expression-patch dialect, not a general-purpose security sandbox for arbitrary Rust programs.

Default resource limits:

- source <= 32 KiB;
- patch <= 256 bytes;
- compiler diagnostics <= 256 KiB;
- compile timeout 10 s/case;
- test timeout 5 s/case.

The #74 `EvaluationSuite` publishes corresponding wall/resource metadata.

## Objective scoring

A case is exact only when:

- patch application succeeds;
- source policy accepts the replacement;
- compilation succeeds within the timeout;
- the protected test binary exits successfully within the timeout.

Failure taxonomy:

- `patch-rejected`;
- `sandbox-violation`;
- `compile-error`;
- `compile-timeout`;
- `test-failure`;
- `test-timeout`.

Compiler errors are further classified as:

- `syntax`;
- `type-mismatch`;
- `borrow-check`;
- `missing-item`;
- `other`.

The intentional regression profile uses a missing symbol for compile tasks and wrong-but-compiling expressions for the other families. It must score 0 exact while surfacing both compile and test failures.

## Registry metrics

The suite registers 16 metrics:

- exact-pass, compile-pass and test-pass ratios;
- compile/test latency;
- six failure counts;
- five compiler-error taxonomy counts.

Timing is descriptive. Executable correctness and failure classification are gates.

## Running

```bash
cargo run --release --bin auralis_code_suite_bench -- smoke
cargo run --release --bin auralis_code_suite_bench -- full
cargo run --release --bin auralis_code_suite_bench -- both
```

## Non-goals

#131 does not:

- execute arbitrary multi-file repositories;
- permit dependency installation or network access;
- treat compiler/test latency as model quality;
- use an LLM judge;
- score source-code similarity;
- let candidates modify protected tests.
