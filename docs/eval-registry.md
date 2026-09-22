# Versioned evaluation registry

Issue #74 defines the first library-first registry contract for Auralis evaluation suites and canonical baselines. The purpose is to make later reasoning, code, memory and agent evaluations comparable without relying on implicit context.

The registry is offline and deterministic. It does not fetch datasets or external benchmarks during smoke tests.

## Schema

The registry contract uses:

```text
EVALUATION_REGISTRY_SCHEMA_VERSION = 1
```

The core versioned objects are:

- `EvaluationSuite`
- `EvaluationTask`
- `EvaluationMetric`
- `Baseline`
- `SuiteRef`
- `EvaluationRunDescriptor`

Suite, task, metric and baseline versions use explicit semantic versions `major.minor.patch`.

## Exact suite identity

A suite definition includes:

- suite id/version and description;
- dataset id + dataset revision;
- split name;
- evaluated population description;
- fixture fingerprint and sample count;
- ordered task definitions and task versions;
- metric definitions, metric versions, units and direction;
- seed policy;
- resource limits.

`EvaluationSuite::definition_fingerprint()` fingerprints the canonical definition. `SuiteRef` stores:

```text
suite id + exact semantic version + definition fingerprint
```

An evaluation run is valid only when its `EvaluationRunDescriptor` resolves that exact reference.

## Version-collision rule

Changing dataset metadata, fixture identity, task semantics, metrics, seed policy or limits changes the suite definition fingerprint.

If a caller attempts to register a different definition using an already registered suite id + version, the registry rejects it with `SuiteVersionCollision`.

This is the enforcement mechanism for the rule:

> dataset or metric semantics cannot change while pretending to be the same suite version.

The caller must bump the suite version explicitly.

Task and metric identities are also global within the registry. Reusing a task id/version or metric id/version with different semantics is rejected with `TaskVersionCollision` or `MetricVersionCollision`; changing semantics therefore requires bumping the task/metric version as well as the suite version when applicable.

The same rule applies to baselines: an existing baseline id + version cannot silently change metrics/config/model metadata. Such drift is rejected with `BaselineVersionCollision`.

## Dataset, split, seed and population metadata

`DatasetMetadata` records:

- `id`
- `revision`
- `split`
- `population`
- `fixture_fingerprint`
- `sample_count`

`SeedPolicy` stores the exact ordered seed set and rejects duplicates.

`ResourceLimits` pins the declared maximum examples, steps, tokens and wall-clock budget. #74 records these limits as experiment identity metadata; enforcement by a campaign runner belongs to #76.

## Tasks and metrics

Every task records its own id, semantic version, kind, fixture fingerprint and fixture count.

Every metric records its own id, semantic version, unit, direction and description.

A baseline must report exactly one finite value for every metric in the referenced suite. Unknown, missing or duplicate metrics fail closed.

## Historical baselines

The reference `EvaluationRegistry` retains all registered baseline versions. Registering a newer baseline does not delete the previous one.

Baseline metadata requires a real canonical `YYYY-MM-DD` calendar date plus non-empty code revision/model identity. Invalid dates such as `2026-02-30` fail closed.

Queries can address:

- a baseline by `id + version`;
- every historical version of a baseline id;
- baselines attached to an exact `SuiteRef`.

No API automatically replaces a canonical baseline. Promotion policy is intentionally outside #74.

## Deprecation

Suites are never deleted merely because they become obsolete.

`deprecate_suite` records:

- an explicit reason;
- an optional exact replacement `SuiteRef`.

The old suite and baselines remain queryable. A suite cannot deprecate itself in favor of itself.

## Compatibility

`exact_compatible(a, b)` returns true only for the same registered exact `SuiteRef`.

Different semantic versions or fingerprints are not merged or treated as statistically equivalent by this layer.

Later comparison tooling may define explicit cross-version migration/interpretation policies, but #74 refuses to invent one automatically.

## Offline smoke fixture

Run:

```bash
cargo run --release --bin auralis_eval_registry_smoke
```

The smoke program:

1. registers deterministic suite v1;
2. registers baseline v1;
3. registers a deliberately version-bumped suite v2;
4. records explicit v1 → v2 deprecation;
5. registers baseline v2;
6. proves the historical v1 baseline remains queryable;
7. reports exact references and fingerprints.

The fixture bytes are embedded in the repository and fingerprinted with the existing Auralis experiment fingerprint primitive.

## Reproduction contract for another agent

To reproduce an evaluation, another agent needs the recorded:

- suite id/version/fingerprint;
- dataset id/revision/split/population/fingerprint;
- task versions and fixture fingerprints;
- metric versions;
- seed policy;
- limits;
- code revision;
- model/checkpoint/config identity.

If any suite-defining metadata differs, the exact suite reference no longer matches and registration/resolution fails rather than silently combining results.

## Non-goals

#74 does not:

- implement the reasoning/code/memory/agent tasks themselves;
- run campaigns or retries;
- auto-promote a baseline;
- aggregate different suite versions as equivalent;
- require a network service;
- provide an artifact database.

Those belong to subsequent Research Platform issues.
