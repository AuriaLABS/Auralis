# External-memory evaluation suite

Issue #135 adds an objective evaluation suite for the #37 external-memory contract and #42 model-memory boundary. This slice evaluates the **external-memory profile only**. Session-memory policy from #98 is intentionally out of scope.

## Contract

```text
MEMORY_SUITE_SCHEMA_VERSION = 1
MEMORY_SUITE_SEED = 135659918
backend = external-memory schema 1
capacity policy = reject-new
eviction = none
```

The suite never uses an LLM judge. Fixtures, retrieval policy, scoring and failure classification are deterministic.

## Profiles

- `smoke`: 2 fixtures per task family, 10 cases total.
- `full`: 12 fixtures per task family, 60 cases total.

Both profiles use the exact #74 evaluation registry, exact fixture fingerprints and one pinned seed.

## Task families

### Exact retrieval

One authoritative record is queried by exact namespace/key while unrelated records are present. The exact stored scalar must be returned.

### Temporal order

The same key receives an ordered history of values. The suite queries all matching records and defines **latest-record-wins** by monotonically increasing record ID. Selecting an older record is `wrong-order`.

### Conflict resolution

Two different values exist for the same key in the evaluated namespace, while another namespace contains a distractor with the same key. The latest matching record in the requested namespace is authoritative. Capturing the older conflicting value is `conflict-capture`.

### Stale resolution

An old value is written, unrelated writes advance the history, then the target is refreshed. Returning the pre-refresh value is `stale-value`.

### Long-horizon recall

The target is written before a long series of unrelated writes and must remain exactly retrievable. The full profile spans horizons from at least 32 to at least 120 intervening writes. The benchmark emits one `memory_suite_horizon` row per long-horizon fixture with exact-match, query latency, heap bytes and snapshot bytes, so results are visible at each evaluated horizon rather than only as one aggregate.

## Objective scoring

Each prediction records:

- selected scalar value;
- matching record IDs;
- selected record ID.

A correct temporal/conflict/stale answer must select the last matching ID. The failure taxonomy is:

- `missing`;
- `wrong-value`;
- `wrong-order`;
- `conflict-capture`;
- `stale-value`.

The intentional regression fixture forces every failure class to appear. This is a sensitivity gate, not a model-quality claim.

## Memory-off baseline

Every case is also scored with memory disabled. Because these fixtures are intentionally memory-dependent, the expected memory-off exact-match is **0.0**. This keeps retrieval failure separate from reasoning or language-generation quality.

## Capacity and eviction behavior

The #37 reference backend has an explicit **reject-new** policy at capacity; it does not evict existing records. #135 therefore does not invent LRU/FIFO semantics.

The benchmark sweeps:

- smoke: capacities 4, 8, 16 over 16 writes;
- full: capacities 8, 32, 128 over 128 writes.

For every point it reports:

- stored/rejected writes;
- exact retrieval ratio;
- retained-prefix hits;
- observed evictions, which must remain zero;
- query latency;
- estimated heap bytes;
- serialized snapshot bytes.

At the largest capacity, retrieval must reach 1.0. Smaller capacities degrade explicitly because new writes are rejected, not because older records silently disappear.

## Registry metrics

The suite registers 12 metrics:

- global exact-match;
- memory-off exact-match;
- long-horizon exact-match;
- capacity retrieval ratio;
- query latency;
- heap bytes;
- snapshot bytes;
- five failure counts.

Latency and footprint are descriptive measurements. Correctness, memory-off separation and capacity semantics are gates.

## Running

```bash
cargo run --release --bin auralis_memory_suite_bench -- smoke
cargo run --release --bin auralis_memory_suite_bench -- full
cargo run --release --bin auralis_memory_suite_bench -- both
```

## Non-goals

#135 does not:

- define #98 session-memory semantics;
- change the #37 backend or its capacity policy;
- add eviction to a backend that currently rejects new writes;
- evaluate free-form reasoning quality;
- promote memory-on as a model default;
- use external services.
