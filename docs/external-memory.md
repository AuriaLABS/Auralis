# Experimental external memory

Issue #37 defines a **model-independent** memory boundary for future Brain experiments. It does not attach memory to the transformer and it does not change any default model behavior.

## Status and invariant

External memory is experimental. Auralis continues to run completely without it.

The reference implementation is InMemoryExternalMemory in src/memory.rs. Future model integration belongs to #42 and must remain opt-in and A/B measured.

## Contract

ExternalMemory exposes the minimum operations requested by the experiment:

- write — append one validated record and return a monotonic numeric ID;
- read — retrieve one record by ID;
- query — deterministically filter by exact namespace, exact key and/or memory class, with an explicit result limit;
- reset — clear session, one class, one namespace, or all records;
- version — report EXTERNAL_MEMORY_SCHEMA_VERSION = 1.

A record contains:

- monotonic id;
- UTF-8 namespace;
- UTF-8 key;
- finite Vec<f32> value;
- MemoryClass.

There is intentionally no approximate nearest-neighbour search, vector database dependency, autonomous write policy or model coupling in this issue.

## Memory classes

### Session

session is ephemeral. It can be cleared with MemoryReset::Session and is **never serialized** into a memory snapshot. This is the reference mechanism used by the contamination fixture.

### Persistent

persistent survives snapshot encode/decode and file save/load. It represents user/application state that a future integration may choose to retrieve.

### Trainable

trainable is persisted and explicitly marked as a future optimization target, but #37 does **not** attach gradients or an optimizer to it. The class is metadata only until a separately measured model experiment defines update semantics.

## Capacity policy

Capacity is measured in records and must be in 1..=1_000_000.

The reference backend uses **reject-new** semantics. When full, write returns MemoryError::AtCapacity; it never silently evicts or overwrites an older record.

That choice makes degradation auditable: when a working set is larger than capacity, misses are caused by explicit rejected writes rather than a hidden eviction heuristic.

Per-record namespace, key and value lengths are bounded. Snapshot files are also bounded before parsing.

## Persistence format

Schema 1 is deterministic text:

    auralis_external_memory=1
    capacity=<records>
    next_id=<u64>
    persisted_count=<records>
    record=<id>;<class>;<namespace-hex>;<key-hex>;<f32-bit-list>
    ...
    checksum=<fnv64>

Important rules:

- only persistent and trainable records are serialized;
- float values are serialized by exact IEEE-754 bits;
- unknown/future schema versions fail closed;
- malformed records, duplicate IDs, non-finite values, count mismatches and checksum corruption fail closed;
- next_id is preserved so a loaded store does not reuse prior IDs;
- the checksum covers every preceding byte of the canonical snapshot body.

save and load are convenience helpers around the same canonical format.

## Determinism and contamination fixtures

Unit tests pin:

- identical operation sequences produce identical stores and snapshots;
- exact read/query behavior;
- full-capacity rejection without mutation;
- session reset removes session state while preserving persistent/trainable state;
- snapshots exclude session records;
- persistence round-trips float bits exactly;
- future-version and checksum-corrupted snapshots are rejected.

## Synthetic benchmark

Run:

    cargo run --release --bin auralis_memory_bench -- 128 40

The benchmark emits one memory_protocol line and three memory_capacity lines for capacities 8, 32 and 128.

It reports:

- exact retrieval hits and ratio over the same working set;
- rejected writes;
- query nanoseconds per operation;
- estimated heap bytes;
- canonical snapshot bytes;
- cross-session contamination count;
- restored persistent/trainable record count and snapshot fingerprint.

The capacity experiment is intentionally simple: with reject-new semantics and a 128-record working set, exact retrieval should increase monotonically as capacity increases and reach 1.0 at capacity 128.

Timing is descriptive runner evidence, not a promotion gate.

## Future model integration (#42)

A future adapter should keep retrieval separate from representation fusion:

    model input/state
          |
    optional retrieval request
          v
    ExternalMemory::query
          |
    retrieved values + trace
          |
    explicit fusion policy
          v
    model computation

Required #42 invariants remain:

- memory=off is baseline-equivalent;
- empty memory is defined behavior;
- reads/writes are traceable;
- the persistence format remains this contract unless migrated explicitly;
- usefulness is measured on a predeclared synthetic task through #34;
- latency and memory cost are published alongside quality.

#37 itself makes no claim that external memory improves reasoning.
