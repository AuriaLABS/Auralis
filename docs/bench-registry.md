# Benchmark registry

`auralis bench list` catalogs reproducible Engine and Brain benchmarks. Listing a benchmark does not run it.

```
auralis bench list
auralis bench describe engine
auralis bench list --json
auralis bench list --csv
auralis bench describe engine --json
```

`--json` and `--csv` are exclusive. Library helpers:

- `bench::list_json()` / `bench::list_csv()`
- `bench_format::render(command, id, format)`

To add a benchmark:

1. Keep the existing `src/bin/auralis_*_bench.rs` (or profile bin).
2. Append a `BenchSpec` in `src/bench.rs`.
3. Document default args used by CI in `genesis.yml`.
4. Do not change kernel math in the same PR as the registry.

Running a listed bench stays:

```
cargo run --release --bin auralis_engine_bench -- 3 20 5
```


## Brain external-memory contract benchmark

The `external-memory` entry runs:

```
cargo run --release --bin auralis_memory_bench -- 128 40
```

It measures exact retrieval, explicit capacity degradation, per-query latency, estimated heap bytes, snapshot bytes and session-contamination count for the #37 reference backend. Timing is descriptive; correctness/capacity semantics are the gate.


## Brain model-memory integration benchmark

The `memory-model` entry runs:

```
cargo run --release --bin auralis_memory_model_bench -- 100 7
```

It compares the exact memory-off baseline with opt-in retrieval + `last_hidden_mean_add` fusion on a controlled synthetic task, and publishes on/off latency, query trace, heap estimate and snapshot size. It also proves inference leaves the #37 memory snapshot unchanged. The benchmark is evidence for #42; it does not promote memory to the default model path.
