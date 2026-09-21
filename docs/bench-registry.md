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


## Brain recurrent-reasoning benchmark

The `recurrent-reasoning` entry runs:

```
cargo run --release --bin auralis_recurrent_reasoning_bench
```

It compares reasoning steps 1, 2 and 3 with a fixed 12 physical block-application budget, identical model parameter count, seed, optimizer policy and token stream. It reports optimizer updates, tokens seen, eval loss/perplexity, training and inference latency, gradient norm and final-state fingerprint. The curve is descriptive evidence; the benchmark does not auto-promote a recurrent depth.


## Brain KV-cache benchmark

The `kv-cache` entry runs:

```
cargo run --release --bin auralis_kv_cache_bench
```

It compares growing-prefix autoregressive decode with full uncached prefix recomputation against one-token cached decode at contexts 8, 16, 32 and 64. It reports exact cached/uncached equivalence, total and per-token latency, speedup, active/allocated cache bytes and fixed model parameter count. Hosted-runner timing is descriptive; the longest-context advantage is the performance gate for #40.


## Brain GQA/MQA benchmark

The `gqa-mqa` entry runs:

```
cargo run --release --bin auralis_gqa_mqa_bench
```

It compares MHA (`n_kv_head=n_head`), GQA and MQA under the same seed, token stream, optimizer and training protocol, then measures inference at contexts 8, 16 and 32. It reports eval loss/perplexity, training throughput, parameter count, compact K/V width, full/cache prefill latency, uncached/cached next-token latency and active KV-cache bytes. The benchmark is descriptive evidence for #140; it does not auto-promote GQA or MQA over the MHA default.
