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


## Brain local-attention benchmark

The `local-attention` entry runs:

```
cargo run --release --bin auralis_local_attention_bench
```

It compares dense causal attention with local windows 8 and 16 under the same seed, token stream, optimizer and parameter budget. It reports quality A/B plus context-length scaling at 8, 16, 32 and 64 tokens, median forward/backward latency, actual compact probability slots/bytes, finite loss and finite gradients. Dense attention remains the reference/default; timing and quality are descriptive evidence and do not auto-promote the local variant.


## Brain MoE-router benchmark

The `moe-router` entry runs:

```
cargo run --release --bin auralis_moe_router_bench
```

It measures deterministic top-k routing plus dispatch/gather against a dense identity-copy reference for multiple token/expert/top-k shapes. A forced saturation case exercises the explicit `dense-fallback` policy and verifies that every token is either routed or falls back, `dropped_tokens=0`, expert capacity is never exceeded, and identity-expert gather reconstructs the dense input within floating-point tolerance. Hosted-runner timing is descriptive evidence for #112; this benchmark does not claim a model-quality or end-to-end speed improvement.


## Research deterministic reasoning-suite benchmark

The `reasoning-suite` entry runs:

```
cargo run --release --bin auralis_reasoning_suite_bench -- both
```

It generates the versioned #130 smoke and full profiles, verifies deterministic fixture fingerprints, scores the exact oracle, and runs an intentional regression that must surface arithmetic/value, sequence-length, variable-binding, finite-state transition and distractor-capture failures separately. It also publishes generation/scoring timing and the train-vs-eval difficulty/length boundaries. Timing is descriptive; determinism, objective scoring and regression sensitivity are the gates.


## Research external-memory evaluation suite

The `memory-suite` entry runs:

```
cargo run --release --bin auralis_memory_suite_bench -- both
```

It executes the #135 deterministic smoke/full external-memory profiles over exact retrieval, temporal ordering, conflict/stale resolution and long-horizon recall. It also publishes the memory-off control plus reject-new capacity sweeps with retrieval ratio, query latency, heap bytes, snapshot bytes and observed evictions. Correctness and explicit capacity semantics are gates; hosted-runner latency is descriptive.


## Research executable code evaluation suite

The `code-suite` entry runs:

```
cargo run --release --bin auralis_code_suite_bench -- both
```

It executes #131 standalone Rust fixtures through a confined single-expression patch dialect, `rustc --test` and protected tests. It publishes exact/compile/test pass ratios, compiler/test latency, failure taxonomy and compiler-error classes. Timing is descriptive; successful patch application, compilation and protected tests are the gate.


## Research integrated capability battery

The `capability-suite` entry runs:

```
cargo run --release --bin auralis_capability_suite_bench -- smoke
```

It composes the #130/#131/#135 suites plus #47 agent mock-replay evaluation under #75, reports each capability separately, and proves isolated reasoning or agent regressions cannot be hidden by exact sibling capabilities. The descriptive mean is published and is not a gate.
