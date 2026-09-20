# Brain A/B experiment harness

`auralis_brain_ab` is the reproducible architecture comparison harness for Brain 0.4.

It is deliberately a **measurement tool**, not an automatic architecture selector.

## Protocol

Both variants share:

- seed;
- synthetic token stream and token fingerprint;
- optimizer;
- learning rate;
- gradient clipping;
- batch size;
- gradient accumulation;
- optimizer steps;
- repetition count.

Order alternates A-first/B-first between repetitions.

Each variant records normalization and positional policy. Both policies are included in the final state fingerprint, because identical weight bytes interpreted under different execution semantics are not the same model state.

Each measurement records:

- complete model config;
- normalization policy;
- positional policy;
- final training loss;
- eval loss and perplexity;
- tokens/s;
- parameter count and parameter bytes;
- Adam state bytes;
- real AURLIS03 checkpoint bytes;
- final state fingerprint.

The experiment record also contains the build commit/revision.

## Reproducibility control

```bash
cargo run --release --bin auralis_brain_ab -- 4 3 same
```

The `same` scenario uses identical LayerNorm A/B configs. Deterministic state, losses, checkpoint size and fingerprints must match across A and B and across repetitions. Wall-clock throughput is not required to be identical.

## Depth comparison

```bash
cargo run --release --bin auralis_brain_ab -- 8 5 depth
cargo run --release --bin auralis_brain_ab -- 8 5 depth --json
```

The `depth` scenario compares 1-layer vs 2-layer LayerNorm models under the same training budget.

## LayerNorm vs RMSNorm

```bash
cargo run --release --bin auralis_brain_ab -- 8 5 normalization
cargo run --release --bin auralis_brain_ab -- 8 5 normalization --json
```

This scenario holds vocab, width, heads, depth, block, FF width, seed, data and training budget constant. Only normalization changes.

RMSNorm keeps the same parameter count and AURLIS03 checkpoint byte size by reserving the historical beta slots with zero gradient.

## Learned absolute vs RoPE

```bash
cargo run --release --bin auralis_brain_ab -- 8 5 rope
cargo run --release --bin auralis_brain_ab -- 8 5 rope --json
```

This scenario holds vocab, width, heads, depth, block, FF width, normalization, seed, data and training budget constant. Only positional policy changes.

RoPE keeps the historical learned-position parameter slots reserved/inert, so parameter count and raw AURLIS03 tensor layout remain directly comparable. The harness reports both variants and does **not** promote RoPE automatically.

The harness reports all variants. It does **not** declare a winner from a noisy hosted-runner timing or a single metric.
