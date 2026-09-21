# Brain A/B experiment harness

`auralis_brain_ab` is the reproducible architecture comparison harness for Brain 0.4.

It is deliberately a **measurement tool**, not an automatic architecture selector.

## Protocol

Both variants share:

- seed;
- synthetic token stream and token fingerprint;
- learning rate;
- gradient clipping;
- batch size;
- gradient accumulation;
- optimizer steps;
- repetition count.

Order alternates A-first/B-first between repetitions.

Each variant records normalization, positional policy and optimizer identity. All three are included in the final state fingerprint, because identical weight bytes interpreted under different execution semantics or optimizer state are not the same experiment state.

Each measurement records:

- complete model config;
- normalization policy;
- positional policy;
- optimizer identity;
- final training loss;
- eval loss and perplexity;
- tokens/s;
- parameter count and parameter bytes;
- optimizer state-vector bytes;
- canonical serialized optimizer-state bytes;
- checkpoint bytes;
- whether the checkpoint contains the legacy Adam payload;
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

## Learned absolute vs ALiBi

```bash
cargo run --release --bin auralis_brain_ab -- 8 5 alibi
cargo run --release --bin auralis_brain_ab -- 8 5 alibi --json
```

This scenario holds vocab, width, heads, depth, block, FF width, normalization, optimizer, seed, token stream and training budget constant. Only positional policy changes.

ALiBi leaves Q/K values unrotated and adds a fixed per-head linear distance bias to causal attention logits. The historical learned-position parameter slots remain reserved/inert, so parameter count and raw AURLIS03 tensor layout stay directly comparable. The harness reports both variants and does **not** promote ALiBi automatically.

## Optimizer comparisons

```bash
cargo run --release --bin auralis_brain_ab -- 8 5 optimizer-adamw
cargo run --release --bin auralis_brain_ab -- 8 5 optimizer-lion
```

These scenarios hold architecture, LayerNorm, learned-absolute position, seed, token stream, learning rate, clipping, batch, accumulation and optimizer-step budget constant.

- `optimizer-adamw`: historical Adam vs AdamW with `weight_decay=0.01`.
- `optimizer-lion`: historical Adam vs Lion with `weight_decay=0.01`.

This first comparison deliberately uses the **same learning rate** for all optimizers. It is an algorithm-isolation baseline, not a hyperparameter-tuned leaderboard.

Adam keeps the historical AURLIS03 optimizer payload in the checkpoint. AdamW and Lion do not pretend to be Adam inside AURLIS03: their canonical schema-1 optimizer state is reported separately, while the model checkpoint remains loadable through the existing checkpoint format. The JSON field `checkpoint_includes_optimizer` makes this distinction explicit.

Lion uses one moment vector while Adam/AdamW use two; `optimizer_state_bytes` reports that difference directly.

The harness reports all variants. It does **not** declare a winner from a noisy hosted-runner timing or a single metric.
