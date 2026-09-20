# RoPE experiment

Issue #38 evaluates rotary position encoding against the historical learned-absolute baseline.

## Status

RoPE is **experimental**. Learned absolute remains the default.

## Semantics

For each token position and attention head, RoPE rotates adjacent Q/K pairs using:

```text
theta(position, pair) = position / 10000^(2*pair/head_width)
```

The same rotation is applied to Q and K before causal attention. Backward applies the inverse rotation to dQ/dK before gradients propagate into Wq/Wk and the residual stream.

Position zero is identity. RoPE requires an even per-head width.

## Parameter-layout rule

The historical learned `pos_emb` table remains allocated and serialized under RoPE, but is not added to token embeddings and receives zero gradient.

This deliberately keeps:

- flat parameter count;
- AURLIS03 tensor order;
- checkpoint byte layout;

comparable with the learned-absolute baseline.

## Architecture metadata

Architecture schema 3 records:

```text
position=learned_absolute
```

or:

```text
position=rope
```

Schema 1 and 2 migrate to `learned_absolute`.

Legacy checkpoints without a `.architecture` sidecar cannot be reinterpreted as RoPE on resume.

## Reproducible A/B

```bash
cargo run --release --bin auralis_brain_ab -- 8 5 rope --json
```

Both variants use the same model dimensions, LayerNorm, seed, token stream, optimizer, clipping, batch/accumulation and step budget.

## Context protocol

```bash
cargo run --release --bin auralis_rope_context_bench -- 6 40 5
```

The benchmark trains the same 2-layer block=32 model under each positional policy, then measures contexts 4, 8, 16 and 32 in isolated child processes.

Each record reports:

- loss and perplexity;
- logits tokens/s;
- peak RSS on Linux;
- parameter count;
- optimizer step;
- timing repetitions.

The experiment does **not** evaluate context beyond the configured block and therefore does not claim extrapolation.

## Promotion rule

#38 may integrate RoPE as a selectable experiment if correctness, resume and cost gates pass. Changing the default requires a separate evidence-based decision. A single hosted-runner throughput result is not sufficient.
