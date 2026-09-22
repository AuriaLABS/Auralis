# ALiBi experiment

Issue #39 evaluates Attention with Linear Biases (ALiBi) against the historical learned-absolute positional baseline.

## Status

ALiBi is **experimental**. Learned absolute remains the default. RoPE remains a separate selectable experiment and is not combined with ALiBi here.

## Semantics

For attention head `h`, causal query position `i` and key position `j <= i`, Auralis computes the normal scaled dot-product score and adds:

```text
bias(h, i, j) = slope[h] * (j - i)
```

The current position receives zero bias. More distant past positions receive increasingly negative bias. Future positions are still excluded by the existing causal loop; ALiBi does not replace or relax the causal mask.

Per-head slopes use the standard deterministic ALiBi schedule, including the published non-power-of-two construction. ALiBi does not rotate Q/K and has no trainable positional parameters.

## Parameter-layout rule

The historical learned `pos_emb` table remains allocated and serialized under ALiBi, but is not added to token embeddings and receives zero gradient.

This deliberately keeps flat parameter count, AURLIS03 tensor order and checkpoint byte layout comparable with learned absolute and with the existing positional-experiment infrastructure.

## Architecture metadata

Architecture schema 5 records either:

```text
position=learned_absolute
```

or:

```text
position=alibi
attention_window=0
```

Schema 1 and 2 continue to migrate to `learned_absolute`. Legacy checkpoints without a `.architecture` sidecar cannot be reinterpreted as ALiBi on resume.

Example: `examples/architecture-alibi.cfg`.

## Reproducible A/B

```bash
cargo run --release --bin auralis_brain_ab -- 8 5 alibi --json
```

Both variants use the same model dimensions, LayerNorm, Adam, seed, token stream, clipping, batch/accumulation and optimizer-step budget. Brain A/B schema 4 already includes positional policy in experiment identity, so #39 does not require a schema bump.

## Context protocol

```bash
cargo run --release --bin auralis_alibi_context_bench -- 6 40 5
```

The benchmark trains the same 2-layer block=32 model under learned absolute and ALiBi, then measures contexts 4, 8, 16 and 32 in isolated child processes.

Each record reports loss and perplexity, logits tokens/s, peak RSS on Linux, parameter count, optimizer step and timing repetitions.

The experiment does **not** evaluate context beyond the configured block and therefore does not claim extrapolation.

## Promotion rule

#39 may integrate ALiBi as a selectable experiment if correctness, exact resume and resource gates pass. Changing the default requires a separate evidence-based decision. A single tiny-model quality result or hosted-runner timing is not sufficient.
