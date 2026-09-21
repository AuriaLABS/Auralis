# Recurrent reasoning experiment

Issue #41 evaluates **shared-weight recurrent computation**. It does not add hidden parameters and it does not expose or require textual chain-of-thought.

## Hypothesis

The same physical transformer block stack can be applied more than once before the final normalization/output projection.

If the model has `L` physical layers and uses `R` reasoning steps, a forward pass performs `R * L` block applications while parameter count remains unchanged.

## Baseline invariant

`RecurrentConfig { reasoning_steps: 1 }` is defined as the existing transformer.

The recurrent public APIs explicitly delegate one-step calls to the normal model APIs:

- `logits_recurrent(..., 1)` -> baseline logits;
- `loss_recurrent(..., 1)` -> baseline loss;
- `backward_recurrent_into(..., 1)` -> baseline backward.

CI requires bit-for-bit equality of logits, loss and gradients.

## Multi-step forward

Token and learned-absolute input position embeddings are applied once.

For each recurrent step, the complete physical block stack is reapplied with the **same weights**. Q/K positional transforms such as RoPE or ALiBi still execute at each block application under their existing contracts.

Final normalization and output projection execute once after the last recurrent step.

## Shared-weight backward

Training forward caches every block application in execution order.

Backward walks those caches in reverse. Cache index modulo the physical layer count selects the shared parameter block.

Bias and normalization gradients were already additive. Weight gradients for recurrent block uses are accumulated with the scalar reference `A^T * dY` contract, so each use contributes to the same physical gradient tensor without relying on the zeroed-output Engine kernel.

One-step training does not use this path and remains the exact optimized baseline.

## Gradient verification

CI requires:

- one-step exact baseline;
- finite gradients for two and three recurrent steps;
- recurrent eval/backward loss agreement;
- finite-difference checks at selected token embedding, position embedding, Q projection and output projection parameters.

## Fixed-compute comparison

`auralis_recurrent_reasoning_bench` reuses the `AbProtocol` budget fields from the Brain A/B harness.

The tiny comparison fixes:

- identical parameter count;
- seed;
- optimizer and learning rate;
- clipping threshold;
- token stream;
- **12 total physical block applications** per run.

With one physical layer:

- R=1 -> 12 optimizer updates;
- R=2 -> 6 optimizer updates;
- R=3 -> 4 optimizer updates.

This is an explicit compute-budget normalization. It also reports tokens seen because reducing update count is part of the tradeoff.

For every recurrent depth the benchmark publishes:

- eval loss and perplexity;
- training wall time;
- ns per physical block application;
- 100-forward inference latency;
- parameter count;
- optimizer updates;
- tokens seen;
- full final parameter fingerprint;
- maximum pre-clipping gradient norm.

Repetitions must reproduce final state/loss exactly. Wall-clock is descriptive and may vary.

Run:

```bash
cargo run --release --bin auralis_recurrent_reasoning_bench
```

## Interpretation

The curve is evidence, not an automatic promotion rule.

A recurrent depth is not considered better merely because its loss is lower, nor worse merely because it is slower. The evidence must be read as quality versus compute, data exposure and latency under a fixed parameter budget.

The default model remains one reasoning step.
