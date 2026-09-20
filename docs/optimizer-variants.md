# Optimizer variants: AdamW and Lion

Issue #141 evaluates optimizer variants behind the common `Optimizer` boundary from #110.

These implementations are **experimental**. Adam remains the historical reference/default until a separate same-budget A/B evaluation justifies any promotion.

## Adam reference

Historical Adam remains unchanged:

- beta1 = 0.9
- beta2 = 0.999
- eps = 1e-8
- two state vectors: m and v
- no decoupled weight decay

A unit gate requires AdamW with `weight_decay=0` to reproduce Adam weights, m, v and global step bit-for-bit.

## AdamW

AdamW uses the same Adam moments and bias correction, plus explicit decoupled weight decay.

For parameter `p`, adaptive update `u`, learning rate `lr` and decay `wd`:

```text
p <- p - lr * (u + wd * p)
```

The decay term does not enter m/v.

`AdamWConfig` records beta1, beta2, eps and weight_decay and has a deterministic config fingerprint.

`AdamWState` schema 1 records:

- config;
- exact LR bits;
- global step;
- parameter count;
- m/v exact bits.

## Lion

Lion keeps one momentum vector.

For gradient `g` and momentum `m`:

```text
direction <- sign(beta1 * m + (1 - beta1) * g)
p <- p * (1 - lr * weight_decay)
p <- p - lr * direction
m <- beta2 * m + (1 - beta2) * g
```

For an exactly zero blended update, Auralis uses direction 0.

`LionConfig` records beta1, beta2 and weight_decay.

`LionState` schema 1 records:

- config;
- exact LR bits;
- global step;
- parameter count;
- m exact bits.

Lion therefore uses half the moment-vector bytes of Adam/AdamW for the same flat parameter count.

## Failure semantics

All variants reject:

- non-positive or non-finite learning rate;
- negative/non-finite weight decay;
- invalid beta/epsilon values;
- parameter/gradient/state length mismatch;
- global-step overflow;
- future state schemas;
- wrong optimizer kind;
- config-fingerprint mismatch;
- non-finite serialized state.

## Performance microbenchmark

`auralis_optimizer_variant_bench` measures only the update boundary on identical flat parameters/gradients.

It reports:

- ns/update;
- optimizer-state vector bytes;
- global step;
- final state fingerprint.

This benchmark does **not** measure model quality and cannot select a default optimizer.

Quality/loss/perplexity comparisons must use the same-token-budget Brain A/B harness from #34 after the active positional experiment is integrated.
