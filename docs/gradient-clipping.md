# Gradient clipping policy

Auralis uses clip-by-global-L2-norm after gradient accumulation normalization and before the optimizer update.

## RunConfig schema 2

The persisted fields are:

```text
grad_clip_norm=1
grad_clip_enabled=true
```

The historical behavior remains the default: clipping enabled with threshold 1.0.

RunConfig schema 1 had only `grad_clip_norm`. Loading schema 1 migrates explicitly to:

```text
grad_clip_enabled=true
```

A schema-1 manifest can resume only with clipping enabled. Clipping OFF cannot reuse a schema-1 fingerprint because that policy did not exist in v1.

## Disable clipping explicitly

Use [`../examples/clipping-off.cfg`](../examples/clipping-off.cfg):

```bash
cargo run --release -- train-fresh 20 /tmp/auralis-off.bin --config examples/clipping-off.cfg
```

OFF is persisted and fingerprinted in RunConfig v2. The training kernel maps OFF to a non-triggering internal threshold while still computing the pre-gradient norm for observability.

## Metrics

Each optimizer step exposes:

- `grad_norm_before_clip`;
- `grad_norm_after_clip`;
- `grad_scale`;
- `clip_applied`.

If clipping is disabled, or the norm is below the threshold:

- post norm equals pre norm;
- scale is 1;
- clip_applied is false.

When clipping triggers, the gradients are scaled once and the post norm is measured from the actual scaled buffer.

NaN/Inf in loss or pre-clip gradient norm aborts before optimizer update.

## Resume

RunConfig v2 fingerprint includes both `grad_clip_enabled` and `grad_clip_norm`.

Changing either policy field on resume is an experiment mismatch and is rejected by the manifest guard.

Historical manifest v4 + RunConfig schema 1 remains resumable only under the historical enabled policy.
