# Optimizer boundary

Brain/Train #110 separates optimizer policy/state from the training loop.

## Current supported optimizer

The reference implementation is Adam with:

- beta1 = 0.9
- beta2 = 0.999
- eps = 1e-8

These values are represented by AdamConfig. Its canonical encoding is schema 1 and has a deterministic config fingerprint.

## Training contract

Training receives dyn Optimizer and uses only:

- optimizer id;
- learning rate getter/setter;
- global optimizer step;
- parameter count;
- update over the canonical flat parameter/gradient slices;
- diagnostic state slices;
- config/state fingerprints.

The training loop does not read Adam fields or mutate Adam moments directly.

A tiny test executes the reference training loop through a non-Adam mock optimizer to gate this separation.

## Scheduler interaction

The scheduler from #95 computes the learning rate from the zero-based global step.

Training applies that value through Optimizer::set_learning_rate before the update. The scheduler therefore does not need direct access to Adam fields.

## Versioned optimizer state

AdamState uses:

- OPTIMIZER_STATE_SCHEMA_VERSION = 1
- optimizer kind;
- AdamConfig;
- exact learning-rate bits;
- global step;
- parameter count;
- exact m/v float bits.

Its text roundtrip is bit-for-bit and rejects:

- future state versions;
- config-fingerprint mismatch;
- parameter-count mismatch;
- negative/overflowing legacy step;
- non-finite moments;
- invalid learning rate or Adam hyperparameters.

OptimizerStateIdentity records a compact machine-readable identity:

- state schema;
- optimizer kind;
- config fingerprint;
- global step;
- parameter count;
- state fingerprint.

Training emits this identity at startup and after checkpoint persistence.

## AURLIS03 compatibility

AURLIS03 remains byte-format compatible. Its optimizer payload is still:

1. learning rate;
2. i32 optimizer step;
3. moment count;
4. m values;
5. v values.

Loading maps that legacy binary payload into the validated Adam state contract. Invalid legacy state returns InvalidData instead of reaching optimizer/model assertions.

No checkpoint magic/version bump is introduced by #110.

## Future parameter groups

The current canonical contract is one flat parameter group, matching Gpt::collect_params and all historical checkpoints. Future parameter-group support belongs behind the optimizer boundary; training should not duplicate optimizer-specific update logic to add it.
