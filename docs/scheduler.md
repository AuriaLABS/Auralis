# Learning-rate scheduler

The scheduler is a versioned policy separate from Adam and from RunConfig v1.

## Schema 1

Example constant baseline:

\`\`\`text
auralis_scheduler=1
kind=constant
warmup_steps=0
decay_steps=0
min_lr_ratio=1
\`\`\`

Available policies:

- constant — historical Auralis behavior;
- linear_warmup — linearly reaches base LR on warmup_steps;
- cosine — optional warmup followed by cosine decay to min_lr_ratio.

Examples:

- [constant](../examples/scheduler-constant.cfg)
- [linear warmup](../examples/scheduler-warmup.cfg)
- [cosine](../examples/scheduler-cosine.cfg)

## Training

\`\`\`bash
cargo run --release -- train-fresh 20 auralis.bin \
  --scheduler-config examples/scheduler-warmup.cfg
\`\`\`

The optimizer update with global_step=0 is the first scheduled update.

For linear warmup with 4 steps and base LR 0.003, updates use:

\`\`\`text
0.00075, 0.00150, 0.00225, 0.00300, ...
\`\`\`

## Resume

Non-historical policies are stored in CHECKPOINT.scheduler.

On resume:

1. if the sidecar exists, Auralis auto-loads it;
2. a supplied --scheduler-config must match it exactly;
3. if an old checkpoint has no sidecar, only Constant is accepted;
4. LR is recomputed from RunConfig base LR + Adam global step, so resume does not depend on wall time or prior process state.

The sidecar is written only after checkpoint and manifest save succeed.

Without --scheduler-config, a fresh historical run remains Constant and creates no scheduler sidecar. This preserves the historical checkpoint/manifest contract.

Changing scheduler policy mid-run is rejected rather than silently creating a different experiment.
