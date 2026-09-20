# Model architecture configuration

Brain experiments can select model depth/head/width without changing the historical `RunConfig` training schema.

## Contract

Architecture files use schema 1:

```text
auralis_architecture=1
n_embd=32
n_head=4
n_layer=2
block=32
n_ff=96
```

The historical default is [`../examples/architecture-tiny.cfg`](../examples/architecture-tiny.cfg) and is exactly equivalent to `Config::tiny(vocab)`.

Examples:

- [`../examples/architecture-small-1x1.cfg`](../examples/architecture-small-1x1.cfg)
- [`../examples/architecture-small-3x2.cfg`](../examples/architecture-small-3x2.cfg)

## Training

```bash
cargo run --release -- train-fresh 20 auralis.bin --model-config examples/architecture-small-3x2.cfg
```

Without `--model-config`, training keeps the historical tiny architecture.

For a new/fresh checkpoint, the selected architecture controls model construction. For resume, the checkpoint remains authoritative. Supplying `--model-config` during resume acts as a compatibility assertion: a mismatch aborts before training and before overwriting the checkpoint or manifest.

## Invariants

- `n_embd` must be divisible by `n_head`;
- all dimensions are positive and bounded by the same safety limits used by checkpoint loading;
- unknown/duplicate fields and future schema versions fail closed;
- checkpoints AURLIS02/AURLIS03 already persist vocab, embedding width, heads, layers, block and FF width;
- manifests already record the complete architecture and validate it on resume;
- `RunConfig` remains schema 1 and keeps optimizer/data/training semantics separate from model architecture.

Changing architecture is an experiment, not a compatible resume of an existing checkpoint.
