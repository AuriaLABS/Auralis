# Model architecture configuration

Brain experiments select model depth/head/width and normalization without changing the training/optimizer RunConfig contract.

## Contract

New architecture files use schema 2:

```text
auralis_architecture=2
n_embd=32
n_head=4
n_layer=2
block=32
n_ff=96
normalization=layernorm
```

Supported normalization policies:

- `layernorm` — historical Auralis baseline;
- `rmsnorm` — RMS normalization with learnable gamma. Historical beta slots remain reserved/inert so parameter layout and AURLIS03 tensor storage stay unchanged.

Schema 1 files remain valid and migrate deterministically to `normalization=layernorm`. Future/unknown schemas and normalization values fail closed.

The historical default is [`../examples/architecture-tiny.cfg`](../examples/architecture-tiny.cfg) and remains exactly equivalent to `Config::tiny(vocab)` + LayerNorm.

Examples:

- [`../examples/architecture-small-1x1.cfg`](../examples/architecture-small-1x1.cfg)
- [`../examples/architecture-small-3x2.cfg`](../examples/architecture-small-3x2.cfg)
- [`../examples/architecture-rmsnorm.cfg`](../examples/architecture-rmsnorm.cfg)

## Training

```bash
cargo run --release -- train-fresh 20 auralis.bin --model-config examples/architecture-rmsnorm.cfg
```

Without `--model-config`, training keeps the historical tiny LayerNorm architecture.

For explicit architecture configs, Auralis writes `CHECKPOINT.architecture`. Resume, eval, chat and numeric-forward restore that metadata before executing the model. A mismatch aborts before overwriting checkpoint or manifest.

Legacy checkpoints without an architecture sidecar are interpreted as LayerNorm only. Requesting RMSNorm for such a checkpoint is rejected because the normalization policy cannot be inferred safely.

## Invariants

- `n_embd` must be divisible by `n_head`;
- all dimensions are positive and bounded by checkpoint safety limits;
- LayerNorm remains the exact default path;
- RMSNorm keeps the same flat parameter layout and checkpoint tensor count;
- unknown/duplicate fields and future schemas fail closed;
- AURLIS02/AURLIS03 checkpoint magic is unchanged;
- normalization identity is persisted/fingerprinted in the architecture sidecar;
- `RunConfig` keeps optimizer/data/training semantics separate from model architecture.

Changing normalization is an architecture experiment, not a compatible silent reinterpretation of an existing checkpoint.
