# Model architecture configuration

Brain experiments select model depth/head/width, normalization and positional policy without changing the training/optimizer RunConfig contract.

## Contract

New architecture files use schema 3:

```text
auralis_architecture=3
n_embd=32
n_head=4
n_layer=2
block=32
n_ff=96
normalization=layernorm
position=learned_absolute
```

Supported normalization policies:

- `layernorm` — historical Auralis baseline;
- `rmsnorm` — RMS normalization with learnable gamma. Historical beta slots remain reserved/inert so parameter layout and AURLIS03 tensor storage stay unchanged.


Supported positional policies:

- `learned_absolute` — historical Auralis baseline; adds the learned `pos_emb` table to token embeddings;
- `rope` — experimental rotary position encoding applied to Q/K per attention head. The historical `pos_emb` storage remains reserved and receives zero gradient so parameter count and AURLIS03 tensor layout remain comparable.

RoPE currently obeys the configured `block`; #38 does not claim context extrapolation beyond that limit. RoPE requires an even per-head width.

Schema 1 files remain valid and migrate deterministically to `normalization=layernorm` + `position=learned_absolute`. Schema 2 preserves its explicit normalization and migrates to `position=learned_absolute`. Future/unknown schemas, normalization values and positional values fail closed.

The historical default is [`../examples/architecture-tiny.cfg`](../examples/architecture-tiny.cfg) and remains exactly equivalent to `Config::tiny(vocab)` + LayerNorm.

Examples:

- [`../examples/architecture-small-1x1.cfg`](../examples/architecture-small-1x1.cfg)
- [`../examples/architecture-small-3x2.cfg`](../examples/architecture-small-3x2.cfg)
- [`../examples/architecture-rmsnorm.cfg`](../examples/architecture-rmsnorm.cfg)
- [`../examples/architecture-rope.cfg`](../examples/architecture-rope.cfg)

## Training

```bash
cargo run --release -- train-fresh 20 auralis.bin --model-config examples/architecture-rmsnorm.cfg
```

Without `--model-config`, training keeps the historical tiny LayerNorm + learned-absolute architecture.

For explicit architecture configs, Auralis writes `CHECKPOINT.architecture`. Resume, eval, chat and numeric-forward restore that metadata before executing the model. A mismatch aborts before overwriting checkpoint or manifest.

Legacy checkpoints without an architecture sidecar are interpreted as LayerNorm + learned-absolute only. Requesting RMSNorm or RoPE for such a checkpoint is rejected because those policies cannot be inferred safely.

## Invariants

- `n_embd` must be divisible by `n_head`;
- all dimensions are positive and bounded by checkpoint safety limits;
- LayerNorm remains the exact default path;
- RMSNorm keeps the same flat parameter layout and checkpoint tensor count;
- unknown/duplicate fields and future schemas fail closed;
- AURLIS02/AURLIS03 checkpoint magic is unchanged;
- normalization and positional identity are persisted/fingerprinted in the architecture sidecar;
- `RunConfig` keeps optimizer/data/training semantics separate from model architecture.

Changing normalization or positional policy is an architecture experiment, not a compatible silent reinterpretation of an existing checkpoint.
