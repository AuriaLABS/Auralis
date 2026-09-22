# Model architecture configuration

Brain experiments select model depth/head/width, normalization and positional policy without changing the training/optimizer RunConfig contract.

## Contract

New architecture files use schema 5:

```text
auralis_architecture=5
n_embd=32
n_head=4
n_kv_head=4
attention_window=0
n_layer=2
block=32
n_ff=96
normalization=layernorm
position=learned_absolute
```

`n_kv_head` selects the K/V head layout: `n_kv_head=n_head` is the historical MHA default, a proper divisor greater than 1 selects GQA, and `n_kv_head=1` selects MQA. `attention_window=0` is historical dense causal attention; a positive value selects causal local attention over at most that many keys. Invalid KV-head layouts and windows above `block` fail closed.

Supported normalization policies:

- `layernorm` — historical Auralis baseline;
- `rmsnorm` — RMS normalization with learnable gamma. Historical beta slots remain reserved/inert so parameter layout and AURLIS03 tensor storage stay unchanged.


Supported positional policies:

- `learned_absolute` — historical Auralis baseline; adds the learned `pos_emb` table to token embeddings;
- `rope` — experimental rotary position encoding applied to Q/K per attention head. The historical `pos_emb` storage remains reserved and receives zero gradient so parameter count and AURLIS03 tensor layout remain comparable;
- `alibi` — experimental Attention with Linear Biases. Q/K are not rotated; a fixed per-head linear distance bias is added to causal attention logits before softmax. `pos_emb` remains reserved/inert for layout comparability.

RoPE currently obeys the configured `block`; #38 does not claim context extrapolation beyond that limit. RoPE requires an even per-head width. ALiBi also obeys the configured `block` in this experiment; #39 measures sensitivity only within that supported capacity.

Schema 1 files remain valid and migrate deterministically to `normalization=layernorm` + `position=learned_absolute` + MHA + dense attention. Schema 2 preserves normalization and migrates to learned-absolute + MHA + dense. Schema 3 preserves normalization/position and migrates to MHA + dense. Schema 4 preserves `n_kv_head` and migrates to `attention_window=0`. Future/unknown schemas and invalid policies fail closed.

The historical default is [`../examples/architecture-tiny.cfg`](../examples/architecture-tiny.cfg) and remains exactly equivalent to `Config::tiny(vocab)` + LayerNorm.

Examples:

- [`../examples/architecture-small-1x1.cfg`](../examples/architecture-small-1x1.cfg)
- [`../examples/architecture-small-3x2.cfg`](../examples/architecture-small-3x2.cfg)
- [`../examples/architecture-rmsnorm.cfg`](../examples/architecture-rmsnorm.cfg)
- [`../examples/architecture-rope.cfg`](../examples/architecture-rope.cfg)
- [`../examples/architecture-alibi.cfg`](../examples/architecture-alibi.cfg)

## Training

```bash
cargo run --release -- train-fresh 20 auralis.bin --model-config examples/architecture-rmsnorm.cfg
```

Without `--model-config`, training keeps the historical tiny LayerNorm + learned-absolute architecture.

For explicit architecture configs, Auralis writes `CHECKPOINT.architecture`. Resume, eval, chat and numeric-forward restore that metadata before executing the model. A mismatch aborts before overwriting checkpoint or manifest.

Legacy checkpoints without an architecture sidecar are interpreted as LayerNorm + learned-absolute + dense attention. Their MHA/GQA/MQA K/V-head layout is inferred from the validated parameter count; requesting RMSNorm, RoPE, ALiBi or a local attention window is rejected because those policies cannot be inferred safely.

## Invariants

- `n_embd` must be divisible by `n_head`;
- `n_kv_head` must be in `1..=n_head` and divide `n_head`;
- `attention_window` is `0` for dense or in `1..=block` for local causal attention;
- all dimensions are positive and bounded by checkpoint safety limits;
- LayerNorm remains the exact default path;
- RMSNorm keeps the same flat parameter layout and checkpoint tensor count;
- unknown/duplicate fields and future schemas fail closed;
- AURLIS02/AURLIS03 checkpoint magic is unchanged;
- normalization, positional identity, `n_kv_head` and `attention_window` are persisted/fingerprinted in the architecture sidecar;
- `RunConfig` keeps optimizer/data/training semantics separate from model architecture.

Changing normalization or positional policy is an architecture experiment, not a compatible silent reinterpretation of an existing checkpoint.
