# Positional encoding baseline

Brain #35 isolates positional encoding from the transformer block without promoting a new positional method.

## Current implementation

The supported reference is **learned absolute positional embeddings**.

The contract in `position.rs` owns:

- adding learned position vectors to token embeddings;
- validating context capacity and activation shape;
- accumulating gradients into positional parameters.

Attention does not implement positional semantics.

The learned-absolute helper preserves the historical nested-loop arithmetic exactly in forward and backward tests.

## Extension rule

A future positional experiment must be a separate candidate. RoPE/ALiBi are **not** implemented by #35.

The current interface establishes the reference semantics and a clean place to extend or revise the contract when a non-additive method needs Q/K transforms.

## Context baseline

`auralis_position_context_bench` uses one fixed `Config::tiny(100)` model with block=32 and seed 659918. Contexts 4, 8, 16 and 32 therefore use the same weights.

Each context runs in an isolated child process and reports:

- median logits time;
- tokens/s;
- Linux peak RSS when available;
- input activation bytes;
- logits output bytes;
- learned-position capacity bytes;
- deterministic eval loss/perplexity on the corresponding prefix.

```bash
cargo run --release --bin auralis_position_context_bench -- 30 5
```

The benchmark is a baseline, not a claim that longer contexts improve quality. Loss/perplexity are recorded to make future positional A/B experiments traceable.

A context beyond the configured block is rejected explicitly by the positional contract.
