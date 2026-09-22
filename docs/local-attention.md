# Causal local attention experiment

Issue #111 adds a selectable causal local/window attention variant for Brain experiments. Dense causal attention remains the reference and default.

## Policy

The architecture field `attention_window` controls the pattern. A ready-to-run example is [`../examples/architecture-local-w8.cfg`](../examples/architecture-local-w8.cfg):

- `attention_window=0`: historical dense causal attention;
- `attention_window=W>0`: query position `i` may attend only keys in
  `max(0, i + 1 - W)..=i`.

Future tokens are never visible. The local pattern is therefore causal by construction and the left boundary is explicit for every query row.

Architecture schema 5 persists `attention_window`. Schemas 1-4 migrate deterministically to dense attention. A positive window larger than `block` fails closed.

## Dense reference and reversibility

The dense path is not rewritten. When `attention_window=0`, the model continues through the historical dense attention functions, preserving the existing reference/default behavior.

Local attention is opt-in and reversible. Setting the window back to zero restores the dense policy with unchanged parameters.

## MHA / GQA / MQA

The window policy is orthogonal to K/V head sharing:

- MHA keeps `n_kv_head=n_head`;
- GQA uses a proper divisor of `n_head`;
- MQA uses `n_kv_head=1`.

The local kernels operate on `GroupedAttentionShape`, so the same causal window semantics apply to all three layouts.

## Forward and backward storage

Evaluation computes only keys inside the active window.

Training uses compact probability storage rather than allocating a dense `heads * tokens * tokens` matrix. Per head, the number of stored probabilities is:

```text
sum(i=0..tokens-1) min(i + 1, window)
```

For dense attention the historical full probability matrix remains unchanged. `Gpt::attention_probability_slots` and `attention_probability_bytes` report the actual model contract used by the forward/backward cache.

Backward reads the compact rows and accumulates Q/K/V gradients only over keys that were present in the forward local mask. Targeted finite-difference checks cover Q/K/V projection parameters.

## Cached decode and KV cache

The KV cache retains complete historical K/V rows; this first local-attention slice does **not** evict old K/V state. Cached decode scans only the active trailing window.

KV-cache schema 3 includes `attention_window` in session identity. A cache created under dense attention, another local window, or an incompatible MHA/GQA/MQA layout is rejected instead of being silently reinterpreted.

This deliberately separates compute sparsity from future sliding-window eviction/paged-storage work.

## A/B quality protocol

`run_attention_window_experiment` uses attention-window A/B schema 1 and reuses the #34 protocol:

- same model dimensions and K/V-head layout;
- same initialization seed;
- same token stream;
- same normalization and positional policy;
- same optimizer, batch, accumulation and optimizer-step budget;
- only the attention window differs.

The harness reports loss, perplexity, training throughput, parameter bytes and checkpoint bytes. Quality evidence is descriptive and does not auto-promote local attention.

## Benchmark

Run:

```bash
cargo run --release --bin auralis_local_attention_bench
```

The benchmark publishes:

- dense vs window-8 quality A/B;
- dense vs window-16 quality A/B;
- contexts 8, 16, 32 and 64;
- windows dense/8/16;
- median forward latency;
- median backward latency;
- actual probability-cache slots and bytes;
- finite loss and gradients.

Hosted-runner timing is evidence, not a hardware guarantee. The hard gates are correctness, causality/window boundaries, finite-difference gradients, valid configuration handling, dense-reference equivalence when the window covers the full context, and actual probability-memory reduction. A latency or quality advantage is not required for acceptance.

## Non-goals

This slice does not add block-sparse GPU kernels, learned sparsity, global tokens, paged attention, KV eviction, or a change of default attention policy.
