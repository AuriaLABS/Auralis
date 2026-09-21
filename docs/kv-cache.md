# Autoregressive KV cache

Issue #40 adds an explicit, per-session key/value cache for autoregressive inference. The cache is an inference acceleration structure only: it is not model state, is not part of checkpoints, and does not change training/backward.

## Contract

`KvCache` schema 1 records only session metadata and derived K/V activations:

- physical layer count;
- model width;
- maximum context capacity;
- active token count;
- positional-policy identity;
- one K tensor and one V tensor per physical layer.

The model creates a matching empty cache with `Gpt::new_kv_cache()`.

## Lifecycle

The lifecycle is explicit:

- **prefill**: `prefill_kv_cache(prefix, cache)` requires an empty cache and returns the same row-major logits as uncached `logits(prefix)`;
- **decode**: `decode_kv_cached(token, cache)` appends one token and returns only its vocabulary-logit row;
- **clone**: cloning a cache creates an independent session branch;
- **reset**: `reset()` removes all active K/V state while retaining reusable vector capacity;
- **overflow**: appending at `block` capacity fails explicitly. The reference uncached path remains available for sliding-window/rebuild policies.

A token append is transactional across layers. New K/V rows are staged while attention reads the immutable history; the cache length advances only after all layers and the output projection complete successfully.

## Positional policies

Cached decode is covered for every currently selectable positional policy:

- learned absolute: the new token receives the embedding row at its absolute session position;
- RoPE: the single Q/K row is rotated at the absolute session position;
- ALiBi: decode attention adds the same head slope times key/query distance used by full-prefix attention.

Cached and uncached logits are required to match bit-for-bit in the tiny reference tests.

## Attention semantics

The decode primitive evaluates one query row against:

1. cached historical K/V rows; and
2. the current token's staged K/V row.

The scalar accumulation and softmax order match the corresponding last causal row in the full eval attention path. No full probability matrix is materialized.

## Memory accounting

`logical_bytes()` reports active K/V payload:

```text
tokens * layers * width * 2 * sizeof(f32)
```

`allocated_bytes()` reports current vector capacity and may exceed logical bytes because Rust vectors grow geometrically and retain capacity after reset.

## Benchmark

Run:

```bash
cargo run --release --bin auralis_kv_cache_bench
```

The benchmark compares a growing-prefix autoregressive session in two modes:

- **uncached**: recompute `logits(prefix)` for every prefix length;
- **cached**: append exactly one token per step.

It uses one fixed model/seed and contexts 8, 16, 32 and 64. For every context it reports:

- total and per-token latency for both paths;
- cached/uncached speedup;
- logical and allocated cache bytes;
- model parameter count;
- exact cached-vs-uncached correctness.

Hosted-runner timing is evidence, not a stable hardware guarantee. CI requires correctness, fixed parameter count, monotonic logical cache bytes and a measurable cached advantage at the longest context.

## Non-goals

This implementation does not add paged attention, multi-session batching, speculative decoding, GPU storage or checkpoint serialization of session cache state.

The ordinary uncached inference path remains the reference and fallback.
