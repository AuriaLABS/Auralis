# GQA / MQA experiment

GQA and MQA are **experimental** Brain attention variants for #140. **MHA remains the default** and reference model path.

## Policy and shapes

The model keeps `n_head` query heads and adds explicit `n_kv_head`.

- MHA: `n_kv_head == n_head`.
- GQA: `1 < n_kv_head < n_head`, with `n_head % n_kv_head == 0`.
- MQA: `n_kv_head == 1`.

Invalid zero, oversized or non-divisible KV-head counts fail closed. Architecture schema 5 persists `n_kv_head` plus the orthogonal `attention_window`; schema 4 migrates to dense attention and historical schemas 1-3 migrate to MHA + dense.

## Physical layout

This is not a logical-only grouping layer. GQA/MQA use **compact Wk/Wv** matrices with output width
`head_width * n_kv_head`. Q and the output projection remain full-width. Backward accumulates gradient contributions from every query-head in a group onto the shared compact K/V activations and projection weights.

Initialization consumes the same RNG budget as MHA before compacting K/V columns so non-K/V parameters remain directly comparable across variants.

## Positional policies and decode

Learned absolute and ALiBi do not transform compact K/V tensors. RoPE rotates all query heads and each compact KV head at the same absolute positions.

The per-session KV cache stores compact K/V width plus explicit query-head/KV-head identity, so GQA/MQA reduce active **KV-cache bytes** relative to MHA while incompatible layouts with the same byte width fail closed. Cached decode remains transactional across layers.

## Training and quality evidence

`run_attention_head_experiment` reuses the versioned Brain A/B protocol while recording `n_kv_head` per variant. The same seed, token stream, optimizer policy, batch and number of optimizer steps are used. The harness reports quality evidence (loss/perplexity), throughput, parameter bytes and checkpoint bytes; it does not declare a winner.

`cargo run --release --bin auralis_gqa_mqa_bench` adds decode and memory curves for MHA, GQA and MQA at contexts 8, 16 and 32. Hosted-runner timing is descriptive and does not auto-promote any attention variant. Each context reports full-prefix prefill latency, cache-building prefill latency, uncached next-token latency and cached next-token latency separately. Correctness, targeted finite-difference gradients for compact Wk/Wv, compact parameter counts and monotonic/reduced cache memory are hard gates; quality and latency are published evidence and do not auto-promote a variant.

## Default and promotion

This PR does not change the default architecture. MHA remains the default. GQA/MQA must remain selectable and reversible, and any future default change requires separate evidence and an explicit decision.
