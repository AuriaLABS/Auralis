# E1.7 — forward de eval sin ForwardCache

`Gpt::logits` / `forward_eval` calculan los mismos logits que `forward_internal`
sin reservar `LayerCache` ni `ForwardCache`.

- `loss` y `generate` usan `forward_eval`.
- `backward_into_reuse` sigue usando `forward_internal` + caches.
- Test: `eval_forward_matches_training_forward_logits` (bit-exact).

Cierra el camino de #27. No cambia kernels, sampling ni el orden aritmético del train path.
