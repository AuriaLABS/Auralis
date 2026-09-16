# E1.8 — Workspaces de entrenamiento

Issue: #28. No modifica `model.rs` (#27).

## Consumidores conocidos (inventario)

| Buffer | Dónde | Vida |
| --- | --- | --- |
| `TrainWorkspace.micro_grads` | `training.rs` | step, reutilizado |
| `TrainWorkspace.sample_grads` | `training.rs` | sample, reutilizado |
| `TrainWorkspace.params` | `training.rs` | espejo Adam |
| `BackwardWorkspace` | `model.rs` | opaco; #27 |
| `grads` del caller | CLI / este bin | step |
| batches materializados | `train_step` reference | por microbatch (alloc) |

## ForwardCache (solo training-forward)

Ruta `forward_internal`. `forward_eval` (#27) no debe reservar esta tabla.

Por capa (`LayerCache`), con `t` tokens y `d = n_embd`:

| Campo | Elementos f32 |
| --- | --- |
| `ln1.xhat` + `ln2.xhat` | `2 * t * d` |
| `ln1.inv_std` + `ln2.inv_std` | `2 * t` |
| `h1`, `q`, `k`, `v`, `att`, `h2` | `6 * t * d` |
| `probs` | `n_head * t * t` |
| `ff_pre`, `ff_act` | `2 * t * n_ff` |

Más `ln_f` (`t*d + t`) y `h_final` (`t*d`).

### Config::tiny, secuencia llena (`t=32`, `d=32`, `n_head=4`, `n_layer=2`, `n_ff=96`)

- por capa: `18496` f32 ≈ `72 KiB`
- 2 capas + `ln_f` + `h_final`: ≈ `152 KiB` solo caches de backward
- `probs` es el término cuadrático: `n_head * t * t` (aquí 4096 f32/capa)

Hotspot de reducción **sin cambiar la matemática**: no construir `LayerCache` en eval/chat (#27). Reuse de `TrainWorkspace` ya cubre grads; no justifica recompute de atención todavía.

## Cómo medir

```bash
cargo run --release --bin auralis_train_profile -- 8
```

Espera dos líneas:

```
train_profile | mode=reference ...
train_profile | mode=reuse ...
```

`reuse` debe gastar menos `alloc_bytes_per_iter` que `reference` si el workspace cumple su contrato. Pegar números en #28 antes de tocar activaciones internas.
