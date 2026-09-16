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
