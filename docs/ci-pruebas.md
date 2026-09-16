# Flujos de pruebas

El gate histórico sigue en `.github/workflows/genesis.yml` (job `core`).
Las familias nuevas viven en `.github/workflows/pruebas.yml`.

Catálogo machine-readable: `ci_matrix::list_json()` / `ci_matrix::JOBS`.

## Jobs

| Job | Qué corre | Cuándo | Bloquea merge |
| --- | --- | --- | --- |
| `core` | genesis.yml gate | PR + main | sí |
| `unit` | `cargo test --lib` | PR + main + manual | sí |
| `integration` | genesis, properties, equivalence | PR + main + manual | sí |
| `fuzz` | `fuzz_parsers` | PR + main + manual | sí |
| `cli` | `check`, `bpe`, `train-fresh` corto | PR + main + manual | sí |
| `profile` | `auralis_eval_profile` | PR + main + manual | sí |
| `benches` | matmul/engine bench corto | **solo main o Run workflow** | no |

Pendiente #136: fixtures de compatibilidad, artifact checksum job, flaky tracker, hardware opcional.

## Manual

Actions → **Pruebas** → Run workflow → elegir `suite`.

## Local

```bash
cargo test --lib
cargo test --test genesis --test properties --test engine_equivalence --test engine_state_equivalence
cargo test --test fuzz_parsers
cargo run --release -- check
cargo run --release --bin auralis_eval_profile -- 4
```

No sustituye a `core`. Si un job de Pruebas falla y `core` pasa, el Integrator no mergea.
