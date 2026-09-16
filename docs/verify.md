# Cómo probar Auralis

Nada de esta página vale si `cargo test --lib docs_check` falla.
Los comandos de abajo no usan red.

## Gates de merge (deben estar verdes)

| Qué | Comando |
| --- | --- |
| unidad | `cargo test --lib` |
| integración | `cargo test --test genesis --test properties --test engine_equivalence --test engine_state_equivalence` |
| parsers | `cargo test --test fuzz_parsers` |
| gradcheck CLI | `cargo run --release -- check` |
| BPE | `cargo run --release -- bpe` |
| eval profile | `cargo run --release --bin auralis_eval_profile -- 4` |

`benches` no bloquea PRs. Ver `docs/ci-pruebas.md` y `ci_matrix::JOBS`.

Verbos y flags: [cli.md](cli.md).

## Ciclo Genesis (README)

```bash
cargo test
cargo run --release -- check
cargo run --release -- train-fresh 2 /tmp/auralis-verify.bin 659918 2 1
cargo run --release -- inspect /tmp/auralis-verify.bin
cargo run --release -- bench list --json
cargo run --release -- sec-audit src
```

`train-fresh` corto deja checkpoint + manifiesto. No sustituye al soak de Foundation.

## Contratos que ya tienen test de docs

| Doc | Qué se comprueba |
| --- | --- |
| `docs/ci-pruebas.md` | cada id de `ci_matrix::JOBS` |
| `docs/bench-registry.md` | verbos `bench list/describe` y flags |
| `docs/sec-unsafe.md` | límites y `EXPECTED_UNSAFE_BLOCKS == 0` |
| `docs/cli.md` | verbos presentes en `fn usage()` |
| `README.md` | comandos del ciclo documentado |
| esta página | existe y nombra los mismos comandos |

## Fuera de este slice

Links externos, snippets Rust compilables, hardware GPU.
