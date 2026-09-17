# Log de entorno limpio

El runner es el job `cli smoke` de `.github/workflows/pruebas.yml`.
Si este archivo y el workflow divergen, `docs_check` falla.
Esto no es el soak de Foundation ni un tag RC.

## Comandos que CI ejecuta hoy

```bash
cargo run --release -- check
cargo run --release -- bpe
cargo run --release -- train-fresh 2 /tmp/auralis-pruebas.bin 659918 2 1
cargo run --release --bin auralis_eval_profile -- 4
```

## Marcadores que deben aparecer

- `ok=true` (salida de `check`)
- `run_summary |` (salida de `train-fresh`)
- archivo `/tmp/auralis-pruebas.bin`
- archivo `/tmp/auralis-pruebas.bin.manifest`
- `eval_profile | mode=loss`
- `eval_profile | mode=generate_one`

El job `cli` vive en ubuntu-latest, checkout fresco, toolchain stable.
No usa GPU ni red de entrenamiento.

Cómo encaja: [verify.md](verify.md).
