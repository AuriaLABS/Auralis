# CLI

Fuente de verdad del texto de ayuda: `fn usage()` en `src/main.rs`.
`auralis --help`, `-h` y `help` imprimen ese bloque.

## Verbos

```
auralis train [steps] [checkpoint] [seed] [batch] [accum] [--config FILE] [--model-config FILE] [--scheduler-config FILE] [--diagnostics]
auralis train-fresh [steps] [checkpoint] [seed] [batch] [accum] [--config FILE] [--model-config FILE] [--scheduler-config FILE] [--diagnostics]
auralis config [FILE]
auralis inspect [checkpoint] [--json]
auralis release-check [ROOT] [--json]
auralis release-manifest [ROOT] [--out FILE] [--verify FILE]
auralis sec-audit [SRC_ROOT]
auralis bench list [--json|--csv]
auralis bench describe ID [--json|--csv]
auralis eval [checkpoint]
auralis chat [checkpoint]
auralis check
auralis bpe
```

Binario auxiliar (no está en `fn usage()`): `cargo run --release --bin auralis_numeric`.

## Flags reconocidos

`--config`, `--model-config`, `--diagnostics`, `--json`, `--csv`, `--out`, `--verify`, `--help`/`-h`/`help`.

`--json` y `--csv` son exclusivos en `bench`.

Config de ejemplo (schema 2): [`examples/tiny.cfg`](../examples/tiny.cfg). Clipping OFF: [`examples/clipping-off.cfg`](../examples/clipping-off.cfg).

Cómo ejecutarlos: [verify.md](verify.md).

Model architecture config (schema 1): [architecture-config.md](architecture-config.md).

Learning-rate scheduler (schema 1): [scheduler.md](scheduler.md).

Gradient clipping policy (RunConfig schema 2): [gradient-clipping.md](gradient-clipping.md).
