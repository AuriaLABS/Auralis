# CLI

Fuente de verdad del texto de ayuda: `fn usage()` en `src/main.rs`.
`auralis --help`, `-h` y `help` imprimen ese bloque.

## Verbos

```
auralis train [steps] [checkpoint] [seed] [batch] [accum] [--config FILE]
auralis train-fresh [steps] [checkpoint] [seed] [batch] [accum] [--config FILE]
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

## Flags reconocidos

`--config`, `--json`, `--csv`, `--out`, `--verify`, `--help`/`-h`/`help`.

`--json` y `--csv` son exclusivos en `bench`.

Config de ejemplo (schema 1): [`examples/tiny.cfg`](../examples/tiny.cfg).

Cómo ejecutarlos: [verify.md](verify.md).
