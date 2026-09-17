# Model card — Auralis 0.1.0 Genesis

Esto no es un modelo de frontera. Es la tarjeta de un experimento reproducible.
Si una sección falta, `docs_check` falla.

## Intended use

Entrenar e inferir un transformer decoder-only mínimo escrito en Rust,
con checkpoints, manifiesto Foundation y CLI local. Uso de investigación.

## Out of scope

- asistente de producción
- multimodal
- GPU / distribuido
- afirmar SOTA o capacidades no medidas

## Architecture

Decoder-only GPT-like (`src/model.rs`). Tokenizador BPE por defecto (`bpe_merges=64`).
Optimizador Adam. Entrenamiento con `TrainWorkspace` y `train_step_reuse`.

## Training data

Corpus local `data/corpus.txt` más ancla repetida en el binario.
No hay dataset público versionado todavía. Fingerprint vive en el manifiesto v4.

## Evaluation

Comandos medibles: [verify.md](verify.md).
No hay tabla de ppl/human-eval publicada para un tag RC.
Un `train-fresh` corto **no** es el soak de Foundation.

## Metrics

Cualquier número que no salga de un manifiesto + `docs/verify.md` es anecdótico.
Este archivo no inventa baselines.

## Limitations

- contexto tiny; generación todavía débil fuera del corpus
- sin GPU; CPU only
- `unsafe` de librería debe ser 0 ([sec-unsafe.md](sec-unsafe.md))
- `forward_eval` sin cache de backward sigue en #27; no afirmar ahorro de memoria hasta merge

## Ethics / risks

Modelo pequeño de texto. Puede alucinar y no debe usarse como fuente de verdad.
No hay filtro de contenido en el sampling.

## Versioning

[versions.md](versions.md). Crate `0.1.0`. Manifest experiment `4`. Run-config schema `1`.
Tag `v1.0.0` solo con gate [rc-gate.md](rc-gate.md) y aprobación humana (#84).

## Supported vs experimental

Soportado = verbo en `fn usage()` y ejecutable hoy con [cli.md](cli.md).
Experimental = existe como dirección o prototipo; no es contrato de release.

### Supported

- `auralis train`
- `auralis train-fresh`
- `auralis config`
- `auralis inspect`
- `auralis release-check`
- `auralis release-manifest`
- `auralis sec-audit`
- `auralis bench list`
- `auralis bench describe`
- `auralis eval`
- `auralis chat`
- `auralis check`
- `auralis bpe`
- CPU tiny decoder-only + BPE + Adam + checkpoint/manifiesto v4

### Experimental

- GPU / SIMD / matmul candidato como hot path (#30)
- `forward_eval` sin `ForwardCache` (#27)
- `cargo run --bin auralis_numeric` (fixtures NaN/Inf; no cableado al engine)
- agente con herramientas / memoria larga
- multimodal, Scientist, self-improvement
- tag `v1.0.0` y model card con métricas RC
- asistente de producción
