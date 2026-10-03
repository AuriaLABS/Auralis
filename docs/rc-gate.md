# Gate local de release candidate

Herramienta: `auralis release-check` y `auralis release-manifest`.
Fuente de artefactos: `DEFAULT_RELEASE_ARTIFACTS` en `src/release.rs`.
Schema del manifiesto de release: `RELEASE_MANIFEST_VERSION = 1`.
Schema de compatibilidad: `COMPAT_SCHEMA_VERSION = 1`.

Esta herramienta **nunca crea tags** ni registra `human_approval`.
`automated_pass` solo mira archivos locales.
Does **not** cut `v1.0.0`.

## Artefactos que deben existir

- `Cargo.toml`
- `README.md`
- `VISION.md`
- `ROADMAP.md`
- `.github/workflows/genesis.yml`
- `.github/workflows/pruebas.yml`
- `docs/ci-pruebas.md`
- `LICENSE`
- `docs/api-policy.md`
- `docs/packaging.md`
- `docs/baselines.md`
- `docs/post-tag.md`

También presente: [`docs/model-card.md`](model-card.md) (contenido Genesis, no RC). El gate exige el fichero y las secciones; no inventa métricas RC.

## Gate `rc_tag`

- sin candidato → pending;
- `v1.0.0` con tag SHA = source SHA → pass;
- mismatch o nombre distinto → fail-closed;
- `create_tag` está prohibido.

## Gate `live_ci`

- sin snapshot → pending; live GitHub check-run status is not queried locally;
- snapshot con `required_pr_jobs` en `success` → pass;
- job requerido ausente o conclusion distinta de `success` → fail-closed;
- `benches` no bloquea.

## Gate `artifacts_checksums`

- sin manifiesto almacenado → pending; use auralis release-manifest to emit/verify checksums;
- texto `RELEASE_MANIFEST_VERSION = 1` que decodifica y `verify` → pass;
- `target` y `features` pueden quedar `unrecorded`; `bench_run_ids` y `eval_run_ids` vacíos no inventan runs;
- `--features` registra solo nombres `a-z0-9_-` ordenados; no inventa features;
- `checkpoint_id` y `model_id` vacíos por defecto; no inventa checkpoint_id;
- una ruta relativa existente se verifica; `..` o ausente → fail-closed;
- `cargo_lock` es el checksum de `Cargo.lock` o `unrecorded`; no inventa versiones de dependencias;
- `limitations=docs/model-card.md` es una referencia, no una métrica RC;
- orden determinista: campos fijos y artefactos por path;
- `code_revision` es `unknown` o un SHA de 40 hex; no inventa el SHA;
- `verify` rechaza un placeholder de `code_revision` antes de cualquier tag;
- dos SHA de 40 hex distintos → fail-closed; `unknown` no inventa un match;
- el report copia `code_revision`, `features` y `cargo_lock`; `bench_run_ids` vacío no inventa runs; `eval_run_ids` vacío no inventa runs;
- decode inválido o checksum mismatch → fail-closed;
- el CLI no inventa el manifiesto;
- `--manifest FILE` aplica ese texto; ausente sigue pending;
- `check_release_full` aplica checksums y fixtures al report; el CLI sigue sin inventarlos.

## Gate `compatibility_fixtures`

- sin fixture → pending; migrate never rewrites;
- checkpoint/manifest/session v1 que `accept` → pass;
- version 0 o desconocida → fail-closed;
- `check_release_full` aplica el fixture; ausente sigue pending.

## Gate `suites_evals`

- sin snapshot → pending; release-check does not invent scores;
- reasoning/code/memory/agent complete → pass;
- suite requerida ausente o incompleta → fail-closed;
- el CLI no inventa resultados.

## Gate `benchmarks`

- sin snapshot → pending; release-check does not invent numbers;
- `BASELINES_SCHEMA_VERSION = 1`; matmul/forward comparable y sin regresión bloqueante → pass;
- bench requerido ausente, Inconclusive o regresión bloqueante → fail-closed;
- GPU is not a release gate.

## Gate `model_card`

- `docs/model-card.md` ausente o sin sección requerida → fail-closed;
- secciones presentes y `no inventa baselines` → pass;
- `CARD_SCHEMA_VERSION = 1`; does not invent RC metrics;
- contenido Genesis no es una tabla RC.

## Gate `declared_blockers`

- sin snapshot → pending; release-check does not query GitHub;
- issue abierto marcado release gate → fail-closed;
- #68 is not a v1 release gate.

## Gates pendientes (no locales)

- `rc_tag` — pending si no hay candidato; nunca crea el tag
- `live_ci` — pending si no hay snapshot; no hay red
- `artifacts_checksums` — pending si no hay manifiesto almacenado
- `compatibility_fixtures` — pending si no hay fixture
- `suites_evals` — pending si no hay snapshot; no inventa scores
- `benchmarks` — pending si no hay snapshot; no inventa números
- `declared_blockers` — pending si no hay snapshot; no consulta GitHub; #68 is not a v1 release gate
- `human_approval` — revisión humana (#84)

## Report

- `RELEASE_REPORT_SCHEMA_VERSION = 1`;
- JSON incluye `schema`, `automated_pass`, `human_approval` y `creates_tags: false`;
- la herramienta no aprueba ni crea el tag.

Cómo lanzarlo: [verify.md](verify.md).
