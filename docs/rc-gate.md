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

También presente: [`docs/model-card.md`](model-card.md) (contenido Genesis, no RC).

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
- decode inválido o checksum mismatch → fail-closed;
- el CLI no inventa el manifiesto.

## Gate `compatibility_fixtures`

- sin fixture → pending; migrate never rewrites;
- checkpoint/manifest/session v1 que `accept` → pass;
- version 0 o desconocida → fail-closed.

## Gates pendientes (no locales)

- `rc_tag` — pending si no hay candidato; nunca crea el tag
- `live_ci` — pending si no hay snapshot; no hay red
- `artifacts_checksums` — pending si no hay manifiesto almacenado
- `compatibility_fixtures` — pending si no hay fixture
- `declared_blockers` — #68 GPU experimental no es gate de release
- `human_approval` — revisión humana (#84)

Cómo lanzarlo: [verify.md](verify.md).
