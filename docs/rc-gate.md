# Gate local de release candidate

Herramienta: `auralis release-check` y `auralis release-manifest`.
Fuente de artefactos: `DEFAULT_RELEASE_ARTIFACTS` en `src/release.rs`.
Schema del manifiesto de release: `RELEASE_MANIFEST_VERSION = 1`.

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

## Gates pendientes (no locales)

- `rc_tag` — pending si no hay candidato; nunca crea el tag
- `live_ci` — check-runs de GitHub
- `artifacts_checksums` — usar `release-manifest`
- `declared_blockers` — #68 GPU experimental no es gate de release
- `human_approval` — revisión humana (#84)

Cómo lanzarlo: [verify.md](verify.md).
