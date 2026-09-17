# Gate local de release candidate

Herramienta: `auralis release-check` y `auralis release-manifest`.
Fuente de artefactos: `DEFAULT_RELEASE_ARTIFACTS` en `src/release.rs`.
Schema del manifiesto de release: `RELEASE_MANIFEST_VERSION = 1`.

Esta herramienta **nunca crea tags** ni registra `human_approval`.
`automated_pass` solo mira archivos locales.

## Artefactos que deben existir

- `Cargo.toml`
- `README.md`
- `VISION.md`
- `ROADMAP.md`
- `.github/workflows/genesis.yml`
- `.github/workflows/pruebas.yml`
- `docs/ci-pruebas.md`

También presente: [`docs/model-card.md`](model-card.md) (contenido Genesis, no RC).

## Gates pendientes (no locales)

- `rc_tag` — identidad git/RC
- `live_ci` — check-runs de GitHub
- `artifacts_checksums` — usar `release-manifest`
- `human_approval` — revisión humana (#84)

Cómo lanzarlo: [verify.md](verify.md).
