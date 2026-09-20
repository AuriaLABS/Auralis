# Versiones vigentes

Si este archivo miente, `docs_check` falla.
Subir un número aquí **no** es un release; el tag `v1.0.0` sigue el gate #84.

| Contrato | Valor ahora |
| --- | --- |
| crate (`Cargo.toml`) | `0.1.0` |
| Rust edition | `2021` |
| run config schema | `RUN_CONFIG_SCHEMA_VERSION = 2` |
| architecture config schema | `ARCHITECTURE_CONFIG_SCHEMA_VERSION = 2` |
| scheduler config schema | `SCHEDULER_CONFIG_SCHEMA_VERSION = 1` |
| optimizer config schema | `OPTIMIZER_CONFIG_SCHEMA_VERSION = 1` |
| optimizer state schema | `OPTIMIZER_STATE_SCHEMA_VERSION = 1` |
| AdamW state schema | `ADAMW_STATE_SCHEMA_VERSION = 1` |
| Lion state schema | `LION_STATE_SCHEMA_VERSION = 1` |
| brain A/B schema | `BRAIN_AB_SCHEMA_VERSION = 2` |
| experiment manifest | `MANIFEST_VERSION = 4` |
| release manifest | `RELEASE_MANIFEST_VERSION = 1` |

`auralis_run_config=1` migra explícitamente a schema 2 con `grad_clip_enabled=true`; schemas de run config distintos de 1/2 se rechazan. Un manifiesto v3 se rechaza. `auralis_architecture=1` migra explícitamente a LayerNorm; schemas de arquitectura distintos de 1/2 se rechazan (fail-closed).
Gate local: [rc-gate.md](rc-gate.md).
