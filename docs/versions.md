# Versiones vigentes

Si este archivo miente, `docs_check` falla.
Subir un número aquí **no** es un release; el tag `v1.0.0` sigue el gate #84.

| Contrato | Valor ahora |
| --- | --- |
| crate (`Cargo.toml`) | `0.1.0` |
| Rust edition | `2021` |
| run config schema | `RUN_CONFIG_SCHEMA_VERSION = 1` |
| architecture config schema | `ARCHITECTURE_CONFIG_SCHEMA_VERSION = 1` |
| experiment manifest | `MANIFEST_VERSION = 4` |
| release manifest | `RELEASE_MANIFEST_VERSION = 1` |

Un manifiesto v3, un `auralis_run_config` distinto de 1 o un `auralis_architecture` distinto de 1 se rechaza (fail-closed).
Gate local: [rc-gate.md](rc-gate.md).
