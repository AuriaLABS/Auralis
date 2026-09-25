# Versiones vigentes

Si este archivo miente, `docs_check` falla.
Subir un número aquí **no** es un release; el tag `v1.0.0` sigue el gate #84.

| Contrato | Valor ahora |
| --- | --- |
| crate (`Cargo.toml`) | `0.1.0` |
| Rust edition | `2021` |
| run config schema | `RUN_CONFIG_SCHEMA_VERSION = 2` |
| architecture config schema | `ARCHITECTURE_CONFIG_SCHEMA_VERSION = 5` |
| scheduler config schema | `SCHEDULER_CONFIG_SCHEMA_VERSION = 1` |
| optimizer config schema | `OPTIMIZER_CONFIG_SCHEMA_VERSION = 1` |
| optimizer state schema | `OPTIMIZER_STATE_SCHEMA_VERSION = 1` |
| AdamW state schema | `ADAMW_STATE_SCHEMA_VERSION = 1` |
| Lion state schema | `LION_STATE_SCHEMA_VERSION = 1` |
| brain A/B schema | `BRAIN_AB_SCHEMA_VERSION = 4` |
| attention-head A/B schema | `ATTENTION_HEAD_AB_SCHEMA_VERSION = 1` |
| attention-window A/B schema | `ATTENTION_WINDOW_AB_SCHEMA_VERSION = 1` |
| external memory schema | `EXTERNAL_MEMORY_SCHEMA_VERSION = 1` |
| memory integration schema | `MEMORY_INTEGRATION_SCHEMA_VERSION = 1` |
| recurrent config schema | `RECURRENT_CONFIG_SCHEMA_VERSION = 1` |
| KV cache session schema | `KV_CACHE_SCHEMA_VERSION = 3` |
| MoE router schema | `MOE_ROUTER_SCHEMA_VERSION = 1` |
| evaluation registry schema | `EVALUATION_REGISTRY_SCHEMA_VERSION = 1` |
| reasoning suite schema | `REASONING_SUITE_SCHEMA_VERSION = 1` |
| memory suite schema | `MEMORY_SUITE_SCHEMA_VERSION = 1` |
| code suite schema | `CODE_SUITE_SCHEMA_VERSION = 1` |
| capability suite schema | `CAPABILITY_SUITE_SCHEMA_VERSION = 2` |
| agent eval schema | `AGENT_EVAL_SCHEMA_VERSION = 1` |
| tool protocol schema | `TOOL_PROTOCOL_SCHEMA_VERSION = 1` |
| agent permission policy schema | `PERMISSION_POLICY_SCHEMA_VERSION = 1` |
| tool registry schema | `TOOL_REGISTRY_SCHEMA_VERSION = 1` |
| planner schema | `PLAN_SCHEMA_VERSION = 1` |
| plan executor schema | `EXECUTOR_SCHEMA_VERSION = 1` |
| session schema | `SESSION_SCHEMA_VERSION = 1` |
| local API schema | `LOCAL_API_SCHEMA_VERSION = 1` |
| event stream schema | `EVENT_STREAM_SCHEMA_VERSION = 1` |
| session memory schema | `SESSION_MEMORY_SCHEMA_VERSION = 1` |
| tool provider schema | `TOOL_PROVIDER_SCHEMA_VERSION = 1` |
| adversarial suite schema | `ADVERSARIAL_SUITE_SCHEMA_VERSION = 1` |
| lab registry schema | `LAB_REGISTRY_SCHEMA_VERSION = 1` |
| lab runner schema | `LAB_RUNNER_SCHEMA_VERSION = 1` |
| lab review schema | `LAB_REVIEW_SCHEMA_VERSION = 1` |
| lab proposal schema | `LAB_PROPOSAL_SCHEMA_VERSION = 1` |
| lab orchestrator schema | `LAB_ORCHESTRATOR_SCHEMA_VERSION = 1` |
| campaign schema | `CAMPAIGN_SCHEMA_VERSION = 1` |
| campaign scheduler schema | `CAMPAIGN_SCHEDULER_SCHEMA_VERSION = 1` |
| lab branch schema | `LAB_BRANCH_SCHEMA_VERSION = 1` |
| lab catalog schema | `LAB_CATALOG_SCHEMA_VERSION = 1` |
| modality schema | `MODALITY_SCHEMA_VERSION = 1` |
| multimodal dataset schema | `MM_DATASET_SCHEMA_VERSION = 1` |
| vision schema | `VISION_SCHEMA_VERSION = 1` |
| experiment manifest | `MANIFEST_VERSION = 4` |
| release manifest | `RELEASE_MANIFEST_VERSION = 1` |

`auralis_run_config=1` migra explícitamente a schema 2 con `grad_clip_enabled=true`; schemas de run config distintos de 1/2 se rechazan. Un manifiesto v3 se rechaza. `auralis_architecture=1` migra a LayerNorm + learned-absolute + MHA; schema 2 conserva normalización y migra a learned-absolute + MHA; schema 3 conserva normalización/posición y migra a MHA (`n_kv_head=n_head`); schema 4 registra `n_kv_head` y migra a atención dense; schema 5 registra también `attention_window`. Otros schemas se rechazan (fail-closed). El contrato MoE router schema 1 es experimental e independiente del formato de checkpoint/modelo; dense permanece default. El evaluation registry schema 1 versiona suites/tareas/métricas/baselines y rechaza drift de definición sin bump explícito. La reasoning suite schema 1 fija generación determinista, splits de dificultad/longitud y scoring estructurado objetivo. La memory suite schema 1 fija el perfil de evaluación de memoria externa, latest-record-wins para historias de una clave y capacity policy reject-new/no-eviction. La code suite schema 1 fija fixtures Rust ejecutables, patch de una expresión, rustc --test, timeouts y taxonomía de compile/test. La capability suite schema 2 compone reasoning/code/memory/agent con reporte por capacidad y regresiones aisladas; el promedio descriptivo no es gate. El agent eval schema 1 fija trazas versionadas, replay mock-only, redacción de secretos, taxonomía objetiva tool/reasoning/permission y el baseline común `agent-common-baseline-1` que pinna los schemas de protocol/permissions/planner/executor/registry/session/API. El tool registry schema 1 fija discovery determinista por nombre/versión y herramientas reference locales; registrar una definición no concede autoridad y #45 sigue siendo el único gate de permisos para ejecución. El planner schema 1 fija planes inspectables y un runner mínimo. El plan executor schema 1 añade la máquina de estados, abort, rollback lógico y denegación de replay mutativo. El session schema 1 versiona el REPL local: reset/save/load/export, rechazo de schema incompatible y separación del checkpoint del modelo. El local API schema 1 expone las mismas sesiones por handle in-process y loopback 127.0.0.1. El event stream schema 1 fija envelope versionado, sequence IDs, buffer acotado con backpressure y cancelación que bloquea eventos mutativos. El session memory schema 1 fija un store opcional de sesión, evict-oldest y memory-off sin tocar pesos. El tool provider schema 1 fija list/describe/invoke/negotiate detrás de #45. El adversarial suite schema 1 fija fixtures fail-closed para tools/permisos/resultados malformados. El lab registry schema 1 fija hipótesis/experimentos/resultados/decisiones y exige baseline + criterio antes de Running. El lab runner schema 1 ejecuta solo fixtures allowlisted y trata timeout/artefacto corrupto como fallo cerrado. El lab review schema 1 aplica el criterio original y separa revisor de implementer. El lab proposal schema 1 exige hipótesis, baseline, presupuesto y falsación antes de crear un experimento. El lab orchestrator schema 1 coordina agentes con scopes exclusivos y handoffs explícitos. El campaign schema 1 ejecuta candidatos×suites×seeds con presupuesto global y resume sin repetir runs. El campaign scheduler schema 1 persiste la cola local y detiene el scheduling si el presupuesto se agota. El lab branch schema 1 abre PRs experimentales en draft desde un baseline registrado y nunca escribe `main`. El lab catalog schema 1 indexa evidencia previa y mantiene positivos, negativos e inconclusos separados. El modality schema 1 fija la frontera común texto/off sin tocar el baseline textual. El multimodal dataset schema 1 versiona manifests, fingerprints y splits deterministas. El vision schema 1 proyecta parches de imagen al contrato común sin un encoder externo.
Gate local: [rc-gate.md](rc-gate.md).
