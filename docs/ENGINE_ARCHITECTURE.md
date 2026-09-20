# Auralis Engine 0.3 — arquitectura actual

Este documento describe **el Engine que existe y está integrado hoy**. No es una lista de aspiraciones del roadmap.

La regla de lectura es simple:

- **reference** significa una ruta deliberadamente sencilla que sirve como oráculo semántico;
- **optimized** significa una ruta promovida solo después de equivalencia + benchmark;
- **experimental** significa que existe código o evidencia, pero no forma parte de la ruta productiva;
- una capacidad futura se nombra como deuda o transición, no como si ya estuviera implementada.

El Engine sigue siendo FP32, CPU-first y sin dependencia de un framework externo de deep learning.

---

## 1. Módulos y responsabilidades

### `model`
Contiene el decoder-only GPT, sus parámetros, forward de entrenamiento, forward de evaluación/inferencia, backward y workspaces específicos del modelo.

Responsabilidades principales:

- configuración estructural del modelo;
- parámetros y escritura/lectura del vector plano;
- forward con caches cuando el backward los necesita;
- forward de evaluación sin materializar caches de backward;
- logits/generación;
- backward explícito;
- diagnóstico forward opt-in por layer/tensor.

### `training`
Orquesta un optimizer step reproducible.

Mantiene dos familias deliberadas:

- `train_step`: referencia Foundation, simple y allocation-heavy;
- `train_step_reuse`: ruta Engine productiva con buffers persistentes.

Además expone rutas opt-in separadas:

- `train_step_reuse_diagnostics`;
- `train_step_reuse_timing` / ruta timed.

La ruta normal no activa esos scans/timers implícitamente.

### `optim`
Implementa Adam y su estado `m/v/t`.

### `arena` / `TrainWorkspace` / `BackwardWorkspace`
Concentran scratch y buffers reutilizables para evitar heap churn dentro del step.

### `kernels`
Contiene kernels CPU de referencia y kernels promovidos. El matmul de referencia permanece accesible como oráculo.

### `backend`
Frontera Phase A integrada para matmul:

- `ScalarCpuBackend`;
- `OptimizedCpuBackend`;
- `MockBackend`;
- `DeviceId`, `BackendId`;
- `MatrixRef` / `MatrixMut` prestados;
- errores explícitos de shape/storage/device.

**Importante:** esta frontera todavía no controla globalmente `model.rs`. La integración model-wide es Phase B.

### `numeric` / `numeric_state`
Escaneo de NaN/Inf, summaries numéricos y contexto de primera corrupción.

### `metrics`
Métricas canónicas y `EngineStepTiming` schema v1.

### `bench` / `bench_runner` / `bench_result`
Registro, ejecución in-process y serialización reproducible de benchmarks.

### `checkpoint` / `manifest` / `run_config`
Persistencia reproducible:

- AURLIS03 puede incluir Adam;
- manifest registra identidad/configuración/estado;
- RunConfig está versionado y tiene fingerprint estable.

---

## 2. Rutas de ejecución

## 2.1 Training reference

La referencia Foundation usa `train_step`.

Propósito:

- conservar una implementación sencilla;
- servir de oráculo contra optimizaciones;
- aceptar mayor coste de allocations/copias a cambio de claridad.

Esquema conceptual:

```text
token stream
  ↓
deterministic batch materialization
  ↓
forward + loss + backward
  ↓
gradient accumulation
  ↓
gradient normalization / clipping
  ↓
collect flat params
  ↓
Adam
  ↓
write params to model
```

No debe eliminarse mientras siga siendo útil como baseline independiente.

## 2.2 Training Engine productivo

La ruta normal usa `train_step_reuse`.

```text
token stream
  ↓
deterministic windows
  ↓
backward_deterministic_batch_from_stream_into
  ↓
persistent micro/sample gradient buffers
  ↓
prepare_grads
  ↓
Adam over persistent flat parameter mirror
  ↓
Gpt::write_params
```

La selección de ejemplos, orden de acumulación, clipping y actualización Adam deben permanecer semánticamente equivalentes a la referencia.

### Buffers persistentes

`TrainWorkspace` posee:

- `micro_grads`;
- `sample_grads`;
- espejo plano de `params`;
- `BackwardWorkspace`.

Estos buffers se crean fuera del step y se reutilizan.

## 2.3 Eval / inference

El Engine separó el forward de evaluación de la ruta que necesita caches de backward.

La ruta eval/inference no debe construir estructuras cuya única finalidad sea el backward.

`Gpt::logits()` es la superficie normal de logits/inferencia y permanece libre de diagnóstico numérico implícito.

### Limitación actual

La generación autoregresiva todavía no dispone de un contrato Engine completo tipo `ExecutionMode` con todas las especializaciones futuras. La separación relevante de caches ya existe, pero el diseño de una ruta autoregresiva más especializada sigue siendo deuda.

## 2.4 Forward debug

`Gpt::forward_diagnostics` es una API separada.

Usa la ruta interna que ya materializa caches y produce summaries compactos en orden determinista:

por layer:

1. `h1`;
2. `q`;
3. `k`;
4. `v`;
5. attention probabilities;
6. attention output;
7. `h2`;
8. `ff_pre`;
9. `ff_act`;

y después:

10. `h_final`;
11. logits.

En el tiny model de dos layers el benchmark actual produce 20 summaries.

La ruta normal `Gpt::logits()` no se convierte en una ruta diagnóstica.

---

## 3. Ownership y memoria

El objetivo de Engine no es solo reducir bytes totales, sino hacer explícita la vida útil de buffers.

### Invariantes

1. El workspace pertenece al caller del training loop, no a un microbatch.
2. Un buffer reusable se limpia/reset antes de reutilizar contenido que semánticamente debe empezar en cero.
3. No se crean alias mutables incompatibles.
4. La arena usa APIs seguras para obtener múltiples slots disjuntos.
5. El Engine mantiene `unsafe` crudo en cero.
6. Un cambio de memoria no puede alterar silenciosamente el orden matemático si declara equivalencia exacta.

### Baseline integrado de scratch

Última evidencia consolidada:

- allocation calls/step: **264 → 136** en la última gran reducción;
- allocation bytes/step: **1,127,616 → 728,256 B**;
- reallocations: **0**;
- frente al E0 histórico:
  - calls aproximadamente **−90.83 %**;
  - bytes solicitados al heap aproximadamente **−83.03 %**.

En el profiler block=32:

- peak RSS: **4,760 → 4,736 KiB**;
- throughput: **31,163 → 31,710 tok/s**.

Los números pertenecen al protocolo/runner de esos benchmarks; no son una promesa universal para cualquier tamaño de modelo.

---

## 4. Invariantes de shapes

Los kernels y workspaces dependen de shapes explícitos.

Principios actuales:

- los tamaños de buffers deben derivarse de la configuración del modelo;
- los workspaces verifican que siguen correspondiendo al modelo/config actual;
- un shape inválido debe fallar antes de ejecutar un kernel cuando existe una frontera de validación;
- las pruebas incluyen shapes regulares e irregulares;
- no se permiten transposes/copias ocultas como mecanismo de “arreglo” de un mismatch.

La nueva frontera `backend` lleva esta regla a una interfaz explícita:

- `MatrixRef` y `MatrixMut` validan storage;
- matmul valida dimensiones internas;
- el output valida rows/cols esperados;
- el device debe coincidir con el backend.

---

## 5. Parámetros, gradientes y Adam

La ruta Engine mantiene un espejo plano persistente de parámetros dentro de `TrainWorkspace`.

Motivo:

- Adam opera sobre un slice plano;
- volver a recolectar/reasignar el vector completo cada step era coste evitable;
- el modelo recibe el resultado mediante `Gpt::write_params`.

### Invariantes

- longitud de grads/params/Adam debe coincidir;
- `adam.t` define el optimizer step reproducible;
- el estado limpio Engine debe coincidir exactamente con la referencia cuando el orden aritmético se conserva;
- diagnósticos numéricos inspeccionan gradients/params/Adam antes y después del optimizer cuando están activados.

---

## 6. Scratch arena

La arena reusable existe para temporales cuya vida puede limitarse a una fase.

Reglas:

- los slots se reutilizan en vez de reasignarse;
- multi-borrow seguro solo se permite entre slots disjuntos;
- tests cubren reuse entre steps y contextos/shapes variables;
- no se introduce `unsafe` para recuperar rendimiento;
- una optimización de scratch declarada debe pasar el workflow de comparación específico.

### Política CI

Un PR normal debe demostrar **no regresión** de scratch.

Solo un HEAD que declare el trailer exacto:

```text
Auralis-Scratch-Optimization: true
```

activa el gate fuerte histórico de mejora material.

Esto evita exigir “mejoras” a un cambio neutral de `model.rs` sin dejar pasar regresiones reales.

---

## 7. Kernels CPU

## 7.1 Scalar reference

`matmul_reference_into` es el oráculo sencillo.

No se elimina aunque una ruta optimizada sea mejor.

## 7.2 RowSlices productivo

`matmul_row_slices_into` es el kernel optimizado promovido para el matmul forward relevante.

Conserva el orden de acumulación y está cubierto por equivalencia exacta.

## 7.3 Blocked32 no promovido

El candidato Blocked32 tuvo resultados históricos favorables contra scalar, pero la revalidación contra el baseline **actual** RowSlices mostró que era más lento en todos los shapes medidos:

- 32×32×32: candidate/reference **1.3184**;
- 32×32×96: **1.2996**;
- 32×96×32: **1.2282**;
- 32×32×100: **1.3758**;
- 9×13×37: **1.6009**.

Por tanto:

- no existe tabla `Auto` para promover Blocked32;
- RowSlices permanece como optimized path;
- el resultado negativo se conserva como evidencia para evitar repetir la misma hipótesis sin nueva información.

---

## 8. Backend boundary

Phase A está integrada.

La frontera actual cubre matmul y usa views prestadas.

```text
MatrixRef(A) ─┐
              ├─ validate storage / shape / device
MatrixRef(B) ─┤
MatrixMut(O) ─┘
       ↓
     Backend
       ↓
scalar CPU | optimized CPU | mock
```

### Backends actuales

- `ScalarCpuBackend` → scalar reference;
- `OptimizedCpuBackend` → RowSlices;
- `MockBackend` → reference math + traza/call count en un device mock.

### Coste medido de la frontera

En 32×32×32, 7 repeticiones × 200 iteraciones:

- RowSlices directo median: **898,490 ns**;
- `dyn Backend` + validación median: **883,593 ns**;
- candidate/direct: **0.9834**.

La lectura correcta es “no se observó overhead material en este protocolo”, no que el trait haga el kernel mágicamente más rápido.

### Lo que Phase A NO hace

- no selecciona backend globalmente desde `model.rs`;
- no tiene GPU;
- no tiene device allocator/memory pool;
- no transfiere datos host↔device;
- no modifica manifests/config para seleccionar backend.

Eso pertenece a Phase B / Scale.

---

## 9. Diagnósticos numéricos

Los diagnósticos están completos como superficie opt-in.

### CLI

```text
auralis numeric "1 2 3"
auralis numeric --fixture nan-logits
auralis numeric --fixture neg-inf-grad
auralis numeric forward auralis.bin --tokens 1,2,3
auralis train 20 auralis.bin --diagnostics
```

`--diagnostics` es efímero:

- no entra en `RunConfig`;
- no cambia su fingerprint;
- no cambia el schema de checkpoint.

### Training diagnostics

Inspecciona:

- loss;
- gradients;
- parameters;
- Adam m;
- Adam v.

Puede abortar antes del optimizer si la corrupción ya estaba presente.

### Persistencia segura

El gate end-to-end demostró que un Adam corrupto produce error numérico y no sobrescribe checkpoint/manifiesto.

### Overhead

Benchmark final de la superficie CLI:

- OFF wrapper/baseline: **1.0147** (~+1.47 %);
- ON/baseline: **1.3008** (~+30.08 %).

El modo ON es debug; no se activa por defecto.

---

## 10. Observabilidad de rendimiento

## 10.1 Throughput / memoria

Existen métricas de:

- tokens/s;
- tiempo por step;
- allocation calls/bytes;
- peak RSS;
- raw timings de benchmarks.

## 10.2 EngineStepTiming v1

La instrumentación timed es una ruta opt-in separada.

Fases reales actualmente medidas:

- `backward_accum_ns`;
- `grad_process_ns`;
- `optimizer_ns`;
- `writeback_ns`;
- `total_ns`;
- `unattributed_ns`.

Campos no observables en esa frontera se expresan como ausencia explícita:

- `data_ns = null`;
- `checkpoint_ns = null`;
- alloc fields = null si el profiler no está activo.

No se inventan ceros.

### Overhead medido

60 steps × 5 repeticiones:

- OFF/baseline: **1.0011** (~+0.11 %);
- ON/baseline: **1.0129** (~+1.29 %);
- known-phase share mediana: **0.9994**.

---

## 11. Benchmark CLI

El binario empaquetado puede ejecutar los primeros runners sin invocar Cargo:

```text
auralis bench run matmul --warmup 5 --iterations 40 --repeats 5
auralis bench run engine --warmup 3 --iterations 20 --repeats 5
auralis bench run matmul --json
auralis bench run engine --csv
```

Cada `BenchRunRecord` conserva:

- benchmark ID;
- backend;
- hardware;
- toolchain;
- config fingerprint;
- warmup;
- repetitions;
- todos los `raw_ns`;
- resumen mediana/IQR derivado.

La comparación referencia/candidato se calcula sobre records compatibles.

Los bins legacy de matmul y Engine siguen siendo wrappers compatibles porque CI histórico todavía valida sus prefijos de salida.

---

## 12. Checkpoint y reproducibilidad

AURLIS03 puede persistir:

- configuración estructural;
- tokenizer;
- parámetros;
- Adam completo.

El manifest Foundation mantiene identidad reproducible, incluyendo configuración/fingerprint y revision de código.

La reanudación valida el manifest antes de continuar un experimento.

## Soak de resume

Benchmark sostenido actual:

- 300 optimizer steps;
- 38,400 tokens por path;
- continuous vs split/resume al 50 %;
- parámetros + Adam finales exactos;
- bytes finales de checkpoint exactos;
- fingerprint de estado idéntico;
- loss final idéntica: **0.006636993**;
- throughput estable;
- RSS acotado por el gate.

Existe además un protocolo manual-long de 3000 steps con el mismo contrato. Que exista el workflow no implica que ese run largo se haya ejecutado para cada commit.

---

## 13. Política unsafe / SIMD / threads

### Unsafe

La política actual conserva **cero bloques raw unsafe** en el núcleo auditado.

Una mejora no puede rebajar este gate de forma oportunista.

### SIMD

Actualmente no existe SIMD promovido.

Los prototipos no justificaron romper la política vigente; futuras rutas SIMD necesitan evidencia nueva y un contrato seguro aceptado.

### Multithreading

Existe diseño y evidencia experimental, pero no hay worker pool CPU productivo promovido.

El experimento previo se conserva como resultado negativo/parked.

Por tanto no debe documentarse “multithreaded Engine” como capacidad actual.

---

## 14. Evidencia reproducible

Comandos útiles desde un checkout con Rust:

```bash
cargo test --all-targets
cargo run --release -- check
cargo run --release -- train-fresh 3 /tmp/auralis-engine.bin 659918 2 2
cargo run --release -- eval /tmp/auralis-engine.bin

cargo run --release -- bench run matmul --warmup 5 --iterations 40 --repeats 5
cargo run --release -- bench run engine --warmup 3 --iterations 20 --repeats 5

cargo run --release -- numeric --fixture nan-logits
cargo run --release -- numeric forward /tmp/auralis-engine.bin --tokens 0,1,2

cargo run --release --bin auralis_engine_soak -- 300 50
cargo run --release --bin auralis_engine_step_timing_bench -- 5 60 5
cargo run --release --bin auralis_backend_boundary_bench -- 5 200 7
```

Los hosted runners son ruidosos. Las conclusiones de rendimiento importantes deben apoyarse en protocolos side-by-side y múltiples repeticiones.

---

## 15. Límites conocidos

Engine 0.3 todavía no es un motor GPU ni un runtime final.

Pendientes relevantes:

- backend Phase B dentro de un boundary explícito del modelo;
- GPU/device memory real;
- mixed/low precision promovida;
- worker pool productivo;
- SIMD productivo;
- energía/tokens por joule;
- timers separados de data/checkpoint a nivel correcto;
- auditoría adicional de layout/strides;
- ruta autoregresiva más especializada;
- documentación/validación de ownership y shapes seguirá evolucionando con Phase B.

Estos puntos son deuda visible, no capacidades implícitas.

---

## 16. Regla para modificar Engine

Antes de integrar un cambio que declara equivalencia o rendimiento:

1. identificar reference y candidate;
2. declarar qué matemática puede o no cambiar;
3. mantener selección de datos/config comparables;
4. ejecutar tests/equivalencia;
5. ejecutar benchmark side-by-side si hay claim de performance;
6. conservar raw evidence;
7. no reducir gates para hacer pasar el candidate;
8. documentar si el resultado es positivo **o negativo**.

Una hipótesis descartada con evidencia también es progreso: evita complejidad sin valor.
