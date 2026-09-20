# Auralis — hoja de ruta

Horizonte inicial: **septiembre de 2026 → septiembre de 2027** para construir una v1.0 sólida.
Horizonte posterior: investigación abierta y evolución continua hacia capacidades de frontera.

Auralis no medirá su progreso por promesas ni por número de parámetros. Cada etapa debe entregar software que corre, métricas reproducibles y una mejora observable sobre la etapa anterior.

## Norte del proyecto

Construir una inteligencia artificial propia desde los fundamentos, desarrollada de forma iterativa con ayuda de IA, capaz de aprender, razonar, usar herramientas, mantener memoria, investigar y participar de forma controlada en la mejora de su propia arquitectura y software.

La aspiración de largo plazo es competir con los sistemas de IA más avanzados por **capacidad real y eficiencia**, no solo por escala bruta.

---

# Fase 0 — Genesis

## Auralis 0.1.0 — Primer cerebro funcional

Objetivo: cerrar el ciclo completo de aprendizaje desde cero.

Debe existir y estar probado:

- tensores y operaciones numéricas esenciales;
- tokenizador y BPE;
- transformer decoder-only funcional;
- forward pass;
- función de pérdida;
- backpropagation explícito;
- Adam;
- actualización de parámetros;
- gradient checks;
- checkpoint con pesos + estado del optimizador;
- reanudación de entrenamiento;
- evaluación;
- generación autoregresiva de texto;
- CLI reproducible;
- tests automáticos del núcleo.

**Criterio de salida:** Auralis aprende un corpus desde pesos inicializados desde cero, reduce la pérdida, guarda el estado, lo recupera y vuelve a generar texto sin depender de un framework de deep learning externo.

## Auralis 0.2 — Foundation

- batching real;
- datasets mayores y pipeline de datos;
- separación train/validation/test;
- perplexity y métricas de entrenamiento;
- sampling configurable: temperature, top-k, top-p;
- seeds y reproducibilidad;
- serialización versionada;
- perfiles de memoria y tok/s;
- mejores pruebas numéricas.

## Auralis 0.3 — Engine

Objetivo: dejar de ser solo correcto y empezar a ser rápido.

Estado técnico integrado y límites actuales: [docs/ENGINE_ARCHITECTURE.md](docs/ENGINE_ARCHITECTURE.md). El listado siguiente es roadmap de la fase, no implica que cada línea esté ya promovida.

- multithreading;
- SIMD donde aporte beneficio medido;
- kernels optimizados;
- reducción de asignaciones y copias;
- mixed/low precision experimental;
- abstracción de backend;
- primeros backends GPU cuando el núcleo CPU sea estable;
- benchmarks automatizados de velocidad, RAM y rendimiento por watt cuando sea posible.

## Auralis 0.4 — Brain

- múltiples capas y cabezas configurables;
- contextos mayores;
- positional encodings experimentales;
- variantes de atención;
- memoria externa/persistente;
- razonamiento recurrente experimental;
- MoE u otras arquitecturas cuando exista una hipótesis y benchmark que lo justifique.

## Auralis 0.5 — Agent

- REPL robusto;
- herramientas explícitas;
- ejecución de acciones con permisos definidos;
- memoria de sesión y memoria persistente;
- planificación y descomposición de tareas;
- servidor/API local;
- trazas y observabilidad del agente.

## Auralis 0.6 — Scientist

Auralis comienza a participar formalmente en su propia investigación.

Ciclo de trabajo:

```text
hipótesis
  ↓
propuesta de cambio
  ↓
branch/experimento
  ↓
tests
  ↓
entrenamiento controlado
  ↓
benchmarks/evaluaciones
  ↓
comparación contra baseline
  ↓
revisión
  ↓
aceptar o descartar
```

Ninguna modificación se considera mejora porque la IA lo afirme. Debe superar criterios previamente definidos.

## Auralis 0.7 — Multimodal

- representación común o interoperable para texto, código, imagen y audio;
- encoders/decoders propios cuando sea viable;
- datasets multimodales versionados;
- evaluaciones por modalidad;
- fusión multimodal experimental.

## Auralis 0.8 — Scale

- entrenamiento multi-GPU;
- distribución entre nodos;
- sharding de parámetros/optimizador cuando sea necesario;
- checkpoint distribuido;
- recuperación de fallos;
- telemetría de entrenamiento;
- escalado basado en benchmarks, no por tamaño arbitrario.

## Auralis 0.9 — Research Platform

- harness de experimentos reproducibles;
- registro de hipótesis y resultados;
- comparación automática con baselines;
- suites de razonamiento, código, memoria y agentes;
- búsqueda de arquitecturas;
- experimentación con nuevas funciones de pérdida, memoria, atención y entrenamiento;
- selección empírica de cambios.

## Auralis 1.0 — Sistema completo y reproducible

Una v1.0 no significa “IA final”. Significa que Auralis se ha convertido en una plataforma de IA propia, entrenable y evaluable de extremo a extremo.

Debe incluir:

- motor de entrenamiento estable;
- inferencia estable;
- documentación técnica;
- formatos versionados;
- benchmarks publicados;
- límites conocidos;
- tarjeta de modelo;
- binarios reproducibles;
- suite de evaluación;
- tag `v1.0.0`.

---

# Después de v1.0 — Frontier Program

## 1. Eficiencia radical

Buscar mayor inteligencia por FLOP, byte, dato y joule. La escala solo se aumenta cuando los experimentos indiquen que compensa.

## 2. Arquitectura propia

El transformer es una base inicial, no una doctrina. Auralis podrá explorar arquitecturas híbridas, memoria diferenciable, recurrencia, sparsity, MoE, routing, búsqueda, planificación y mecanismos nuevos.

## 3. Aprendizaje continuo

Investigar cómo incorporar conocimiento nuevo sin reentrenar todo el sistema ni destruir capacidades anteriores.

## 4. Razonamiento y memoria

Construir evaluaciones específicas de razonamiento largo, planificación, memoria episódica, recuperación y consistencia a largo plazo.

## 5. Multimodalidad profunda

Pasar de conectar modalidades a razonar de forma conjunta sobre ellas.

## 6. Investigación asistida por Auralis

Auralis podrá analizar sus propios resultados, proponer experimentos, generar implementaciones y comparar alternativas. El proceso seguirá siendo trazable, reproducible y sujeto a criterios externos de aceptación.

## 7. Mejora controlada del propio sistema

La meta no es un bucle ciego de autoedición. La meta es una fábrica de investigación donde Auralis pueda producir candidatos de mejora y un sistema de evaluación independiente determine si realmente aportan valor.

## 8. Capacidades de frontera

Comparar Auralis con sistemas punteros usando evaluaciones públicas, internas y benchmarks de eficiencia. Solo se afirmará una ventaja cuando los datos reproducibles la respalden.

---

# Regla de oro

**Auralis no avanza porque una versión tenga un nombre mayor. Auralis avanza cuando una capacidad nueva funciona, se mide y supera el baseline.**
