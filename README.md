# Auralis

**Auralis es un proyecto de inteligencia artificial construido desde sus fundamentos, desarrollado de forma iterativa con ayuda de IA y orientado a alcanzar capacidades de frontera mediante arquitectura propia, eficiencia y experimentación verificable.**

> Auralis no pretende ser un wrapper de otro modelo. Auralis pretende ser el modelo.

## Estado actual — Genesis

Auralis está en una fase experimental temprana. El núcleo actual está escrito en Rust e incluye un transformer decoder-only, tokenización, backpropagation explícito, Adam, checkpoints y herramientas de entrenamiento/evaluación.

La ambición a largo plazo es muy alta: evolucionar desde un modelo mínimo que aprende desde cero hacia un sistema general, multimodal, eficiente, capaz de usar herramientas, mantener memoria, investigar y participar en la mejora de su propio código y arquitectura bajo un proceso controlado y medible.

Ser “la IA más potente” no se considera una promesa ni un eslogan verificable hoy. Se considera una **dirección de investigación**: acercarse progresivamente al estado del arte y superarlo allí donde los resultados medidos lo demuestren.

## Principios del proyecto

1. **Desde los fundamentos.** El núcleo del modelo debe ser nuestro: entrenamiento, inferencia, optimización y arquitectura.
2. **Rust como base.** Seguridad, rendimiento, control de memoria y capacidad de evolucionar hacia CPU/GPU/distribuido.
3. **IA desarrollando IA.** La IA puede diseñar, programar, revisar y proponer experimentos para Auralis.
4. **Nada se acepta por intuición.** Una mejora debe demostrar valor mediante tests, benchmarks o evaluaciones reproducibles.
5. **Eficiencia antes que tamaño.** No perseguir parámetros por sí mismos; maximizar capacidad por unidad de cómputo, memoria, energía y datos.
6. **Arquitectura evolutiva.** El transformer actual es el punto de partida, no una restricción permanente.
7. **Mejora controlada.** Las modificaciones propuestas por IA pasan por código, tests, benchmarks, revisión y una decisión explícita antes de integrarse.
8. **Capacidad real antes que marketing.** Cada versión debe ejecutar algo nuevo y medible.

## Objetivo inmediato — Auralis 0.1.0 Genesis

La primera versión base se considera completa cuando Auralis pueda realizar de forma reproducible este ciclo completo:

```text
texto
  ↓
tokenizador Auralis
  ↓
modelo Auralis
  ↓
forward
  ↓
loss
  ↓
backprop
  ↓
Adam
  ↓
actualización de pesos
  ↓
checkpoint
  ↓
carga del checkpoint
  ↓
inferencia
  ↓
generación de texto
```

Además, `cargo test` deberá validar los componentes críticos y el entrenamiento deberá poder reanudarse desde checkpoint sin perder el estado del optimizador.

## Ejecución actual

```bash
cargo test
cargo run --release -- check
cargo run --release -- train 80 auralis.bin
cargo run --release -- eval auralis.bin
cargo run --release -- chat auralis.bin
```

`train` reanuda checkpoint AURLIS03 (pesos + Adam).
`train-fresh` parte de cero.

Cómo probar cada pieza: [docs/verify.md](docs/verify.md).
Verbos CLI: [docs/cli.md](docs/cli.md).

## Dirección a largo plazo

Auralis evolucionará por capas:

**Core → Engine → Brain → Agent → Scientist → Multimodal → Scale → Research → Controlled Self-Improvement → Frontier**

El objetivo no es copiar indefinidamente arquitecturas existentes. Primero construiremos una base correcta y medible; después Auralis deberá convertirse en una plataforma donde sea posible investigar arquitecturas, memoria, razonamiento, entrenamiento y sistemas de agentes propios.

## Documentación

- [VISION.md](VISION.md) — misión, identidad y principios de largo plazo.
- [ROADMAP.md](ROADMAP.md) — hitos técnicos y evolución del proyecto.
- [docs/verify.md](docs/verify.md) — comandos de prueba sin red.
- [docs/cli.md](docs/cli.md) — verbos y flags.
- [docs/versions.md](docs/versions.md) — crate y schemas vigentes.
- [docs/ci-pruebas.md](docs/ci-pruebas.md) — matriz CI.
- [docs/sec-unsafe.md](docs/sec-unsafe.md) — unsafe y límites.
- [docs/bench-registry.md](docs/bench-registry.md) — catálogo de benches.
- [docs/brain-ab.md](docs/brain-ab.md) — harness A/B reproducible para experimentos Brain.
- [docs/alibi-experiment.md](docs/alibi-experiment.md) — contrato, A/B y benchmark por contexto del experimento ALiBi.
- [docs/optimizer-boundary.md](docs/optimizer-boundary.md) — boundary de optimizer y estado Adam versionado.
- [docs/optimizer-variants.md](docs/optimizer-variants.md) — AdamW/Lion experimentales bajo el mismo contrato.
- [docs/gradient-clipping.md](docs/gradient-clipping.md) — clipping global-norm ON/OFF, métricas y migración RunConfig.
- [docs/ENGINE_ARCHITECTURE.md](docs/ENGINE_ARCHITECTURE.md) — arquitectura, invariantes y evidencia actual de Engine 0.3.

## Licencia

MIT
