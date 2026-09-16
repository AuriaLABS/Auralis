# E3.0 — Paralelismo CPU determinista

Parent: #4. Issue: #31. Depende de E2 estable (#26 mergeado, #30 candidato en #151).

Principio: **primero el orden aritmético; después los threads.**

## Decisión

1. Default de producción y de CI: **1 thread**, mismo kernel que hoy.
2. Paralelismo opt-in (`AURALIS_THREADS=N` o `RunConfig.threads`) con pool **proceso-global**, no un pool por matmul.
3. Primera ola: **particionar filas de matmul** (`i` en `0..rows`) porque cada celda conserva `k=0..inner` y el orden intra-fila no cambia.
4. No reducir en paralelo floats de distintas filas hacia un mismo acumulador.
5. Cualquier kernel que cambie el orden `k` pasa a contrato *tolerant* y **no** puede ser default.

## Alternativas descartadas

| Idea | Por qué no |
| --- | --- |
| Rayon por defecto en `kernels::matmul` | oculta no-determinismo; CI dejaría de ser bit-idéntico |
| Thread por operación | oversubscription + jitter |
| Paralelizar `k` (reducción) | cambia asociatividad de `f32`; rompe fingerprint |
| GPU / multi-proceso | fuera de E3 |
| Reorder de capas | cambia semántica del modelo |

## Mapa de operaciones

| Op | ¿Paralelo? | Granularidad | Contrato |
| --- | --- | --- | --- |
| `matmul` / `matmul_into` | sí, 1ª ola | tiles de filas ≥ 8 | exact si el orden `k` se conserva |
| attention scores `T×T` | sí, 2ª ola | filas de query | exact si softmax sigue siendo por fila serial |
| softmax / gelu / add | no (salvo T grande) | — | exact; poco trabajo |
| embedding gather | no | — | exact |
| Adam step | no en E3 | — | exact; fingerprint de momentos |
| loss / cross-entropy | no | — | exact |
| BPE encode | no | — | no es el cuello |

Tamaños actuales (`Config::tiny`, block 32): un matmul 32×32×96 cabe en L1. **No se espera speedup material en tiny.** El diseño existe para cuando `n_embd`/`block` crezcan; el bench debe decirlo, no el código.

## Determinismo

- `threads == 1` ⇒ bitwise igual a `matmul_reference` / kernel slice actual (#26).
- `threads > 1` con partición por filas y `k` serial ⇒ **también** bitwise igual (cada celda es independiente).
- Si un experimento reordena `k` o usa reduction tree: etiqueta `numeric=tolerant` y compara contra #29 con atol/rtol explícitos. No entra a `main` como default.
- Seed, data order, Adam `t` y fingerprint de manifiestos no cambian.
- Prohibido `thread_rng` en kernels. El RNG de init/sampling sigue fuera.

## Pool

- Un `ThreadPool` lazy, tamaño `min(N, available_parallelism(), 8)` para no saturar CI.
- Nunca crear threads dentro de `backward` por layer.
- Work-stealing permitido solo si la unidad de trabajo es una fila completa (resultado independiente).
- Fallback: si el pool falla al arrancar → 1 thread y warning, no panic.

## Benchmark

Mismo protocolo que #26/#151:

```
AURALIS_THREADS=1 cargo run --release --bin auralis_matmul_bench -- 5 50 7
AURALIS_THREADS=2 cargo run --release --bin auralis_matmul_bench -- 5 50 7
AURALIS_THREADS=4 cargo run --release --bin auralis_matmul_bench -- 5 50 7
```

Reportar por shape: mediana µs, ratio vs 1 thread, tok/s de `tiny` train 20 steps si el kernel está en el hot path.

Regla de promoción: mediana ≥ 1.3× en **dos** shapes representativos y 0 fallos de exactitud. Si no, el pool existe pero el dispatcher elige 1 thread.

## Tickets siguientes (no este PR)

1. `src/threads.rs`: pool + `effective_threads()` leyendo env/config. Tests: 1 thread idéntico.
2. `matmul_rows_parallel_into` junto a `cpu_matmul` (#30), **no** reemplaza `kernels`.
3. Bench 1/2/4 en el mismo runner que #151.
4. Flag en `RunConfig` versionado (#91) `threads: u16` default 1.
5. Recién entonces, dispatcher en hot path. Requiere #29 verde y #30 promovido o rechazado.

## Invariantes de #2 / #4 que no se tocan

- loss/logits bitwise en modo reference
- resume + manifest fingerprint
- Adam state
- sampling greedy seed
- un solo orden de batch

## Coste esperado hoy

En tiny: overhead de spawn/sync ≥ trabajo útil. Por eso el default es 1. E3 no se implementa para presumir de threads; se implementa para no pintarlos encima del motor cuando alguien los pida.
