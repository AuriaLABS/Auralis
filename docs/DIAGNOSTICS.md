# Diagnósticos numéricos

Los diagnósticos de Auralis son explícitos y opt-in. No forman parte de `RunConfig` ni del formato de checkpoint.

## Scan y fixtures

```text
auralis numeric "1 2 3"
auralis numeric --fixture nan-logits
auralis numeric --fixture neg-inf-grad
```

Un scan finito termina con código 0. Una corrupción numérica detectada termina con código 3. Un error de argumentos o carga termina con código 2.

## Forward real desde checkpoint

```text
auralis numeric forward auralis.bin --tokens 1,2,3
```

Los IDs deben estar dentro del vocabulario y no superar el block del modelo. La salida `numeric_tensor` enumera summaries compactos por layer/tensor; no copia activaciones al reporte.

## Entrenamiento

```text
auralis train 20 auralis.bin --diagnostics
auralis train-fresh 20 scratch.bin --diagnostics
```

`--diagnostics` es una opción CLI efímera. Sin el flag se usa la ruta normal `train_step_reuse`. Con el flag se comprueban loss, gradientes, parámetros y estado Adam antes/después del optimizer.

Si se detecta NaN/Inf durante un step, el entrenamiento aborta antes de la escritura final del checkpoint/manifiesto. La opción no modifica el fingerprint de `RunConfig`, el schema de configuración ni el formato AURLIS del checkpoint.

El modo diagnóstico tiene coste deliberado de debug; no debe activarse por defecto.
