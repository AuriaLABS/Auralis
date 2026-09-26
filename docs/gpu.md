# GPU single-device contract (#68)

`GPU_SCHEMA_VERSION = 1`

Single-device GPU boundary on top of #67 buffers.

- `GpuRuntime::detect()` is unavailable in CI;
- hardware `kernel_matmul` fails closed;
- `staged_matmul` copies host→gpu then falls back to scalar CPU;
- the product default stays OptimizedCpu.

Does **not** enable GPU by default and does **not** claim a measured GPU speedup.
