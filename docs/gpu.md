# GPU single-device contract (#68)

`GPU_SCHEMA_VERSION = 1`

Single-device GPU boundary on top of #67 buffers.

- `GpuRuntime::detect()` is unavailable in CI;
- hardware `kernel_matmul` fails closed;
- `staged_matmul` copies host→gpu then falls back to OptimizedCpu and records transfer cost;
- the product default stays OptimizedCpu;
- Engine `DeviceId` / `BackendId` are not extended in this slice.

Does **not** enable GPU by default and does **not** claim a measured GPU speedup.
