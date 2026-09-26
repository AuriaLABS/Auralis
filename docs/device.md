# Device storage contract (#67)

`DEVICE_SCHEMA_VERSION = 1`

Owned buffers and explicit host↔device copies on top of the Engine backend boundary.

- CPU remains the default host device;
- a mock device can hold a copy without rewriting the model;
- `as_host` on a non-CPU buffer fails as an implicit copy;
- provenance records backend + device for manifests.

Does **not** implement GPU kernels. That stays #68.
