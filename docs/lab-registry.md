# Lab registry contract (#52)

`LAB_REGISTRY_SCHEMA_VERSION = 1`

Versioned local registry of hypotheses, experiment designs, results and
decisions.

- running requires a baseline and a success criterion;
- negative results persist the same way as positive ones;
- two runs of the same design differ by `run_id`;
- `canonical` / `load` is machine-readable for other agents.

Does **not** auto-merge, launch remote jobs or edit production weights.
