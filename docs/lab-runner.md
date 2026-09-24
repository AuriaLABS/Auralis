# Lab runner contract (#53)

`LAB_RUNNER_SCHEMA_VERSION = 1`

Controlled runner for #52 experiment specs.

- only allowlisted fixtures: `mock-eval`, `mock-timeout`, `mock-corrupt`;
- timeout and corrupt artifacts fail closed (no partial success);
- results bind `run_id`, commit and config;
- the same spec + seed produce an equivalent mock environment.

Does **not** run arbitrary commands or a distributed scheduler.
