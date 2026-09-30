# Release baselines contract (#304)

`BASELINES_SCHEMA_VERSION = 1`

What an RC performance gate may decide, and what it may not.

- thresholds are published before the candidate is scored;
- comparisons require the same config and the same runner;
- incomparable runners are inconclusive;
- a win on one metric does not hide a blocking regression;
- warmup and repeats are fixed (`WARMUP=2`, `REPEATS=3`);
- raw metrics stay attached to the release report;
- GPU is not a release gate.

Does **not** cut `v1.0.0` or invent a single score that overrides per-metric fails.
