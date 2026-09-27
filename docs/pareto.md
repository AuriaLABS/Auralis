# Multi-objective comparison contract (#77)

`PARETO_SCHEMA_VERSION = 1`

Compare candidates without a single score gate.

- quality/cost/memory/latency stay separate;
- dominance is distinct from a real trade-off;
- missing metrics make the pair incomparable;
- the Pareto front drops dominated complete rows only.

Does **not** invent universal metric weights.
