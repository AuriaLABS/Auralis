# Lab catalog contract (#58)

`LAB_CATALOG_SCHEMA_VERSION = 1`

Searchable catalog of prior #52/#54 evidence.

- queries recover previous attempts with commit/config/budget;
- positive, negative and inconclusive outcomes stay distinct;
- incomparable metadata is not collapsed;
- a duplicate proposal can be flagged before spending compute.

Does **not** treat historical correlation as an automatic decision.
