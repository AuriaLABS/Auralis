# Lab review contract (#54)

`LAB_REVIEW_SCHEMA_VERSION = 1`

Independent review of #52/#53 results.

- criteria are read from the original spec;
- reviewer identity must differ from the implementer;
- non-comparable evidence is `inconclusive`, never `pass`;
- an intentional false-positive fixture is rejected.

Does **not** rewrite criteria after seeing results, auto-merge or edit code.
