# Architectural search contract (#78)

`SEARCH_SCHEMA_VERSION = 1`

Budgeted deterministic grid over a versioned SearchSpace.

- same SearchSpec produces the same candidate sequence;
- budget and max_params are hard caps;
- invalid layers/heads fail before training;
- discarded candidates stay on the record.

Does **not** auto-merge candidates or expand the search space.
