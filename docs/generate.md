# Generation contract (#300)

`GENERATE_SCHEMA_VERSION = 1`

Greedy deterministic path for the supported v1 profile.

- same prompt yields the same tokens;
- empty prompt and context overflow fail closed;
- sampling is not promoted.

Does **not** add a remote server or change architecture for a release metric.
