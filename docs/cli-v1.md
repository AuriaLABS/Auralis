# CLI v1 contract (#301)

`CLI_V1_SCHEMA_VERSION = 1`

Stable exit codes for the supported command surface.

- 0 success, 2 input, 3 runtime, 4 corrupt;
- `numeric` is Experimental;
- exit 0 is never a partial failure.

Does **not** change persistent formats to uniform CLI text.
