# Format freeze contract (#80)

`COMPAT_SCHEMA_VERSION = 1`

v1 freeze for checkpoint, manifest and session formats.

- current versions are explicit;
- unknown or future versions fail closed;
- version 0 is corrupt, not a silent default;
- migrate() only accepts the frozen v1 and never rewrites semantics.

Does **not** freeze private internals or invent silent migrations.
