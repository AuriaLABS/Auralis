# Post-tag verification contract (#305)

`POST_TAG_SCHEMA_VERSION = 1`

What happens after an approved RC is tagged, and what must never happen.

- tag, source, binaries, checksums and manifest converge to the same commit SHA;
- mismatch is a release incident;
- never mutate tagged artifacts;
- rebuild and smoke run from the published tag, not the live checkout;
- known issues and deferred debt are recorded as post-v1 follow-ups;
- this contract never creates a git tag;
- `release-check --create-tag` and `release-manifest --create-tag` exit with `tag creation forbidden`;
- `release-check --approve` and `release-manifest --approve` exit with `approval forbidden`.

Does **not** cut `v1.0.0` or rewrite published artifacts after the tag exists.
