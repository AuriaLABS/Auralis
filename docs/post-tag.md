# Post-tag verification contract (#305)

`POST_TAG_SCHEMA_VERSION = 1`

What closes a published v1 tag, and what this slice still refuses to do.

- tag source artifact and manifest must share one SHA;
- mismatch is a release incident;
- tagged artifacts are immutable;
- smoke uses the published payload, not the dev checkout;
- never creates the tag;
- no features land in the post-tag check.

Does **not** cut `v1.0.0` or rewrite artifacts already bound to a tag.
