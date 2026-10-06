# RC go/no-go contract (#84)

`RC_SCHEMA_VERSION = 1`

Checklist before `v1.0.0`.

- formats, provenance, soak and model card must be present;
- automated RC stays no-go until human_approval;
- `tag_v1` never creates a git tag;
- `release-check --approve` and `release-manifest --approve` exit with `approval forbidden`;
- `release-check --create-tag` and `release-manifest --create-tag` exit with `tag creation forbidden`.

Does **not** cut `v1.0.0` from this slice.
