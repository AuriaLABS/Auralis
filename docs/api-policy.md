# Public API policy (#302)

`API_POLICY_SCHEMA_VERSION = 1`

What v1 promises, and what it does not.

- Public surfaces receive semver (cli, checkpoint-v1, local-api, session);
- Internal stays unsupported even when the item is `pub` in Rust: internal is not stable by visibility;
- Experimental may break without a major;
- a Public breaking change is a major; an additive Public change is a minor;
- deprecation of Public requires an announced minor first.

Does **not** freeze every pub item.
