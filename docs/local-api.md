# Local API contract (#50)

`LOCAL_API_SCHEMA_VERSION = 1`

The API and the REPL share `SessionRepl`. This crate does **not** add a
public HTTP stack. `LocalAgentApi::handle` is the runtime; `serve_one`
accepts a single request on `127.0.0.1` only.

## Routes

- `GET /v1/version`
- `GET /v1/tools` — registry listing, no execution
- `GET /v1/capabilities`
- `POST /v1/sessions` — body `id=...` optional
- `GET /v1/sessions/{id}`
- `POST /v1/sessions/{id}/messages` — same commands as `SessionRepl::handle`
- `POST /v1/sessions/{id}/close`

`X-Correlation-Id` is echoed. Limits: `max_body_bytes`, `max_sessions`,
`max_concurrent`, `max_wall_ms`. Incompatible session schema → 409.

## Out of scope

Public bind, multi-tenant, OAuth, distributed scale, bypass of #45.
