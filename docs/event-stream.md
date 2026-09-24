# Event stream contract (#97)

`EVENT_STREAM_SCHEMA_VERSION = 1`

In-process stream for token/plan/tool/permission/error/cancel events.

- monotonic sequence IDs;
- bounded buffer; a full buffer returns `Backpressured` and does not grow;
- `cancel` records a cancel event; later mutative pushes return `Cancelled`;
- `load`/`canonical` replay preserves order;
- incompatible schema is rejected;
- payloads go through the same redaction as #47.

No public network fan-out, no unbounded queue, no orphan mutative event after cancel.
