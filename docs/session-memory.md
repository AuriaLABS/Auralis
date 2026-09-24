# Session memory contract (#98)

`SESSION_MEMORY_SCHEMA_VERSION = 1`

Optional short-term store for events, notes and tool results. It does **not** touch model weights.

- retrieval by recency, kind and tag;
- capacity limit with explicit evict-oldest;
- reset clears records without changing the model;
- export/import via `canonical` / `load`;
- two session ids never share records;
- `SessionMemory::off` keeps the API; remember/retrieve are no-ops.

No trainable model memory, no cross-session index.
