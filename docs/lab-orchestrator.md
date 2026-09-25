# Lab orchestrator contract (#56)

`LAB_ORCHESTRATOR_SCHEMA_VERSION = 1`

Coordinates multiple researcher agents on distinct proposals.

- exclusive scopes cannot be double-claimed;
- implementer and reviewer must be distinct actors;
- handoffs are explicit records;
- equivalent proposals are rejected as duplicates;
- incompatible conclusions are not merged.

Does **not** auto-merge, grant security permissions or use private agent chat.
