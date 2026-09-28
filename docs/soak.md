# Release soak contract (#82)

`SOAK_SCHEMA_VERSION = 1`

Long-run stability for the supported v1 profile.

- resume keeps the same fingerprint;
- RSS above the cap fails closed;
- retries are not used to hide flakes (`MAX_RETRIES = 0`).

Does **not** turn optional experiments into release blockers.
