# Campaign scheduler contract (#115)

`CAMPAIGN_SCHEDULER_SCHEMA_VERSION = 1`

Local persistent scheduler over #76 campaign slots.

- queued/running/blocked/done/failed/cancelled;
- pause/resume and snapshot restore;
- completed RunSpecs are not repeated;
- budget overflow stops scheduling explicitly;
- orphan running jobs can be reclaimed.

Does **not** require a cloud scheduler or pick a winner.
