# Campaign runner contract (#76)

`CAMPAIGN_SCHEMA_VERSION = 1`

Batteries of candidates × suites × seeds under a global budget.

- each result keeps `run_id` and the campaign spec id;
- resume skips completed runs;
- infrastructure failures are distinct from experiment failures;
- exceeding the budget is an explicit error, never a silent success.

Does **not** pick a winner or change criteria mid-campaign.
