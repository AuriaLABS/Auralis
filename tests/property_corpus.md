# Property regression corpus (#107)

Pinned witnesses. If a property test finds a new failure, shrink it and add the
minimal case here plus a fixture in `tests/properties.rs`.

| id | kind | witness | expected |
| --- | --- | --- | --- |
| C0 | config | vocab=0, all zeros | reject |
| C1 | config | vocab=1, otherwise tiny-valid | reject |
| C2 | config | n_embd=6, n_head=4 | reject (not divisible) |
| C3 | config | all ones except vocab=2 | accept |
| M0 | mask | t=1 `[true]` | causal |
| M1 | mask | t=2 fully unmasked | not causal |

Generator seed: `0xA11CE_0107`. Trials per property: 64 (16 for attention).
