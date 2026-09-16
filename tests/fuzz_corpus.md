# Fuzz corpus (#108)

Seeded mutator `0xF022_0108`, 48 trials, mutant cap 4096 bytes.

| id | input | expected |
| --- | --- | --- |
| M0 | empty manifest | decode error |
| M1 | only `auralis_manifest=3` | decode error |
| C0 | magic `NOTAURALIS` | InvalidData |
| C1 | AURLIS02 + char tok + string length `u32::MAX` | InvalidData, no alloc |

If a trial panics, minimize the blob, add a row here and a fixture test.
