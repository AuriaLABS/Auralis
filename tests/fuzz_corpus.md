# Fuzz corpus (#108)

Seeded mutator `0xF022_0108`, 48 trials, mutant cap 4096 bytes.

| id | input | expected |
| --- | --- | --- |
| M0 | empty manifest | decode error |
| M1 | only `auralis_manifest=3` | decode error |
| C0 | magic `NOTAURALIS` | InvalidData |
| C1 | AURLIS02 + char tok + string length `u32::MAX` | InvalidData, no alloc |
| C2 | truncated `AURLIS02` header | UnexpectedEof or InvalidData |
| R0 | empty run config | decode error |
| R1 | only schema line | decode error |
| R2 | unknown field | decode error |

If a trial panics, minimize the blob, add a row here and a fixture test.
