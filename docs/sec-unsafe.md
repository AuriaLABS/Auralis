# Unsafe and parser limits

## Unsafe inventory

Expected `unsafe` in library/engine `src/` (excluding `src/bin`): **0**.

Allowed outside the library contract:

- `src/bin/auralis_profile.rs`
- `src/bin/auralis_eval_profile.rs`
- `src/bin/auralis_train_profile.rs`

Those binaries implement a counting `GlobalAlloc` over `std::alloc::System`
so E1 allocation profiles can run. They are not on the train/eval/chat path.

If a future kernel needs raw `unsafe` in the library:

1. Keep `EXPECTED_UNSAFE_BLOCKS = 0` while the exception is only proposed.
2. Open a separate security review naming the exact file/function/ISA and memory invariants.
3. Keep a safe portable reference path and equivalence tests.
4. Before merging any exception, replace the count-only contract with an exact fail-closed allowlist for the approved sites.
5. Only then update the expected contract to match that reviewed allowlist.

Merely increasing `EXPECTED_UNSAFE_BLOCKS` is not an approved exception path. See ADR 0001.

Run `auralis sec-audit [SRC_ROOT]` before merging engine changes.

## Parser / resource limits

| Surface | Limit | Failure mode |
| --- | --- | --- |
| checkpoint strings | `MAX_CHECKPOINT_STRING` = 1 MiB | `InvalidData` |
| BPE tables | `MAX_BPE_TABLE` = 65536 | `InvalidData` |
| model dims | vocab/embd/head/layer/block/ff caps in `validate_cfg` | `InvalidData` |
| run config | unknown/duplicate/future fields | decode error |
| release artifacts | reject `..` and absolute paths | error |
| untrusted paths | `sec_path::confine` — no `..` escape, no absolute, max 32 components | error |
| size arithmetic | `sec_overflow::{checked_add,checked_mul,checked_product}` | error, never wrap |

Malformed checkpoints and configs must fail closed. They must not panic into parameter writes.
