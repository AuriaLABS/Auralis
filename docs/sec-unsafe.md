# Unsafe and parser limits

## Unsafe inventory

Expected `unsafe` in library/engine `src/` (excluding `src/bin`): **0**.

Allowed outside the library contract:

- `src/bin/auralis_profile.rs`
- `src/bin/auralis_eval_profile.rs`
- `src/bin/auralis_train_profile.rs`

Those binaries implement a counting `GlobalAlloc` over `std::alloc::System`
so E1 allocation profiles can run. They are not on the train/eval/chat path.

If a future kernel needs `unsafe` in the library:

1. Raise `EXPECTED_UNSAFE_BLOCKS` in `src/sec.rs`.
2. Document the invariant next to the block and in this file.
3. Keep a safe reference path for equivalence tests.

Run `auralis sec-audit [SRC_ROOT]` before merging engine changes.

## Parser / resource limits

| Surface | Limit | Failure mode |
| --- | --- | --- |
| checkpoint strings | `MAX_CHECKPOINT_STRING` = 1 MiB | `InvalidData` |
| BPE tables | `MAX_BPE_TABLE` = 65536 | `InvalidData` |
| model dims | vocab/embd/head/layer/block/ff caps in `validate_cfg` | `InvalidData` |
| run config | unknown/duplicate/future fields | decode error |
| release artifacts | reject `..` and absolute paths | error |

Malformed checkpoints and configs must fail closed. They must not panic into parameter writes.
