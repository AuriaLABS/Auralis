# ADR 0001 — Unsafe SIMD policy

- Status: Proposed
- Date: 2026-09-17
- Issues: #146, #200, #109
- Evidence: PR #199

## Context

Auralis currently enforces `EXPECTED_UNSAFE_BLOCKS = 0` for library/engine code under `src/` (excluding profiling binaries). The contract is checked by CI through `sec::tests::crate_src_matches_expected_contract`.

An AVX2 matmul prototype in PR #199 demonstrated exact output against the portable reference and repeatable material speedups for two measured shapes:

- `32x32x32`: about `1.82x`
- `32x96x32`: about `1.85x`

The prototype required five raw-`unsafe` sites for x86 intrinsics. General CI rejected the branch even though its dedicated equivalence/benchmark workflow passed. That rejection is considered correct behavior under the current contract.

## Decision

Auralis keeps **zero raw `unsafe` in library/engine code as the default and current policy**.

Performance evidence alone is not sufficient to relax this invariant. SIMD implementations requiring raw `unsafe` in Auralis-owned library code must not be merged until a separate security review explicitly approves a narrowly scoped exception.

A future exception is possible, but it is opt-in and fail-closed. It must be justified per implementation rather than establishing a general permission for unsafe optimization.

## Inventory of the PR #199 prototype

The security scanner reported five raw-`unsafe` hits in `src/simd.rs`. They map to the following concrete boundary:

| Hit | Prototype site | Classification | Required invariant |
| --- | --- | --- | --- |
| 1 | `unsafe { matmul_avx2_into(...) }` | target-feature call boundary | `resolve_backend` must have confirmed AVX2 at runtime before the call; input/output slice lengths must already match the declared matrix shapes. |
| 2 | `unsafe fn matmul_avx2_into` | architecture-specific function boundary | the function is only reachable through the checked dispatch; no caller may bypass ISA detection. |
| 3 | `_mm256_loadu_ps(out_row.as_ptr().add(j))` | unaligned load | `j + 8 <= cols` guarantees eight in-range `f32` values in the current output row. |
| 4 | `_mm256_loadu_ps(b_row.as_ptr().add(j))` | unaligned load | the same loop bound guarantees eight in-range `f32` values in the current weight row. |
| 5 | `_mm256_storeu_ps(out_row.as_mut_ptr().add(j), sum)` | unaligned store | the destination range is the same validated eight-element output window; the unique `&mut [f32]` output borrow must not alias the immutable inputs. |

Runtime feature detection itself (`is_x86_feature_detected!("avx2")`) is safe Rust and is not one of the five raw-`unsafe` hits. Scalar tail handling is also safe Rust.

The prototype used unaligned loads/stores, so it did **not** require stronger-than-`f32` alignment. Its remaining memory-safety obligations were bounds, pointer provenance, output uniqueness/non-aliasing and the AVX2 target-feature precondition.

## Requirements for any future exception

A proposal to admit an Auralis-owned `unsafe` SIMD boundary must satisfy all of the following before the security contract changes:

1. **Concrete scope** — exact file, functions, target architecture and purpose are named. No wildcard allowlist.
2. **Runtime ISA gate** — target-specific code is unreachable unless runtime feature detection confirms the required ISA.
3. **Safe reference path** — a portable safe-Rust implementation remains selectable and acts as the numerical oracle.
4. **Memory invariants** — every unsafe operation documents slice bounds, pointer provenance, alignment assumptions and aliasing rules.
5. **Fail-closed scanner** — CI identifies the exact approved sites. Any additional raw `unsafe` fails automatically.
6. **Independent review** — the implementation cannot be certified only by its authoring agent.
7. **Numerical equivalence** — reference/SIMD tests use the declared exact/tolerance contract and cover irregular/tail shapes.
8. **Performance evidence** — integration requires material, repeatable gains on relevant workloads; a microbenchmark alone is insufficient.
9. **Execution metadata** — benchmark records identify architecture, detected ISA features, toolchain and the selected backend so results are attributable.
10. **Dynamic memory checks where applicable** — run Miri over the safe dispatch/reference surface and sanitizer builds on supported targets when they can exercise the boundary. If Miri or a sanitizer cannot execute a target-feature intrinsic path, record that limitation explicitly rather than treating the tool as a passed check.
11. **Portable execution** — no supported Auralis build or model requires the optional ISA to run.
12. **Removal path** — the exception is removed if an equivalent safe implementation becomes available without material regression.

## Preferred implementation order

Before requesting an unsafe exception, investigate in this order:

1. safe Rust that the compiler can auto-vectorize;
2. stable safe SIMD abstractions whose implementation and dependency risk can be audited;
3. architecture intrinsics behind a narrowly audited raw-unsafe boundary.

Using a dependency merely to hide unaudited unsafe code does not satisfy this policy. Dependency safety and maintenance become part of the review.

## CI policy

Until a concrete exception is separately approved:

- `EXPECTED_UNSAFE_BLOCKS` remains `0`;
- `sec::tests::crate_src_matches_expected_contract` must remain green;
- no broad ignore patterns are added to the scanner;
- profiling binaries keep their existing explicit exclusion because they are outside train/eval/chat library paths;
- benchmark or correctness success cannot override the security gate.

If an exception is ever approved, CI must move from a count-only expectation to an exact allowlist (file/function/purpose) before the optimized code is merged. Merely increasing the expected count is not sufficient.

## Consequences

### Positive

- optimization work cannot silently expand memory-unsafety risk;
- security regressions remain visible in normal CI;
- portable/reference execution remains first-class;
- useful SIMD performance evidence can be retained without forcing premature integration.

### Cost

- some architecture-specific speedups may remain experimental longer;
- safe abstractions may add dependency or abstraction-review work;
- high-performance kernels require an additional security gate.

## Current disposition of AVX2 prototype

PR #199 is evidence, not integration. #109 remains blocked by this security boundary. Work may continue by finding a safe SIMD implementation or by preparing a concrete audited-exception proposal that satisfies this ADR and #146.
