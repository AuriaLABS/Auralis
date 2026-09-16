# Engine E2 matmul contract

This document closes the reference-contract portion of Engine E2 before optimized kernels are promoted.

## Scope

The current CPU reference paths cover the three matrix products used by the model hot path:

1. forward projection: `A[rows, inner] * B[inner, cols] -> Y[rows, cols]`;
2. input gradient: `dY[rows, out_cols] * B[result_cols, out_cols]^T -> dX[rows, result_cols]`;
3. weight gradient: `A[rows, inner]^T * dY[rows, cols] -> dB[inner, cols]`.

All matrices are contiguous row-major `f32` slices. Auralis does not currently expose arbitrary strides in these kernels. A future stride/view abstraction must therefore be introduced as a separate contract change rather than silently changing this one.

## Reference API

The auditable scalar oracles live in `src/kernels.rs`:

- `matmul_reference_into`;
- `matmul_b_t_reference_into`;
- `matmul_b_t_reference_add_into`;
- `matmul_grad_b_reference`.

Candidate kernels currently used for exact-equivalence experiments include the row-slice variants in the same module. Reference implementations remain available even when a candidate is faster.

## Arithmetic order and numerical contract

For the existing row-slice candidates, every individual output cell accumulates products in the same scalar order as the reference implementation. The required equivalence is therefore **bit-for-bit equality** for finite deterministic inputs.

Future optimized kernels may use blocking, vectorization, parallel reduction or another accumulation order. Those candidates must declare the numerical contract before integration. Unless a stricter operation-specific bound is justified, comparisons should report both:

- maximum absolute error;
- maximum relative error using `max(|reference|, 1e-8)` as denominator floor.

A non-exact candidate must not reuse the label `exact_state=true`; its end-to-end loss, gradients and optimizer-state deviation must be verified separately by the Engine equivalence suite.

## Shape validation

Each kernel asserts the exact contiguous lengths implied by its logical dimensions. Invalid shapes are programmer errors at this internal boundary and fail before arithmetic begins.

Representative tiny-model shapes currently exercised by tests and the forward microbenchmark include:

| Operation family | rows | inner / out cols | cols / result cols | Typical role |
| --- | ---: | ---: | ---: | --- |
| forward | 32 | 32 | 32 | attention/output projections |
| forward | 32 | 32 | 96 | FFN expansion |
| forward | 32 | 96 | 32 | FFN contraction |
| forward | 32 | 32 | 100 | logits projection |
| backward `dY * B^T` | 32 | 100 | 32 | logits/input gradient |
| backward `dY * B^T` | 32 | 32 | 96 | FFN/linear input gradient |
| backward `A^T * dY` | 32 | 32 | 96 | FFN/linear weight gradient |

Small irregular cases such as `1x1x1`, `4x3x5` and `7x8x3` are also tested so correctness does not depend on powers of two or model-default dimensions.

## Layout

The model uses contiguous row-major storage:

- element `(r, c)` of `[rows, cols]` is at `r * cols + c`;
- right-hand matrices are stored in their logical row-major orientation;
- `B^T` products do not materialize a transpose: the kernel indexes rows of `B` directly;
- gradient accumulation variants explicitly distinguish overwrite vs additive output semantics.

This layout favors contiguous traversal across the final dimension. The current `i -> k -> j` forward reference reuses one `A[i,k]` value while walking a contiguous row of `B` and `Y`.

## Reproducible microbenchmarks

Build and run with release optimizations:

```bash
cargo test
cargo run --release --bin auralis_matmul_bench -- 5 40 5
cargo run --release --bin auralis_matmul_bt_bench -- 5 40 5
```

Arguments are `warmup_iterations`, `measured_iterations`, and `repeats`.

The forward harness:

- allocates and initializes inputs/outputs before timed regions;
- validates exact reference/candidate equality before timing;
- alternates which implementation runs first;
- performs multiple repetitions;
- reports medians rather than selecting the best run;
- uses a checksum through `black_box` so work is observable to the optimizer.

The benchmark is a comparison harness, not evidence by itself that a candidate should become the production default. Engine #30 requires a material and repeatable improvement in relevant workloads.

## Dominant work and next optimization target

For the current tiny model, dense projections are repeated across attention, FFN and logits. Their arithmetic cost scales as `O(rows * inner * cols)`. The most useful E2 work is therefore to improve locality and loop efficiency for the representative shapes above while preserving the scalar oracle.

The next candidate work belongs in #30: blocked/tiling and other portable CPU strategies, benchmarked side-by-side against these references. SIMD, threading, GPU and precision changes remain separate experiments.

## Acceptance / handoff for #26

- reference kernels remain simple and directly reviewable;
- exact row-slice candidates are covered by bit-for-bit tests;
- representative and irregular shapes are covered;
- setup is outside the timed region of the microbenchmark;
- layout and overwrite/additive semantics are explicit;
- current exact candidates require exact equality;
- any future reordered arithmetic must declare tolerances and cannot claim exact state implicitly.
