# Attention boundary

Brain #36 isolates the current causal multi-head attention semantics behind an internal contract.

## Semantics

The contract owns:

- Q/K/V shape validation;
- causal masking: position i can attend only to j <= i;
- scaled dot-product score;
- softmax;
- weighted value output;
- cached probabilities needed by backward;
- backward gradients for Q/K/V.

Two implementations exist:

- ReferenceAttention — scalar/reference kernels;
- RowSlicesAttention — current optimized training kernels.

No sparse, linear, flash, local or other attention variant is introduced by #36.

## Eval versus training

Eval/inference preserves the historical O(tokens) score scratch and does not allocate the full heads*tokens*tokens probability cache.

Training forward materializes the probability cache because backward requires it.

This distinction is part of the boundary so a future experimental variant cannot silently add training-only cache costs to inference.

## Verification

Reference and RowSlices must be bit-for-bit exact for the current arithmetic order on representative shapes.

The tests also assert that future causal positions retain zero probability and that invalid head/shape contracts fail before entering the kernels.

The boundary benchmark compares the existing RowSlices kernel called directly versus through the contract using the same buffers. It measures abstraction/validation overhead only; it is not a benchmark of a new attention algorithm.
