# Engine tensor layout audit

This document records the current CPU tensor layout used by Auralis Engine and the evidence protocol for deciding whether a physical layout migration is justified.

## Canonical layout today

Auralis uses flat Vec<f32> storage with row-major logical tensors.

| Tensor / parameter | Logical shape | Flat index / layout |
|---|---:|---|
| token/hidden activations | [tokens, d] | token * d + feature |
| Q / K / V | [tokens, d] | contiguous feature row per token |
| attention output | [tokens, d] | contiguous feature row per token |
| attention probabilities | [head, query, key] | (head * t + query) * t + key |
| Wq/Wk/Wv/Wo | [d, d] | row-major input * d + output |
| FFN W1 | [d, n_ff] | row-major input * n_ff + output |
| FFN W2 | [n_ff, d] | row-major |
| output projection | [d, vocab] | row-major |
| flat parameters / gradients | serialized model order | each tensor keeps its native row-major order |
| Adam m / v | flat parameter order | same indexing as flat parameters |

The current forward matmul consumes A [rows, inner], B [inner, cols], and O [rows, cols], all row-major.

The promoted matmul_row_slices_into traverses row i, then inner k, then contiguous B[k, :] and contiguous O[i, :]. This is the current optimized CPU representation.

## Strides and indexing

The Engine does not maintain a general strided tensor object yet. Hot-path tensors are flat contiguous arrays and kernels receive dimensions explicitly.

Consequences:

- token/feature rows are contiguous;
- parameter matrix rows are contiguous;
- output rows are contiguous;
- attention head slices inside Q/K/V are contiguous subranges of each token row;
- attention probabilities are head-major, then query-major, then key-major.

The absence of a general stride abstraction is deliberate at this stage: introducing one is only justified if a measured layout requirement needs it.

## Physical transpose inventory

The current promoted forward path does not physically transpose Wq/Wk/Wv/Wo/W1/W2/Wout.

Backward dY * B^T kernels also do not materialize B^T: they access the original row-major B logically through indexing/slices.

Therefore the current Engine has no general hot-path transpose-B-then-matmul allocation/copy to remove.

This is an important negative finding: a layout project must not claim savings from removing a physical transpose that is not currently present.

## Existing view/slice optimizations

Promoted CPU kernels already use borrowed contiguous slices where that preserves arithmetic order:

- forward RowSlices matmul;
- B^T backward RowSlices;
- additive B^T backward RowSlices;
- attention forward RowSlices;
- attention backward RowSlices;
- reusable workspaces/arena for scratch.

These are views into existing buffers, not copied tensor views.

## Candidate audited: persistent packed B^T

The audit benchmark compares three representations for representative Engine shapes:

1. row_slices — current production layout/kernel;
2. packed_bt_persistent — B is transposed once, then each output cell reads a contiguous B^T row;
3. packed_bt_dynamic — B is physically transposed for every measured call before the packed kernel.

Representative shapes:

- QKV/projection: 32x32x32;
- FFN up: 32x32x96;
- FFN down: 32x96x32;
- logits: 32x32x100;
- short-context QKV: 9x32x32.

The packed kernel preserves the per-output k accumulation order, so bit-for-bit equality is required.

## Why both packed modes are measured

A persistent packed layout could be legitimate for weights because model weights survive many forward calls.

A dynamic transpose is a different proposal: it creates an extra copy on the hot path. Measuring both prevents an unfair comparison that hides packing cost.

If persistent packing wins but dynamic packing loses, the correct conclusion is not transpose-at-runtime. It would justify a separate experiment about persistent duplicated/alternative weight storage and its checkpoint/backward/memory cost.

## Locality proxy

Hosted CI does not expose reliable hardware cache counters. This audit therefore reports:

- raw nanoseconds;
- median/IQR through BenchRunRecord;
- candidate/reference ratio;
- RowSlices nanoseconds per MAC as a coarse locality/shape proxy.

This is not a cache-miss measurement and must not be described as one.

## Promotion rule

A layout change is not promoted from one favorable run.

A persistent packed candidate only becomes eligible for a new implementation slice when:

- exact output holds;
- median packed-kernel ratio is at most 0.95 vs RowSlices on a relevant shape;
- the win is repeatable;
- memory/checkpoint/backward implications are explicitly costed;
- no dynamic transpose is silently introduced into the production hot path.

Otherwise RowSlices remains canonical.

## Migration cost if persistent packed weights are pursued

A production packed layout would affect more than one kernel:

- parameter ownership;
- parameter flatten/writeback order or a secondary packed cache;
- invalidation after Adam updates;
- checkpoint semantics if packed representation were persisted;
- memory footprint if canonical and packed copies coexist;
- backward access patterns;
- diagnostics/inspect tooling.

Because Adam updates weights every training step, a pack-once assumption is valid for inference but not automatically valid for training.

## Reproducible command

    cargo run --release --bin auralis_layout_audit -- 5 100 7

The Engine Layout Audit workflow validates the records and publishes per-shape decision data.

## Measured result — 2026-09-20

The same benchmark content was executed twice on hosted runners before final integration. Absolute timings moved materially between runs, so the decision uses the repeated ratio/direction rather than treating one hosted timing as universal.

### Run A

Persistent packed-B^T / RowSlices median ratios:

- QKV 32x32x32: 3.9516x
- FFN up 32x32x96: 3.9085x
- FFN down 32x96x32: 5.2786x
- logits 32x32x100: 4.1919x
- short QKV 9x32x32: 4.1306x

Dynamic packing was slower again: 4.1227x, 3.9530x, 5.4155x, 4.3647x, and 4.3901x.

### Run B

Persistent packed-B^T / RowSlices median ratios:

- QKV 32x32x32: 2.1679x
- FFN up 32x32x96: 3.0538x
- FFN down 32x96x32: 2.3157x
- logits 32x32x100: 2.8584x
- short QKV 9x32x32: 2.1612x

Dynamic packing was again slower: 2.2578x, 3.1444x, 2.4037x, 2.9288x, and 2.4276x.

All outputs were bit-for-bit exact and promotion_candidate=false for every case in both runs.

The magnitude is runner-sensitive, but the conclusion is not marginal: even the favorable-to-packing protocol that excludes transpose cost loses by more than 2x in the better of the two runs for every measured shape.

The current i-k-j RowSlices traversal benefits from contiguous B rows and contiguous output rows. Converting B to a persistent B^T representation would trade that pattern for per-output dot products and is decisively worse for the measured Auralis shapes.

## Current decision

**RowSlices remains canonical. No tensor-layout migration is justified by this audit.**

Specifically:

- do not add a dynamic physical transpose to the hot path;
- do not add a persistent packed-B^T cache for these shapes;
- do not change checkpoint/parameter layout;
- do not duplicate weights solely for this candidate;
- keep the benchmark as regression/research evidence.

A future layout proposal must introduce a materially different hypothesis or target workload; repeating this packed-B^T candidate without new evidence is not useful.
