# Cross-modal alignment contract (#117)

`MM_ALIGN_SCHEMA_VERSION = 1`

Bidirectional retrieval over #60 sequences.

- text↔image and text↔audio use the same similarity;
- recall@k is per query, not a single aggregate;
- collapse is detected when all embeddings share one value;
- the unaligned baseline is index order.

Does **not** replace unimodal baselines with a fused score.
