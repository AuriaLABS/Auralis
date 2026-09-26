# Multimodal collator contract (#102)

`MM_COLLATOR_SCHEMA_VERSION = 1`

Deterministic padding/masks over #60 sequences.

- same examples produce the same batch;
- text-only keeps original tokens and only pads shorter rows;
- mixed text/image/audio batches carry combined masks;
- padding waste is counted;
- OOM preflight rejects batches above `MAX_BATCH_TOKENS`.

Does **not** change unimodal encoder semantics.
