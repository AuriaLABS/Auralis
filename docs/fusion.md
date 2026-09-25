# Fusion contract (#64)

`FUSION_SCHEMA_VERSION = 1`

Reference concat fusion over #60 common sequences.

- text-only is identity with the textual baseline;
- a required missing modality fails closed;
- cross-modal masks have positive and negative tests;
- fusion budget units equal the concatenated token count.

Does **not** declare a final multimodal architecture.
