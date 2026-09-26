# Multimodal eval contract (#65)

`MM_EVAL_SCHEMA_VERSION = 1`

Separate unimodal and cross-modal scores.

- text, vision, audio and cross-modal tracks are reported apart;
- a modality regression is visible even if other tracks improve;
- reports record commit and config;
- descriptive mean is not a pass/fail gate.

Does **not** use a single aggregate score as the suite criterion.
