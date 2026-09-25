# Modality contract (#60)

`MODALITY_SCHEMA_VERSION = 1`

Common sequence frontier for future image/audio backends.

- `ModalityEncoder` emits tokens, mask and positions;
- the text backend is an identity over existing tokens;
- `modality=off` yields an empty sequence;
- mismatched shapes fail before the core;
- the manifesto records active adapters.

Does **not** implement visual or audio encoders or change the text baseline.
