# Image preprocess contract (#101)

`IMAGE_PREPROCESS_SCHEMA_VERSION = 1`

Versioned image preprocess before the #62 encoder.

- allowed format is `raw8`;
- size/byte limits fail closed;
- resize/normalize is deterministic;
- corrupt or unsupported input never reaches the encoder;
- fingerprints change when the preprocess config changes.

Does **not** decode arbitrary container formats or download assets.
