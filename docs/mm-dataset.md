# Multimodal dataset contract (#61)

`MM_DATASET_SCHEMA_VERSION = 1`

Versioned manifests for text-only or paired multimodal examples.

- fingerprints are stable for the same manifest;
- train/validation/test splits are deterministic;
- missing or corrupt assets fail closed;
- transforms are identified and versioned.

Does **not** download large datasets or apply unseeded augmentations.
