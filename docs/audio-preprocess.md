# Audio preprocess contract (#116)

`AUDIO_PREPROCESS_SCHEMA_VERSION = 1`

Versioned audio preprocess before the #63 encoder.

- allowed format is `pcm16` mono;
- sample-rate and duration limits fail closed;
- amplitude normalize is deterministic;
- corrupt or unsupported input never reaches the encoder;
- fingerprints change when the preprocess config changes.

Does **not** decode compressed containers or stream in realtime.
