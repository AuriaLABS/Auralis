# Video boundary contract (#143)

`VIDEO_SCHEMA_VERSION = 1`

Video as sampled #62 frames on the #60 contract.

- frame sampling and temporal order are deterministic;
- max frames fail closed;
- the visual encoder is reused;
- video-off does not change text or still-image paths.

Does **not** add a heavy video architecture.
