# Audio encoder contract (#63)

`AUDIO_SCHEMA_VERSION = 1`

Reference audio encoder over the #60 common sequence.

- same audio/config yields the same framed tokens;
- partial frames are masked, full frames are not;
- invalid rate/frame/empty input fail closed;
- the encoder can be disabled without changing text or vision.

Does **not** do TTS, full ASR or realtime streaming.
