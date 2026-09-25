# Vision encoder contract (#62)

`VISION_SCHEMA_VERSION = 1`

Reference visual encoder over the #60 common sequence.

- same image/config yields the same patch tokens;
- invalid resolution or corrupt pixels fail closed;
- the encoder can be disabled without changing the text path;
- a tiny fixture covers forward and backward.

Does **not** generate images or require a pretrained vision model.
