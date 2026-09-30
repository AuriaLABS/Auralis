# Packaging contract (#303)

`PACKAGING_SCHEMA_VERSION = 1`

What a v1 source/binary drop must contain before it can be published.

- LICENSE is MIT and ships with the artifact;
- checksums and the release manifest travel with the payload;
- missing license or checksum blocks publish;
- secrets in artifact metadata fail closed;
- smoke uses release artifacts, not the dev checkout.

Does **not** cut `v1.0.0` or claim bit-identical binaries across OS/arch.
