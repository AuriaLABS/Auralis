# Auralis

Motor cognitivo mínimo en Rust (AuriaLABS): transformer decoder-only, backprop y Adam a mano.

https://github.com/AuriaLABS/Auralis

```bash
cargo test
cargo run --release -- check
cargo run --release -- train 80 auralis.bin
cargo run --release -- eval auralis.bin
cargo run --release -- chat auralis.bin
```

`train` reanuda checkpoint AURLIS03 (pesos + Adam).
`train-fresh` parte de cero.
Hoja de ruta de 12 meses: `ROADMAP.md`.
