# Benchmark registry

`auralis bench list` catalogs Engine microbenchmarks. This slice does not run them.

```
auralis bench list
auralis bench describe engine
```

To add a benchmark:

1. Keep the existing `src/bin/auralis_*_bench.rs` (or profile bin).
2. Append a `BenchSpec` in `src/bench.rs`.
3. Document default args used by CI in `genesis.yml`.
4. Do not change kernel math in the same PR as the registry.

Running a listed bench stays:

```
cargo run --release --bin auralis_engine_bench -- 3 20 5
```
