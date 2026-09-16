# Auralis Foundation benchmark

Foundation 0.2 treats reproducibility and regressions as part of the training contract.

Run the benchmark after building the release binary:

```bash
cargo build --release
bash scripts/foundation_benchmark.sh
```

The default workload uses a fixed seed (`659918`), batch size `2`, gradient accumulation `2`, 20 optimizer steps and 3 independent repetitions. Every repetition starts from the same initialization and corpus split.

## Hard determinism checks

All repetitions must produce exactly the same:

- parameter fingerprint;
- validation loss;
- test loss;
- checkpoint byte size.

A mismatch is a reproducibility regression and fails immediately.

## Regression signals

The benchmark also records:

- median training tokens/second across repetitions;
- maximum resident set size (RSS) observed by `/usr/bin/time`;
- validation and test cross-entropy;
- checkpoint size.

Thresholds live in `benchmarks/foundation-baseline.conf`.

The Foundation gates are deliberately broad because GitHub-hosted runner performance is noisy. Their purpose is to catch material regressions, not to claim microbenchmark precision:

- validation loss must be at most `4.60`;
- test loss must be at most `4.60`;
- median throughput must be at least `3000 tok/s`;
- peak RSS must stay at or below `262144 KiB` (256 MiB);
- checkpoint size must stay at or below `400000` bytes.

Engine 0.3 can add tighter hardware-calibrated relative baselines once optimized kernels exist.

## Engine matmul microbenchmarks

Engine E2 keeps auditable scalar matrix-multiplication kernels as numerical oracles. The current contract, layout, representative shapes and exact-vs-tolerant equivalence rules are documented in [`engine-matmul-contract.md`](engine-matmul-contract.md).

Run the reference/candidate harnesses with release optimizations:

```bash
cargo run --release --bin auralis_matmul_bench -- 5 40 5
cargo run --release --bin auralis_matmul_bt_bench -- 5 40 5
```

The arguments are warmup iterations, measured iterations and repetitions. These harnesses validate correctness before timing, keep setup outside the timed kernel loop, alternate run order and report medians. They are intended for side-by-side experiments; they do not by themselves justify promoting a candidate kernel.

## Overrides

The Foundation harness accepts environment overrides for controlled experiments:

- `AURALIS_BENCH_STEPS`
- `AURALIS_BENCH_REPEATS`
- `AURALIS_BENCH_SEED`
- `AURALIS_BENCH_BATCH`
- `AURALIS_BENCH_ACCUM`
- `AURALIS_BENCH_BASELINE`
- `AURALIS_BENCH_BIN`

Changing those values creates a different benchmark experiment; do not compare its numbers directly to the committed Foundation baseline without documenting the change.
