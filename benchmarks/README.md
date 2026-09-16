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

## Overrides

The harness accepts environment overrides for controlled experiments:

- `AURALIS_BENCH_STEPS`
- `AURALIS_BENCH_REPEATS`
- `AURALIS_BENCH_SEED`
- `AURALIS_BENCH_BATCH`
- `AURALIS_BENCH_ACCUM`
- `AURALIS_BENCH_BASELINE`
- `AURALIS_BENCH_BIN`

Changing those values creates a different benchmark experiment; do not compare its numbers directly to the committed Foundation baseline without documenting the change.
