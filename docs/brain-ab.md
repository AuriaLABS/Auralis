# Brain A/B experiment harness

`auralis_brain_ab` is the first reproducible architecture comparison harness for Brain 0.4.

It is deliberately a **measurement tool**, not an automatic architecture selector.

## Protocol

Both variants share:

- seed;
- synthetic token stream and token fingerprint;
- optimizer;
- learning rate;
- gradient clipping;
- batch size;
- gradient accumulation;
- optimizer steps;
- repetition count.

Order alternates A-first/B-first between repetitions.

Each measurement records:

- complete model config;
- final training loss;
- eval loss and perplexity;
- tokens/s;
- parameter count and parameter bytes;
- Adam state bytes;
- real AURLIS03 checkpoint bytes;
- final parameter fingerprint.

The experiment record also contains the build commit/revision.

## Reproducibility control

```bash
cargo run --release --bin auralis_brain_ab -- 4 3 same
```

The `same` scenario uses identical A/B configs. Deterministic state, losses, checkpoint size and fingerprints must match across A and B and across repetitions. Wall-clock throughput is not required to be identical.

## Example depth comparison

```bash
cargo run --release --bin auralis_brain_ab -- 8 5 depth
cargo run --release --bin auralis_brain_ab -- 8 5 depth --json
```

The `depth` scenario compares 1-layer vs 2-layer tiny models under the same training budget.

The harness reports both. It does **not** declare a winner from a noisy hosted-runner timing or from a single metric.

Future architecture issues can reuse the library API with other `Config` values or connect it to versioned architecture files after #33.
