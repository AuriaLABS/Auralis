# Per-layer Brain diagnostics

Brain layer instrumentation is opt-in and does not alter the supported math path.

## Hooks

The internal LayerHooks contract has three useful modes:

- off — direct historical backward path, no report;
- compact — per-layer summaries without histograms/cosine;
- full — summaries + compact histogram + adjacent gradient cosine.

When hooks are off, Auralis calls the existing backward implementation directly.

## Activation summaries

For every transformer layer, the full/compact report covers:

- h1;
- q;
- k;
- v;
- causal attention probabilities;
- attention output;
- h2;
- FF pre-activation;
- FF activation.

Each summary records:

- NaN/Inf scan;
- mean;
- L2 norm;
- max absolute value;
- optional compact histogram.

The compact histogram uses five finite buckets:
less than -1, [-1,-1e-6), near zero, (1e-6,1], greater than 1,
plus a separate non-finite count.

## Gradient summaries

After the normal backward completes, the flat gradient buffer is sliced using the same block parameter layout as collect_params.

Each layer gets one aggregate mean/L2/max/histogram summary.

Full mode also reports cosine similarity between adjacent layer gradient vectors when both norms are nonzero and finite.

## Machine-readable output

LayerDiagnosticsReport::json emits activations, gradients and adjacent cosine values as one JSON object.

The benchmark binary prints it with:

\`\`\`text
layer_diagnostics_json | {...}
\`\`\`

## Cost

\`auralis_layer_diagnostics_bench\` compares:

1. historical backward;
2. hooks OFF wrapper;
3. hooks FULL.

Loss and gradients are checked bit-for-bit before timing.

Only OFF is a hot-path regression gate. FULL is debug instrumentation and its overhead is measured/reported rather than hidden.
