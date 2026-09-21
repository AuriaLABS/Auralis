# Experimental model-memory integration

Issue #42 connects the #37 external-memory contract to model inference through an **opt-in** adapter. It does not make memory part of the default transformer path.

## Invariant: memory off is the baseline

The public memory API is \`Gpt::logits_with_memory\`.

With \`MemoryInferenceMode::Off\`:

- no memory query is executed;
- no external-memory object is required;
- the function delegates directly to \`Gpt::logits\`;
- logits are bit-for-bit the baseline.

If memory is enabled but the query returns no records, the model also returns the normal baseline logits.

## Retrieval is separate from fusion

\`memory_integration.rs\` performs retrieval first.

The trace records:

- schema version;
- whether memory was enabled/present;
- external-memory schema version;
- capacity and current record count;
- query latency;
- hit count;
- record IDs;
- fused width;
- fusion policy and scale.

The model never writes to memory. It receives \`&dyn ExternalMemory\`, not a mutable backend.

## Fusion policy

The first experimental policy is \`last_hidden_mean_add\`.

1. Query #37 memory.
2. Require each retrieved value to have exactly \`n_embd\` floats.
3. Compute their deterministic arithmetic mean in query order.
4. Add \`scale * mean\` to the **last row of h_final**.
5. Run the existing output projection and bias.

All earlier token rows remain unchanged by the fusion operation itself.

This location is deliberate: retrieval is not hidden inside attention, normalization or tokenization, and the memory value has a clear hidden-space contract.

## Empty and malformed memory

- missing backend: defined as zero hits;
- empty/no-hit query: baseline logits;
- wrong memory-vector width: explicit error;
- invalid query: propagated as an explicit error;
- no implicit writes or resets occur.

Persistence/versioning remain exactly the #37 contract.

## Synthetic objective task

\`auralis_memory_model_bench\` constructs a deterministic controlled tiny model:

- vocab=4;
- embedding width=4;
- all parameters zero except an identity-like output projection;
- baseline final hidden state is zero;
- baseline therefore predicts token 0 by stable argmax.

Memory contains three exact-key records:

- \`target-1 -> [0,1,0,0]\`
- \`target-2 -> [0,0,1,0]\`
- \`target-3 -> [0,0,0,1]\`

For the three target tasks, memory-off should score \`0/3\`, while memory-on should score \`3/3\`. This proves that retrieval and fusion can causally alter a model output on an objective synthetic task; it is **not** evidence of improved general reasoning.

Run:

\`\`\`bash
cargo run --release --bin auralis_memory_model_bench -- 100 7
\`\`\`

The benchmark additionally reports:

- on/off latency ratio;
- query latency in the trace;
- backend record count;
- estimated heap bytes;
- snapshot bytes/fingerprint;
- proof that the memory snapshot is unchanged by inference.

Timing is descriptive evidence, not a promotion gate.

## Promotion policy

External memory remains experimental.

No default changes until a broader A/B experiment demonstrates value on predeclared tasks while accounting for latency, memory footprint and failure/contamination behavior.
