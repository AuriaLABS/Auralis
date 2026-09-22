# Deterministic MoE router experiment

Issue #112 introduces the minimum auditable Mixture-of-Experts routing infrastructure. **MoE remains experimental** and the dense model path remains the reference/default.

This slice deliberately separates routing correctness from later model integration: it does not replace the dense FFN, add expert parameters to checkpoints, or claim a quality improvement.

## Versioned contract

The public routing contract is `MOE_ROUTER_SCHEMA_VERSION = 1`.

`MoeRouterConfig` declares:

- `experts`: number of experts;
- `top_k`: number of selected experts per routed token;
- `capacity`: maximum assignments accepted by each expert;
- `overflow_policy`: explicit handling when any selected expert is full.

Zero experts, zero capacity, `top_k=0`, `top_k>experts`, non-finite logits and incompatible tensor lengths fail closed.

## Deterministic top-k

`DeterministicTopKRouter` implements the public `MoeRouter` interface.

For each token:

1. router logits are ranked from highest to lowest;
2. exact score ties are broken by lower expert index;
3. the first `top_k` experts are selected;
4. selected logits are normalized with a stable softmax;
5. repeated calls with the same logits/config produce the same route plan.

The router consumes explicit logits. Learning router weights is intentionally outside this first infrastructure slice.

## Capacity and overflow

Capacity is checked transactionally per token before committing any assignment.

Two policies are explicit:

- `reject`: return `CapacityExceeded` immediately;
- `dense-fallback`: route the whole overflowing token through the dense fallback instead of partially routing it.

There is no silent token drop. `MoeMetrics::dropped_tokens` is therefore always zero for a successful plan, and the invariant is:

```text
tokens = routed_tokens + fallback_tokens
dropped_tokens = 0
```

A future token-dropping policy would require a new explicit contract; it cannot appear implicitly.

## Dispatch / gather

`dispatch` groups token activations by expert using deterministic per-expert slots. Each dispatch group contains:

- source token indices;
- normalized gate values;
- contiguous expert activations.

`gather` consumes expert outputs using the same `(expert, slot)` identity and combines top-k outputs with their gate weights. Tokens marked `dense-fallback` require a full dense reference output; missing fallback data fails closed.

The tests use identity experts to prove dispatch/gather reconstruction and fallback coverage without introducing expert-network math into the routing contract.

## Balance metrics

Every successful `RoutePlan` exposes:

- per-expert loads;
- routed/fallback token counts;
- assignment count;
- empty-expert count;
- min/max/mean load;
- load coefficient of variation;
- capacity utilization;
- dropped token count.

These metrics are descriptive. No auxiliary load-balancing loss is introduced by #112.

## Dense reference and benchmark

The dense model remains the reference/default. Run:

```bash
cargo run --release --bin auralis_moe_router_bench
```

The benchmark compares dense identity-copy overhead with deterministic routing plus dispatch/gather for several token/expert/top-k shapes. It also includes a forced saturation case using `dense-fallback` and verifies:

- deterministic routing;
- no silent token loss;
- capacity is respected;
- identity expert gather matches the input within floating-point tolerance;
- routing/dispatch overhead is published side-by-side with the dense reference.

Hosted-runner timings are descriptive evidence only. #112 does **not** claim that MoE improves model quality or end-to-end speed.

## Non-goals

This slice does not add:

- learned router weights;
- expert FFN parameterization in `Gpt`;
- auxiliary balance loss;
- expert parallelism or all-to-all communication;
- GPU dispatch kernels;
- expert checkpoint formats;
- MoE as a default architecture.

Those require separate integration/evaluation work with dense quality and compute budgets held comparable.
