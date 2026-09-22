use auralis::moe::{
    DeterministicTopKRouter, MoeRouter, MoeRouterConfig, OverflowPolicy,
    MOE_ROUTER_SCHEMA_VERSION,
};
use std::hint::black_box;
use std::time::Instant;

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn router_logits(tokens: usize, experts: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(tokens * experts);
    for token in 0..tokens {
        for expert in 0..experts {
            let raw = ((token * 31 + expert * 17 + token * expert * 3 + 11) % 101) as f32;
            out.push((raw - 50.0) / 13.0);
        }
    }
    out
}

fn activations(tokens: usize, width: usize) -> Vec<f32> {
    (0..tokens * width)
        .map(|i| (((i * 19 + i / 7 + 5) % 97) as f32 - 48.0) / 31.0)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn measure_case(tokens: usize, experts: usize, top_k: usize, width: usize, repeats: usize) {
    let router = DeterministicTopKRouter;
    let logits = router_logits(tokens, experts);
    let input = activations(tokens, width);
    let config = MoeRouterConfig {
        experts,
        top_k,
        capacity: tokens * top_k,
        overflow_policy: OverflowPolicy::Reject,
    };

    let reference_plan = router.route(&logits, tokens, config).unwrap();
    let reference_dispatch = router.dispatch(&reference_plan, &input, width).unwrap();
    let reference_experts = reference_dispatch
        .experts
        .iter()
        .map(|expert| expert.activations.clone())
        .collect::<Vec<_>>();
    let reference_output = router
        .gather(&reference_plan, &reference_experts, None, width)
        .unwrap();
    let diff = max_abs_diff(&reference_output, &input);
    assert!(diff <= 1e-5, "identity expert gather drift: {diff}");

    let mut dense_times = Vec::with_capacity(repeats);
    let mut route_times = Vec::with_capacity(repeats);
    let mut dispatch_gather_times = Vec::with_capacity(repeats);

    for _ in 0..repeats {
        let started = Instant::now();
        black_box(input.clone());
        dense_times.push(started.elapsed().as_nanos().max(1) as u64);

        let started = Instant::now();
        let plan = black_box(router.route(&logits, tokens, config).unwrap());
        route_times.push(started.elapsed().as_nanos().max(1) as u64);

        let started = Instant::now();
        let dispatched = router.dispatch(&plan, &input, width).unwrap();
        let expert_outputs = dispatched
            .experts
            .iter()
            .map(|expert| expert.activations.clone())
            .collect::<Vec<_>>();
        let gathered = router
            .gather(&plan, &expert_outputs, None, width)
            .unwrap();
        black_box(gathered);
        dispatch_gather_times.push(started.elapsed().as_nanos().max(1) as u64);
    }

    let metrics = reference_plan.metrics();
    let dense_ns = median(&mut dense_times);
    let route_ns = median(&mut route_times);
    let dispatch_gather_ns = median(&mut dispatch_gather_times);
    println!(
        "moe_route | tokens={} experts={} top_k={} capacity={} width={} dense_reference_ns={} route_ns={} dispatch_gather_ns={} total_overhead_vs_dense={:.4} assignments={} fallback_tokens={} dropped_tokens={} empty_experts={} min_load={} max_load={} load_cv={:.6} capacity_utilization={:.6} identity_max_abs_diff={:.8}",
        tokens,
        experts,
        top_k,
        config.capacity,
        width,
        dense_ns,
        route_ns,
        dispatch_gather_ns,
        (route_ns + dispatch_gather_ns) as f64 / dense_ns as f64,
        metrics.assignments,
        metrics.fallback_tokens,
        metrics.dropped_tokens,
        metrics.empty_experts,
        metrics.min_load,
        metrics.max_load,
        metrics.load_cv,
        metrics.capacity_utilization,
        diff,
    );
}

fn saturation_case() {
    let router = DeterministicTopKRouter;
    let tokens = 12usize;
    let experts = 4usize;
    let top_k = 1usize;
    let capacity = 2usize;
    let width = 8usize;
    let mut logits = vec![0.0f32; tokens * experts];
    for token in 0..tokens {
        logits[token * experts] = 10.0;
        logits[token * experts + 1] = 1.0;
        logits[token * experts + 2] = 0.0;
        logits[token * experts + 3] = -1.0;
    }
    let input = activations(tokens, width);
    let config = MoeRouterConfig {
        experts,
        top_k,
        capacity,
        overflow_policy: OverflowPolicy::DenseFallback,
    };
    let plan = router.route(&logits, tokens, config).unwrap();
    let dispatch = router.dispatch(&plan, &input, width).unwrap();
    let expert_outputs = dispatch
        .experts
        .iter()
        .map(|expert| expert.activations.clone())
        .collect::<Vec<_>>();
    let output = router
        .gather(&plan, &expert_outputs, Some(&input), width)
        .unwrap();
    let metrics = plan.metrics();
    let diff = max_abs_diff(&output, &input);
    assert!(diff <= 1e-6);
    assert_eq!(metrics.dropped_tokens, 0);
    assert_eq!(metrics.fallback_tokens, tokens - capacity);
    println!(
        "moe_saturation | tokens={} experts={} top_k={} capacity={} overflow_policy={} routed_tokens={} fallback_tokens={} dropped_tokens={} max_load={} identity_max_abs_diff={:.8}",
        tokens,
        experts,
        top_k,
        capacity,
        config.overflow_policy.as_str(),
        metrics.routed_tokens,
        metrics.fallback_tokens,
        metrics.dropped_tokens,
        metrics.max_load,
        diff,
    );
}

fn main() {
    println!(
        "moe_router_protocol | schema={} router=deterministic-top-k tie_break=lower-expert-index overflow=reject,dense-fallback repeats=7 dense_default=true",
        MOE_ROUTER_SCHEMA_VERSION,
    );

    for (tokens, experts, top_k) in [
        (32usize, 4usize, 1usize),
        (128usize, 8usize, 2usize),
        (512usize, 16usize, 2usize),
    ] {
        measure_case(tokens, experts, top_k, 32, 7);
    }
    saturation_case();
}
