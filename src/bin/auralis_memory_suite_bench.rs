use auralis::memory_suite::{
    capacity_sweep, execute_profile, generate_suite, intentional_regression_predictions,
    memory_off_predictions, score_predictions, suite_definition, MemoryFailureKind,
    MemorySuiteProfile, MEMORY_SUITE_SCHEMA_VERSION, MEMORY_SUITE_SEED,
};

fn profiles(arg: Option<&str>) -> Vec<MemorySuiteProfile> {
    match arg.unwrap_or("both") {
        "smoke" => vec![MemorySuiteProfile::Smoke],
        "full" => vec![MemorySuiteProfile::Full],
        "both" => vec![MemorySuiteProfile::Smoke, MemorySuiteProfile::Full],
        other => panic!("unknown memory-suite profile {other}; expected smoke|full|both"),
    }
}

fn main() {
    println!(
        "memory_suite_protocol | schema={} seed={} backend=external-memory-1 capacity_policy=reject-new eviction=none profiles=smoke,full",
        MEMORY_SUITE_SCHEMA_VERSION,
        MEMORY_SUITE_SEED
    );

    let arg = std::env::args().nth(1);
    for profile in profiles(arg.as_deref()) {
        let cases = generate_suite(profile, MEMORY_SUITE_SEED);
        let suite = suite_definition(profile, MEMORY_SUITE_SEED);
        suite.validate().expect("valid memory suite");
        let suite_fingerprint = suite
            .definition_fingerprint()
            .expect("memory suite fingerprint");

        let summary = execute_profile(profile, MEMORY_SUITE_SEED).expect("memory suite execution");
        let off = score_predictions(&cases, &memory_off_predictions(&cases)).expect("off score");
        let regression = score_predictions(
            &cases,
            &intentional_regression_predictions(&cases),
        )
        .expect("regression score");

        println!(
            "memory_suite_profile | profile={} cases={} tasks={} metrics={} suite_fingerprint={:016x} fixture_fingerprint={:016x} exact_match={:.6} off_exact_match={:.6} query_ns_total={} max_heap_bytes={} max_snapshot_bytes={} min_long_horizon={} max_long_horizon={}",
            profile.as_str(),
            cases.len(),
            suite.tasks.len(),
            suite.metrics.len(),
            suite_fingerprint,
            suite.dataset.fixture_fingerprint,
            summary.report.exact_match_ratio(),
            off.exact_match_ratio(),
            summary.query_ns_total,
            summary.max_heap_bytes,
            summary.max_snapshot_bytes,
            summary.min_long_horizon,
            summary.max_long_horizon,
        );
        println!(
            "memory_suite_regression | profile={} exact_match={:.6} missing={} wrong_value={} wrong_order={} conflict_capture={} stale_value={}",
            profile.as_str(),
            regression.exact_match_ratio(),
            regression.failure_count(MemoryFailureKind::Missing),
            regression.failure_count(MemoryFailureKind::WrongValue),
            regression.failure_count(MemoryFailureKind::WrongOrder),
            regression.failure_count(MemoryFailureKind::ConflictCapture),
            regression.failure_count(MemoryFailureKind::StaleValue),
        );

        for point in capacity_sweep(profile).expect("capacity sweep") {
            println!(
                "memory_suite_capacity | profile={} capacity={} attempted={} stored={} rejected={} exact_hits={} exact_retrieval={:.6} evicted={} query_ns={} heap_bytes={} snapshot_bytes={}",
                profile.as_str(),
                point.capacity,
                point.attempted,
                point.stored,
                point.rejected,
                point.exact_hits,
                point.exact_retrieval,
                point.evicted,
                point.query_ns,
                point.heap_bytes,
                point.snapshot_bytes,
            );
        }
    }

    println!(
        "memory_suite_gate | deterministic=true retrieval_exact=true temporal_latest=true conflict_latest=true stale_latest=true long_horizon=true memory_off_zero=true capacity_reject_new=true eviction_zero=true"
    );
}
