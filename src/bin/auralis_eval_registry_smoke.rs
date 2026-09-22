use auralis::eval_registry::{
    smoke_baseline, smoke_suite, EvaluationRegistry, SemVer, EVALUATION_REGISTRY_SCHEMA_VERSION,
};

fn main() {
    let mut registry = EvaluationRegistry::new();

    let v1 = smoke_suite();
    let v1_ref = registry.register_suite(v1).expect("register smoke suite v1");
    let baseline_v1 = smoke_baseline(v1_ref.clone(), SemVer::new(1, 0, 0));
    let baseline_v1_fingerprint = registry
        .register_baseline(baseline_v1)
        .expect("register smoke baseline v1");

    let mut v2 = smoke_suite();
    v2.version = SemVer::new(2, 0, 0);
    v2.description = "offline deterministic registry contract smoke suite v2".to_string();
    v2.dataset.revision = "2".to_string();
    let v2_ref = registry.register_suite(v2).expect("register smoke suite v2");
    registry
        .deprecate_suite(
            &v1_ref,
            "smoke v1 superseded by v2 for registry history validation",
            Some(v2_ref.clone()),
        )
        .expect("deprecate v1");

    let baseline_v2 = smoke_baseline(v2_ref.clone(), SemVer::new(2, 0, 0));
    let baseline_v2_fingerprint = registry
        .register_baseline(baseline_v2)
        .expect("register smoke baseline v2");

    println!(
        "eval_registry_protocol | schema={} suites={} baselines={} exact_compatibility={}",
        EVALUATION_REGISTRY_SCHEMA_VERSION,
        registry.suite_count(),
        registry.baseline_count(),
        registry.exact_compatible(&v1_ref, &v2_ref).unwrap(),
    );
    println!("{}", v1_ref.line());
    println!("{}", v2_ref.line());
    println!(
        "baseline_ref | id=registry-smoke-dense-reference version=1.0.0 fingerprint={:016x}",
        baseline_v1_fingerprint
    );
    println!(
        "baseline_ref | id=registry-smoke-dense-reference version=2.0.0 fingerprint={:016x}",
        baseline_v2_fingerprint
    );
    let deprecation = registry
        .deprecation(&v1_ref)
        .unwrap()
        .expect("v1 deprecation");
    println!(
        "suite_deprecation | id={} version={} replacement={}@{} historical_baselines={}",
        v1_ref.id,
        v1_ref.version,
        deprecation.replacement.as_ref().unwrap().id,
        deprecation.replacement.as_ref().unwrap().version,
        registry.baselines_for_suite(&v1_ref).unwrap().len(),
    );
}
