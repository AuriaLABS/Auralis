use auralis::capability_suite::{
    evaluate_isolated_regression, evaluate_oracle, suite_definition, CapabilityKind,
    CapabilityProfile, CapabilityStatus, CAPABILITY_SUITE_SCHEMA_VERSION, CAPABILITY_SUITE_SEED,
};

fn profiles(arg: Option<&str>) -> Vec<CapabilityProfile> {
    match arg.unwrap_or("both") {
        "smoke" => vec![CapabilityProfile::Smoke],
        "full" => vec![CapabilityProfile::Full],
        "both" => vec![CapabilityProfile::Smoke, CapabilityProfile::Full],
        other => panic!("unknown capability-suite profile {other}; expected smoke|full|both"),
    }
}

fn main() {
    println!(
        "capability_suite_protocol | schema={} seed={} capabilities=reasoning,code,memory,agent profiles=smoke,full agent=blocked-47",
        CAPABILITY_SUITE_SCHEMA_VERSION, CAPABILITY_SUITE_SEED
    );

    let arg = std::env::args().nth(1);
    for profile in profiles(arg.as_deref()) {
        let suite = suite_definition(profile);
        suite.validate().expect("valid capability suite");
        let fingerprint = suite
            .definition_fingerprint()
            .expect("capability suite fingerprint");
        let oracle = evaluate_oracle(profile).expect("oracle capability battery");
        let reasoning_reg =
            evaluate_isolated_regression(profile, CapabilityKind::Reasoning).expect("reasoning isolation");

        println!(
            "capability_suite_profile | profile={} tasks={} metrics={} suite_fingerprint={:016x} fixture_fingerprint={:016x} descriptive_mean={:.6}",
            profile.as_str(),
            suite.tasks.len(),
            suite.metrics.len(),
            fingerprint,
            suite.dataset.fixture_fingerprint,
            oracle.descriptive_mean_executed(),
        );

        for score in &oracle.capabilities {
            println!(
                "capability_suite_capability | profile={} kind={} status={} suite={} version={} schema={} fingerprint={:016x} cases={} exact_pass={:.6}",
                profile.as_str(),
                score.kind.as_str(),
                score.status.as_str(),
                score.suite_id,
                score.suite_version,
                score.schema_version,
                score.suite_fingerprint,
                score.cases,
                score.exact_ratio,
            );
        }

        let broken = reasoning_reg.score(CapabilityKind::Reasoning).unwrap();
        let code = reasoning_reg.score(CapabilityKind::Code).unwrap();
        let memory = reasoning_reg.score(CapabilityKind::Memory).unwrap();
        println!(
            "capability_suite_isolation | profile={} broken=reasoning broken_exact={:.6} code_exact={:.6} memory_exact={:.6} agent={}",
            profile.as_str(),
            broken.exact_ratio,
            code.exact_ratio,
            memory.exact_ratio,
            reasoning_reg
                .score(CapabilityKind::Agent)
                .unwrap()
                .status
                .as_str(),
        );
        assert_eq!(
            reasoning_reg.score(CapabilityKind::Agent).unwrap().status,
            CapabilityStatus::Blocked
        );
    }

    println!(
        "capability_suite_gate | modular=true no_global_score_gate=true isolated_regression=true agent_blocked=true"
    );
}
