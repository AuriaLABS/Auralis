use auralis::agent_eval::{
    generate_suite, score_suite, suite_definition, AgentEvalProfile, AGENT_EVAL_SCHEMA_VERSION,
    AGENT_EVAL_SEED,
};

fn profiles(arg: Option<&str>) -> Vec<AgentEvalProfile> {
    match arg {
        Some("full") => vec![AgentEvalProfile::Full],
        Some("both") => vec![AgentEvalProfile::Smoke, AgentEvalProfile::Full],
        _ => vec![AgentEvalProfile::Smoke],
    }
}

fn main() {
    println!(
        "agent_eval_protocol | schema={} seed={} replay=mock-only redaction=true",
        AGENT_EVAL_SCHEMA_VERSION, AGENT_EVAL_SEED
    );
    let arg = std::env::args().nth(1);
    for profile in profiles(arg.as_deref()) {
        let suite = suite_definition(profile, AGENT_EVAL_SEED);
        suite.validate().expect("valid agent suite");
        let cases = generate_suite(profile, AGENT_EVAL_SEED);
        let (_, oracle) = score_suite(&cases, false).expect("oracle");
        let (_, regress) = score_suite(&cases, true).expect("regression");
        println!(
            "agent_eval_profile | profile={} cases={} exact_pass={:.6} tool_fail={} reasoning_fail={} permission_fail={} steps={} calls={} latency_ms={}",
            profile.as_str(),
            oracle.cases,
            oracle.exact_ratio(),
            oracle.tool_failures,
            oracle.reasoning_failures,
            oracle.permission_failures,
            oracle.steps,
            oracle.tool_calls,
            oracle.latency_ms
        );
        println!(
            "agent_eval_regression | profile={} exact_pass={:.6}",
            profile.as_str(),
            regress.exact_ratio()
        );
        assert_eq!(oracle.exact_ratio(), 1.0);
        assert_eq!(regress.exact_ratio(), 0.0);
    }
    println!("agent_eval_gate | replay_no_effects=true redaction=true isolated_regression=true");
}
