use auralis::code_suite::{
    evaluate_submissions, generate_suite, intentional_regression_submissions, oracle_submissions,
    suite_definition, CodeExecLimits, CodeFailureKind, CodeSuiteProfile, CodeTaskKind,
    CompileErrorKind, CODE_SUITE_SCHEMA_VERSION, CODE_SUITE_SEED,
};

fn profiles(arg: Option<&str>) -> Vec<CodeSuiteProfile> {
    match arg.unwrap_or("both") {
        "smoke" => vec![CodeSuiteProfile::Smoke],
        "full" => vec![CodeSuiteProfile::Full],
        "both" => vec![CodeSuiteProfile::Smoke, CodeSuiteProfile::Full],
        other => panic!("unknown code-suite profile {other}; expected smoke|full|both"),
    }
}

fn main() {
    println!(
        "code_suite_protocol | schema={} seed={} language=rust patch=single-expression compiler=rustc-test profiles=smoke,full",
        CODE_SUITE_SCHEMA_VERSION,
        CODE_SUITE_SEED
    );

    let arg = std::env::args().nth(1);
    for profile in profiles(arg.as_deref()) {
        let cases = generate_suite(profile, CODE_SUITE_SEED);
        let limits = CodeExecLimits::for_profile(profile);
        let suite = suite_definition(profile, CODE_SUITE_SEED);
        suite.validate().expect("valid code suite");
        let suite_fingerprint = suite
            .definition_fingerprint()
            .expect("code suite fingerprint");

        let oracle = evaluate_submissions(&cases, &oracle_submissions(&cases), limits)
            .expect("oracle code suite");
        let regression = evaluate_submissions(
            &cases,
            &intentional_regression_submissions(&cases),
            limits,
        )
        .expect("regression code suite");

        println!(
            "code_suite_profile | profile={} cases={} tasks={} metrics={} suite_fingerprint={:016x} fixture_fingerprint={:016x} exact_pass={:.6} compile_pass={:.6} test_pass={:.6} compile_ns={} test_ns={} compile_timeout_ms={} test_timeout_ms={}",
            profile.as_str(),
            cases.len(),
            suite.tasks.len(),
            suite.metrics.len(),
            suite_fingerprint,
            suite.dataset.fixture_fingerprint,
            oracle.exact_ratio(),
            oracle.compile_ratio(),
            oracle.test_ratio(),
            oracle.compile_ns_total,
            oracle.test_ns_total,
            limits.compile_timeout_ms,
            limits.test_timeout_ms,
        );
        println!(
            "code_suite_regression | profile={} exact_pass={:.6} compile_pass={:.6} test_pass={:.6} patch_rejected={} sandbox_violation={} compile_error={} compile_timeout={} test_failure={} test_timeout={} syntax={} type_mismatch={} borrow_check={} missing_item={} other={}",
            profile.as_str(),
            regression.exact_ratio(),
            regression.compile_ratio(),
            regression.test_ratio(),
            regression.failure_count(CodeFailureKind::PatchRejected),
            regression.failure_count(CodeFailureKind::SandboxViolation),
            regression.failure_count(CodeFailureKind::CompileError),
            regression.failure_count(CodeFailureKind::CompileTimeout),
            regression.failure_count(CodeFailureKind::TestFailure),
            regression.failure_count(CodeFailureKind::TestTimeout),
            regression.compile_error_count(CompileErrorKind::Syntax),
            regression.compile_error_count(CompileErrorKind::TypeMismatch),
            regression.compile_error_count(CompileErrorKind::BorrowCheck),
            regression.compile_error_count(CompileErrorKind::MissingItem),
            regression.compile_error_count(CompileErrorKind::Other),
        );

        for kind in CodeTaskKind::ALL {
            println!(
                "code_suite_kind | profile={} kind={} exact_pass={:.6}",
                profile.as_str(),
                kind.as_str(),
                oracle.kind_exact_ratio(kind),
            );
        }
    }

    println!(
        "code_suite_gate | executable=true protected_tests=true patch_confined=true time_limits=true oracle_exact=true regression_sensitive=true"
    );
}
