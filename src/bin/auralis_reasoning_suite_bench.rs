use auralis::reasoning_suite::{
    canonical_cases, generate_suite, intentional_regression_predictions, oracle_predictions,
    score_predictions, suite_definition, FailureKind, ReasoningProfile,
    REASONING_SUITE_SCHEMA_VERSION, REASONING_SUITE_SEED,
};
use std::time::Instant;

fn parse_profile(arg: Option<&str>) -> Vec<ReasoningProfile> {
    match arg.unwrap_or("both") {
        "smoke" => vec![ReasoningProfile::Smoke],
        "full" => vec![ReasoningProfile::Full],
        "both" => vec![ReasoningProfile::Smoke, ReasoningProfile::Full],
        other => panic!("unknown reasoning profile {other}; expected smoke|full|both"),
    }
}

fn main() {
    println!(
        "reasoning_suite_protocol | schema={} seed={} scoring=objective profiles=smoke,full",
        REASONING_SUITE_SCHEMA_VERSION, REASONING_SUITE_SEED
    );

    let arg = std::env::args().nth(1);
    for profile in parse_profile(arg.as_deref()) {
        let started = Instant::now();
        let cases = generate_suite(profile, REASONING_SUITE_SEED);
        let generation_ns = started.elapsed().as_nanos().max(1) as u64;

        let suite = suite_definition(profile, REASONING_SUITE_SEED);
        suite.validate().expect("valid reasoning suite definition");
        let suite_fp = suite
            .definition_fingerprint()
            .expect("suite definition fingerprint");

        let canonical_a = canonical_cases(&cases);
        let canonical_b =
            canonical_cases(&generate_suite(profile, REASONING_SUITE_SEED));
        assert_eq!(canonical_a, canonical_b);

        let oracle = oracle_predictions(&cases);
        let started = Instant::now();
        let oracle_report = score_predictions(&cases, &oracle).expect("oracle score");
        let oracle_score_ns = started.elapsed().as_nanos().max(1) as u64;
        assert_eq!(oracle_report.correct, cases.len());
        assert_eq!(oracle_report.failure_counts, [0; 6]);

        let regression = intentional_regression_predictions(&cases);
        let regression_report =
            score_predictions(&cases, &regression).expect("regression score");
        assert_eq!(regression_report.correct, 0);

        let train_lengths = cases
            .iter()
            .filter(|case| case.split.as_str() == "train-difficulty")
            .map(|case| case.length)
            .collect::<Vec<_>>();
        let hard_difficulties = cases
            .iter()
            .filter(|case| case.split.as_str() == "eval-difficulty")
            .map(|case| case.difficulty)
            .collect::<Vec<_>>();
        let long_cases = cases
            .iter()
            .filter(|case| case.split.as_str() == "eval-length")
            .collect::<Vec<_>>();
        let long_lengths = long_cases
            .iter()
            .map(|case| case.length)
            .collect::<Vec<_>>();
        let long_difficulties = long_cases
            .iter()
            .map(|case| case.difficulty)
            .collect::<Vec<_>>();

        println!(
            "reasoning_suite_profile | profile={} cases={} tasks={} metrics={} suite_fingerprint={:016x} fixture_fingerprint={:016x} generation_ns={} oracle_score_ns={} train_max_length={} hard_min_difficulty={} long_min_length={} long_max_difficulty={}",
            profile.as_str(),
            cases.len(),
            suite.tasks.len(),
            suite.metrics.len(),
            suite_fp,
            suite.dataset.fixture_fingerprint,
            generation_ns,
            oracle_score_ns,
            train_lengths.iter().copied().max().unwrap_or(0),
            hard_difficulties.iter().copied().min().unwrap_or(0),
            long_lengths.iter().copied().min().unwrap_or(0),
            long_difficulties.iter().copied().max().unwrap_or(0),
        );
        println!(
            "reasoning_suite_oracle | profile={} exact_match={:.6} correct={} total={} failures=0",
            profile.as_str(),
            oracle_report.exact_match_ratio(),
            oracle_report.correct,
            oracle_report.total,
        );
        println!(
            "reasoning_suite_regression | profile={} exact_match={:.6} invalid_format={} wrong_value={} wrong_length={} wrong_binding={} wrong_transition={} distractor_capture={}",
            profile.as_str(),
            regression_report.exact_match_ratio(),
            regression_report.failure_count(FailureKind::InvalidFormat),
            regression_report.failure_count(FailureKind::WrongValue),
            regression_report.failure_count(FailureKind::WrongLength),
            regression_report.failure_count(FailureKind::WrongBinding),
            regression_report.failure_count(FailureKind::WrongTransition),
            regression_report.failure_count(FailureKind::DistractorCapture),
        );
    }
}
