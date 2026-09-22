use crate::eval_registry::{
    DatasetMetadata, EvaluationMetric, EvaluationSuite, EvaluationTask, MetricDirection,
    ResourceLimits, SeedPolicy, SemVer, TaskKind,
};
use crate::experiment::{deterministic_u64, fingerprint_bytes};
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

pub const REASONING_SUITE_SCHEMA_VERSION: u32 = 1;
pub const REASONING_SUITE_SEED: u64 = 130_659_918;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningProfile {
    Smoke,
    Full,
}

impl ReasoningProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Full => "full",
        }
    }

    pub fn cases_per_kind_split(self) -> usize {
        match self {
            Self::Smoke => 2,
            Self::Full => 16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningKind {
    ArithmeticComposition,
    AlgorithmicSequence,
    VariableBinding,
    FiniteStatePlanning,
    DistractorRobustness,
}

impl ReasoningKind {
    pub const ALL: [Self; 5] = [
        Self::ArithmeticComposition,
        Self::AlgorithmicSequence,
        Self::VariableBinding,
        Self::FiniteStatePlanning,
        Self::DistractorRobustness,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ArithmeticComposition => "arithmetic-composition",
            Self::AlgorithmicSequence => "algorithmic-sequence",
            Self::VariableBinding => "variable-binding",
            Self::FiniteStatePlanning => "finite-state-planning",
            Self::DistractorRobustness => "distractor-robustness",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningSplit {
    TrainDifficulty,
    EvalDifficulty,
    EvalLength,
}

impl ReasoningSplit {
    pub const ALL: [Self; 3] = [
        Self::TrainDifficulty,
        Self::EvalDifficulty,
        Self::EvalLength,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::TrainDifficulty => "train-difficulty",
            Self::EvalDifficulty => "eval-difficulty",
            Self::EvalLength => "eval-length",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReasoningAnswer {
    Integer(i64),
    IntegerSequence(Vec<i64>),
    Plan {
        actions: String,
        final_state: usize,
        steps: usize,
    },
    DistractorInteger {
        correct: i64,
        distractors: Vec<i64>,
    },
}

impl ReasoningAnswer {
    pub fn canonical(&self) -> String {
        match self {
            Self::Integer(value) => value.to_string(),
            Self::IntegerSequence(values) => values
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(","),
            Self::Plan {
                actions,
                final_state,
                steps,
            } => format!("plan={actions};state=S{final_state};steps={steps}"),
            Self::DistractorInteger { correct, .. } => correct.to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningCase {
    pub id: String,
    pub kind: ReasoningKind,
    pub split: ReasoningSplit,
    pub difficulty: u8,
    pub length: usize,
    pub prompt: String,
    pub expected: ReasoningAnswer,
}

impl ReasoningCase {
    pub fn canonical(&self) -> String {
        format!(
            "case|id={}|kind={}|split={}|difficulty={}|length={}|prompt={}|answer={}\n",
            escape(&self.id),
            self.kind.as_str(),
            self.split.as_str(),
            self.difficulty,
            self.length,
            escape(&self.prompt),
            escape(&self.expected.canonical()),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    Correct,
    InvalidFormat,
    WrongValue,
    WrongLength,
    WrongBinding,
    WrongTransition,
    DistractorCapture,
}

impl FailureKind {
    pub const FAILURES: [Self; 6] = [
        Self::InvalidFormat,
        Self::WrongValue,
        Self::WrongLength,
        Self::WrongBinding,
        Self::WrongTransition,
        Self::DistractorCapture,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::InvalidFormat => "invalid-format",
            Self::WrongValue => "wrong-value",
            Self::WrongLength => "wrong-length",
            Self::WrongBinding => "wrong-binding",
            Self::WrongTransition => "wrong-transition",
            Self::DistractorCapture => "distractor-capture",
        }
    }

    fn index(self) -> Option<usize> {
        match self {
            Self::Correct => None,
            Self::InvalidFormat => Some(0),
            Self::WrongValue => Some(1),
            Self::WrongLength => Some(2),
            Self::WrongBinding => Some(3),
            Self::WrongTransition => Some(4),
            Self::DistractorCapture => Some(5),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseScore {
    pub exact: bool,
    pub failure: FailureKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReasoningReport {
    pub total: usize,
    pub correct: usize,
    pub failure_counts: [usize; 6],
    pub per_kind_total: [usize; 5],
    pub per_kind_correct: [usize; 5],
    pub per_split_total: [usize; 3],
    pub per_split_correct: [usize; 3],
}

impl ReasoningReport {
    pub fn exact_match_ratio(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    pub fn failure_count(&self, kind: FailureKind) -> usize {
        kind.index()
            .map(|index| self.failure_counts[index])
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReasoningError {
    PredictionCountMismatch { expected: usize, actual: usize },
    InvalidProfile(&'static str),
}

impl fmt::Display for ReasoningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PredictionCountMismatch { expected, actual } => write!(
                f,
                "prediction count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidProfile(message) => write!(f, "invalid reasoning profile: {message}"),
        }
    }
}

impl Error for ReasoningError {}

pub fn generate_suite(profile: ReasoningProfile, seed: u64) -> Vec<ReasoningCase> {
    let per_bucket = profile.cases_per_kind_split();
    let mut cases = Vec::with_capacity(
        ReasoningKind::ALL.len() * ReasoningSplit::ALL.len() * per_bucket,
    );

    for (kind_index, kind) in ReasoningKind::ALL.iter().copied().enumerate() {
        for (split_index, split) in ReasoningSplit::ALL.iter().copied().enumerate() {
            for local_index in 0..per_bucket {
                let global = kind_index * 10_000 + split_index * 1_000 + local_index;
                cases.push(generate_case(kind, split, seed, global as u64));
            }
        }
    }

    cases
}

pub fn score_case(case: &ReasoningCase, prediction: &str) -> CaseScore {
    match &case.expected {
        ReasoningAnswer::Integer(expected) => match parse_integer(prediction) {
            None => fail(FailureKind::InvalidFormat),
            Some(actual) if actual == *expected => pass(),
            Some(_) if case.kind == ReasoningKind::VariableBinding => {
                fail(FailureKind::WrongBinding)
            }
            Some(_) => fail(FailureKind::WrongValue),
        },
        ReasoningAnswer::IntegerSequence(expected) => match parse_integer_sequence(prediction) {
            None => fail(FailureKind::InvalidFormat),
            Some(actual) if actual.len() != expected.len() => fail(FailureKind::WrongLength),
            Some(actual) if actual.as_slice() == expected.as_slice() => pass(),
            Some(_) => fail(FailureKind::WrongValue),
        },
        ReasoningAnswer::Plan {
            actions,
            final_state,
            steps,
        } => match parse_plan(prediction) {
            None => fail(FailureKind::InvalidFormat),
            Some((actual_actions, actual_state, actual_steps))
                if actual_actions == actions.as_str()
                    && actual_state == *final_state
                    && actual_steps == *steps =>
            {
                pass()
            }
            Some(_) => fail(FailureKind::WrongTransition),
        },
        ReasoningAnswer::DistractorInteger {
            correct,
            distractors,
        } => match parse_integer(prediction) {
            None => fail(FailureKind::InvalidFormat),
            Some(actual) if actual == *correct => pass(),
            Some(actual) if distractors.contains(&actual) => fail(FailureKind::DistractorCapture),
            Some(_) => fail(FailureKind::WrongValue),
        },
    }
}

pub fn score_predictions(
    cases: &[ReasoningCase],
    predictions: &[String],
) -> Result<ReasoningReport, ReasoningError> {
    if cases.len() != predictions.len() {
        return Err(ReasoningError::PredictionCountMismatch {
            expected: cases.len(),
            actual: predictions.len(),
        });
    }

    let mut report = ReasoningReport {
        total: cases.len(),
        correct: 0,
        failure_counts: [0; 6],
        per_kind_total: [0; 5],
        per_kind_correct: [0; 5],
        per_split_total: [0; 3],
        per_split_correct: [0; 3],
    };

    for (case, prediction) in cases.iter().zip(predictions) {
        let score = score_case(case, prediction);
        let kind_index = reasoning_kind_index(case.kind);
        let split_index = reasoning_split_index(case.split);
        report.per_kind_total[kind_index] += 1;
        report.per_split_total[split_index] += 1;

        if score.exact {
            report.correct += 1;
            report.per_kind_correct[kind_index] += 1;
            report.per_split_correct[split_index] += 1;
        } else if let Some(index) = score.failure.index() {
            report.failure_counts[index] += 1;
        }
    }

    Ok(report)
}

pub fn oracle_predictions(cases: &[ReasoningCase]) -> Vec<String> {
    cases
        .iter()
        .map(|case| case.expected.canonical())
        .collect()
}

pub fn intentional_regression_predictions(cases: &[ReasoningCase]) -> Vec<String> {
    cases
        .iter()
        .map(|case| match &case.expected {
            ReasoningAnswer::Integer(value) => {
                if case.kind == ReasoningKind::VariableBinding {
                    (value + 1).to_string()
                } else {
                    (value - 1).to_string()
                }
            }
            ReasoningAnswer::IntegerSequence(values) => {
                if values.len() > 1 {
                    values[..values.len() - 1]
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                } else {
                    String::new()
                }
            }
            ReasoningAnswer::Plan {
                actions,
                final_state,
                steps,
            } => format!(
                "plan={}B;state=S{};steps={}",
                actions,
                final_state + 1,
                steps + 1
            ),
            ReasoningAnswer::DistractorInteger { distractors, .. } => distractors[0].to_string(),
        })
        .collect()
}

pub fn suite_definition(profile: ReasoningProfile, seed: u64) -> EvaluationSuite {
    let cases = generate_suite(profile, seed);
    let all_canonical = canonical_cases(&cases);
    let profile_name = profile.as_str();

    let tasks = ReasoningKind::ALL
        .iter()
        .copied()
        .map(|kind| {
            let subset = cases
                .iter()
                .filter(|case| case.kind == kind)
                .cloned()
                .collect::<Vec<_>>();
            EvaluationTask {
                id: format!("reasoning.{profile_name}.{}", kind.as_str()),
                version: SemVer::new(1, 0, 0),
                kind: TaskKind::Reasoning,
                description: task_description(kind).to_string(),
                fixture_fingerprint: fingerprint_bytes(canonical_cases(&subset).as_bytes()),
                fixture_count: subset.len(),
            }
        })
        .collect();

    EvaluationSuite {
        id: format!("reasoning-synthetic-{profile_name}"),
        version: SemVer::new(1, 0, 0),
        description: format!(
            "deterministic synthetic reasoning {profile_name} profile with objective structured scoring"
        ),
        dataset: DatasetMetadata {
            id: "auralis-synthetic-reasoning".to_string(),
            revision: "1".to_string(),
            split: profile_name.to_string(),
            population: "synthetic arithmetic, sequence, binding, finite-state and distractor cases across train/eval-difficulty/eval-length".to_string(),
            fixture_fingerprint: fingerprint_bytes(all_canonical.as_bytes()),
            sample_count: cases.len(),
        },
        tasks,
        metrics: reasoning_metrics(),
        seed_policy: SeedPolicy { seeds: vec![seed] },
        limits: ResourceLimits {
            max_examples: cases.len(),
            max_steps: 1,
            max_tokens: match profile {
                ReasoningProfile::Smoke => 8_192,
                ReasoningProfile::Full => 131_072,
            },
            max_wall_ms: match profile {
                ReasoningProfile::Smoke => 5_000,
                ReasoningProfile::Full => 60_000,
            },
        },
    }
}

pub fn canonical_cases(cases: &[ReasoningCase]) -> String {
    cases.iter().map(ReasoningCase::canonical).collect()
}

fn reasoning_metrics() -> Vec<EvaluationMetric> {
    let mut metrics = vec![EvaluationMetric {
        id: "reasoning.exact-match".to_string(),
        version: SemVer::new(1, 0, 0),
        unit: "ratio".to_string(),
        direction: MetricDirection::HigherIsBetter,
        description: "fraction of cases matching the task-specific structured answer exactly".to_string(),
    }];

    for failure in FailureKind::FAILURES {
        metrics.push(EvaluationMetric {
            id: format!("reasoning.failure.{}", failure.as_str()),
            version: SemVer::new(1, 0, 0),
            unit: "count".to_string(),
            direction: MetricDirection::LowerIsBetter,
            description: format!(
                "number of reasoning cases classified as {}",
                failure.as_str()
            ),
        });
    }
    metrics
}

fn task_description(kind: ReasoningKind) -> &'static str {
    match kind {
        ReasoningKind::ArithmeticComposition => {
            "multi-step integer arithmetic composition with an exact final value"
        }
        ReasoningKind::AlgorithmicSequence => {
            "deterministic arithmetic-sequence continuation with exact ordered outputs"
        }
        ReasoningKind::VariableBinding => {
            "ordered variable assignment and rebinding with an exact queried value"
        }
        ReasoningKind::FiniteStatePlanning => {
            "canonical shortest-plan search in a finite deterministic transition system"
        }
        ReasoningKind::DistractorRobustness => {
            "arithmetic task with an explicit plausible but irrelevant distractor value"
        }
    }
}

fn generate_case(
    kind: ReasoningKind,
    split: ReasoningSplit,
    seed: u64,
    index: u64,
) -> ReasoningCase {
    let (difficulty, length) = difficulty_and_length(split, seed, index);
    let id = format!(
        "{}:{}:{:05}",
        kind.as_str(),
        split.as_str(),
        index
    );

    match kind {
        ReasoningKind::ArithmeticComposition => {
            arithmetic_case(id, split, difficulty, length, seed, index)
        }
        ReasoningKind::AlgorithmicSequence => {
            sequence_case(id, split, difficulty, length, seed, index)
        }
        ReasoningKind::VariableBinding => {
            binding_case(id, split, difficulty, length, seed, index)
        }
        ReasoningKind::FiniteStatePlanning => {
            planning_case(id, split, difficulty, length, seed, index)
        }
        ReasoningKind::DistractorRobustness => {
            distractor_case(id, split, difficulty, length, seed, index)
        }
    }
}

fn difficulty_and_length(split: ReasoningSplit, seed: u64, index: u64) -> (u8, usize) {
    match split {
        ReasoningSplit::TrainDifficulty => {
            let difficulty = 1 + bounded(seed, index, 0, 2) as u8;
            let length = 2 + bounded(seed, index, 1, 3);
            (difficulty, length)
        }
        ReasoningSplit::EvalDifficulty => {
            let difficulty = 3 + bounded(seed, index, 0, 3) as u8;
            let length = 3 + bounded(seed, index, 1, 2);
            (difficulty, length)
        }
        ReasoningSplit::EvalLength => {
            // Hold difficulty at the upper edge of the training band so this
            // split isolates length generalization instead of mixing in the
            // hard-only structures enabled at difficulty >= 3.
            let difficulty = 2;
            let length = 7 + bounded(seed, index, 1, 6);
            (difficulty, length)
        }
    }
}

fn arithmetic_case(
    id: String,
    split: ReasoningSplit,
    difficulty: u8,
    length: usize,
    seed: u64,
    index: u64,
) -> ReasoningCase {
    let mut value = 2 + bounded(seed, index, 10, 8) as i64;
    let initial = value;
    let mut ops = Vec::with_capacity(length);

    for step in 0..length {
        if difficulty >= 3 && step == 0 {
            let factor = 2 + difficulty as i64;
            value *= factor;
            ops.push(format!("*{factor}"));
            continue;
        }
        let operator_count = if difficulty == 1 { 2 } else { 3 };
        let selector = bounded(seed, index, 20 + step as u64, operator_count);
        match selector {
            0 => {
                let delta = 1 + bounded(seed, index, 100 + step as u64, 9) as i64;
                value += delta;
                ops.push(format!("+{delta}"));
            }
            1 => {
                let delta = 1 + bounded(seed, index, 200 + step as u64, 5) as i64;
                value -= delta;
                ops.push(format!("-{delta}"));
            }
            _ => {
                let factor = if difficulty >= 3 {
                    3 + bounded(seed, index, 300 + step as u64, 3) as i64
                } else {
                    2
                };
                value *= factor;
                ops.push(format!("*{factor}"));
            }
        }
    }

    ReasoningCase {
        id,
        kind: ReasoningKind::ArithmeticComposition,
        split,
        difficulty,
        length,
        prompt: format!(
            "Start at {initial}. Apply these operations in order: {}. Return only the final integer.",
            ops.join(" ")
        ),
        expected: ReasoningAnswer::Integer(value),
    }
}

fn sequence_acceleration(difficulty: u8, seed: u64, index: u64) -> i64 {
    if difficulty >= 3 {
        1 + bounded(seed, index, 402, 2) as i64
    } else {
        0
    }
}

fn sequence_case(
    id: String,
    split: ReasoningSplit,
    difficulty: u8,
    length: usize,
    seed: u64,
    index: u64,
) -> ReasoningCase {
    let start = bounded(seed, index, 400, 11) as i64 - 5;
    let delta = 1 + bounded(seed, index, 401, 7) as i64;
    let acceleration = sequence_acceleration(difficulty, seed, index);
    let visible = length.max(3);
    let mut values = Vec::with_capacity(visible + 2);
    let mut current = start;
    let mut current_delta = delta;
    for _ in 0..visible + 2 {
        values.push(current);
        current += current_delta;
        current_delta += acceleration;
    }
    let prefix = values[..visible]
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let expected = values[visible..visible + 2].to_vec();

    ReasoningCase {
        id,
        kind: ReasoningKind::AlgorithmicSequence,
        split,
        difficulty,
        length: visible,
        prompt: format!(
            "Continue the deterministic integer sequence: {prefix}. Return exactly the next two integers as a comma-separated pair."
        ),
        expected: ReasoningAnswer::IntegerSequence(expected),
    }
}

fn binding_case(
    id: String,
    split: ReasoningSplit,
    difficulty: u8,
    length: usize,
    seed: u64,
    index: u64,
) -> ReasoningCase {
    let mut a = 1 + bounded(seed, index, 500, 9) as i64;
    let mut b = 1 + bounded(seed, index, 501, 9) as i64;
    let mut c = a + b;
    let mut lines = vec![format!("a={a}"), format!("b={b}"), "c=a+b".to_string()];

    for step in 0..length {
        if difficulty >= 3 {
            match step % 3 {
                0 => {
                    a = c - b + difficulty as i64;
                    lines.push(format!("a=c-b+{difficulty}"));
                }
                1 => {
                    b = a + 2 * c - difficulty as i64;
                    lines.push(format!("b=a+2*c-{difficulty}"));
                }
                _ => {
                    c = 2 * a - b + difficulty as i64;
                    lines.push(format!("c=2*a-b+{difficulty}"));
                }
            }
        } else {
            match step % 3 {
                0 => {
                    a = b + 1;
                    lines.push("a=b+1".to_string());
                }
                1 => {
                    b = c + 1;
                    lines.push("b=c+1".to_string());
                }
                _ => {
                    c = a + 1;
                    lines.push("c=a+1".to_string());
                }
            }
        }
    }

    let (name, expected) = match bounded(seed, index, 502, 3) {
        0 => ("a", a),
        1 => ("b", b),
        _ => ("c", c),
    };

    ReasoningCase {
        id,
        kind: ReasoningKind::VariableBinding,
        split,
        difficulty,
        length,
        prompt: format!(
            "Execute assignments left-to-right: {}. Return only the final value of {name}.",
            lines.join("; ")
        ),
        expected: ReasoningAnswer::Integer(expected),
    }
}

fn planning_case(
    id: String,
    split: ReasoningSplit,
    difficulty: u8,
    length: usize,
    seed: u64,
    index: u64,
) -> ReasoningCase {
    let state_count = (length.saturating_mul(2).saturating_add(3)).max(7);
    let start = bounded(seed, index, 600, state_count);
    let offset = length.min(state_count - 1).max(1);
    let target = (start + offset) % state_count;
    let action_set: &[(char, usize)] = if difficulty >= 3 {
        &[('A', 1), ('B', 2), ('C', 3)]
    } else {
        &[('A', 1), ('B', 2)]
    };
    let actions = shortest_plan(start, target, state_count, action_set);
    let final_state = apply_plan(start, &actions, state_count);

    ReasoningCase {
        id,
        kind: ReasoningKind::FiniteStatePlanning,
        split,
        difficulty,
        length,
        prompt: if difficulty >= 3 {
            format!(
                "States are S0..S{}. Action A moves +1 mod {}; action B moves +2 mod {}; action C moves +3 mod {}. Start at S{start}, target S{target}. Return the lexicographically first shortest plan using A before B before C, exactly as plan=<ABC...>;state=S<n>;steps=<n>.",
                state_count - 1,
                state_count,
                state_count,
                state_count,
            )
        } else {
            format!(
                "States are S0..S{}. Action A moves +1 mod {}; action B moves +2 mod {}. Start at S{start}, target S{target}. Return the lexicographically first shortest plan using A before B, exactly as plan=<AB...>;state=S<n>;steps=<n>.",
                state_count - 1,
                state_count,
                state_count,
            )
        },
        expected: ReasoningAnswer::Plan {
            steps: actions.len(),
            actions,
            final_state,
        },
    }
}

fn distractor_case(
    id: String,
    split: ReasoningSplit,
    difficulty: u8,
    length: usize,
    seed: u64,
    index: u64,
) -> ReasoningCase {
    let x = 2 + bounded(seed, index, 700, 20) as i64;
    let y = 2 + bounded(seed, index, 701, 20) as i64;
    let correct = x + y;
    let (base_distance, spacing, jitter) = if difficulty >= 3 {
        (1i64, 2i64, 2usize)
    } else {
        (10i64, 11i64, 5usize)
    };
    let distractors = (0..length)
        .map(|i| {
            correct
                + base_distance
                + (i as i64 * spacing)
                + bounded(seed, index, 702 + i as u64, jitter) as i64
        })
        .collect::<Vec<_>>();
    let notes = distractors
        .iter()
        .enumerate()
        .map(|(i, value)| format!("note{}={value}", i + 1))
        .collect::<Vec<_>>()
        .join(", ");

    ReasoningCase {
        id,
        kind: ReasoningKind::DistractorRobustness,
        split,
        difficulty,
        length,
        prompt: format!(
            "Relevant facts: x={x}, y={y}. Compute x+y. Unrelated previous-run notes: {notes}. Ignore every unrelated note and return only the correct integer."
        ),
        expected: ReasoningAnswer::DistractorInteger {
            correct,
            distractors,
        },
    }
}

fn shortest_plan(
    start: usize,
    target: usize,
    state_count: usize,
    action_set: &[(char, usize)],
) -> String {
    if start == target {
        return String::new();
    }

    let mut queue = VecDeque::new();
    let mut visited = vec![false; state_count];
    queue.push_back((start, String::new()));
    visited[start] = true;

    while let Some((state, plan)) = queue.pop_front() {
        for &(action, delta) in action_set {
            let next = (state + delta) % state_count;
            let mut candidate = plan.clone();
            candidate.push(action);
            if next == target {
                return candidate;
            }
            if !visited[next] {
                visited[next] = true;
                queue.push_back((next, candidate));
            }
        }
    }

    unreachable!("finite ring is connected")
}

fn apply_plan(start: usize, plan: &str, state_count: usize) -> usize {
    plan.chars().fold(start, |state, action| match action {
        'A' => (state + 1) % state_count,
        'B' => (state + 2) % state_count,
        'C' => (state + 3) % state_count,
        _ => state,
    })
}

fn parse_integer(value: &str) -> Option<i64> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.chars().any(char::is_whitespace) {
        return None;
    }
    trimmed.parse::<i64>().ok()
}

fn parse_integer_sequence(value: &str) -> Option<Vec<i64>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed
        .split(',')
        .map(|part| parse_integer(part))
        .collect::<Option<Vec<_>>>()
}

fn parse_plan(value: &str) -> Option<(String, usize, usize)> {
    let trimmed = value.trim();
    let mut parts = trimmed.split(';');
    let plan = parts.next()?.strip_prefix("plan=")?.to_string();
    let state = parts
        .next()?
        .strip_prefix("state=S")?
        .parse::<usize>()
        .ok()?;
    let steps = parts
        .next()?
        .strip_prefix("steps=")?
        .parse::<usize>()
        .ok()?;
    if parts.next().is_some() || !plan.chars().all(|c| c == 'A' || c == 'B' || c == 'C') {
        return None;
    }
    Some((plan, state, steps))
}

fn pass() -> CaseScore {
    CaseScore {
        exact: true,
        failure: FailureKind::Correct,
    }
}

fn fail(failure: FailureKind) -> CaseScore {
    CaseScore {
        exact: false,
        failure,
    }
}

fn bounded(seed: u64, index: u64, stream: u64, upper: usize) -> usize {
    debug_assert!(upper > 0);
    (deterministic_u64(seed, index, stream) % upper as u64) as usize
}

fn reasoning_kind_index(kind: ReasoningKind) -> usize {
    match kind {
        ReasoningKind::ArithmeticComposition => 0,
        ReasoningKind::AlgorithmicSequence => 1,
        ReasoningKind::VariableBinding => 2,
        ReasoningKind::FiniteStatePlanning => 3,
        ReasoningKind::DistractorRobustness => 4,
    }
}

fn reasoning_split_index(split: ReasoningSplit) -> usize {
    match split {
        ReasoningSplit::TrainDifficulty => 0,
        ReasoningSplit::EvalDifficulty => 1,
        ReasoningSplit::EvalLength => 2,
    }
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_registry::EvaluationRegistry;

    #[test]
    fn generation_is_deterministic_and_profile_size_is_exact() {
        for profile in [ReasoningProfile::Smoke, ReasoningProfile::Full] {
            let a = generate_suite(profile, REASONING_SUITE_SEED);
            let b = generate_suite(profile, REASONING_SUITE_SEED);
            assert_eq!(a, b);
            assert_eq!(
                a.len(),
                5 * 3 * profile.cases_per_kind_split()
            );
            assert_eq!(canonical_cases(&a), canonical_cases(&b));
        }
    }

    #[test]
    fn splits_separate_difficulty_and_length_generalization() {
        let cases = generate_suite(ReasoningProfile::Full, REASONING_SUITE_SEED);
        let train = cases
            .iter()
            .filter(|case| case.split == ReasoningSplit::TrainDifficulty)
            .collect::<Vec<_>>();
        let hard = cases
            .iter()
            .filter(|case| case.split == ReasoningSplit::EvalDifficulty)
            .collect::<Vec<_>>();
        let long = cases
            .iter()
            .filter(|case| case.split == ReasoningSplit::EvalLength)
            .collect::<Vec<_>>();

        assert!(train.iter().all(|case| case.difficulty <= 2 && case.length <= 4));
        assert!(hard.iter().all(|case| case.difficulty >= 3));
        assert!(long
            .iter()
            .all(|case| case.length >= 7 && case.difficulty == 2));
        let max_train_length = train.iter().map(|case| case.length).max().unwrap();
        let min_long_length = long.iter().map(|case| case.length).min().unwrap();
        assert!(min_long_length > max_train_length);
    }

    #[test]
    fn every_task_kind_and_split_is_present() {
        let cases = generate_suite(ReasoningProfile::Smoke, REASONING_SUITE_SEED);
        for kind in ReasoningKind::ALL {
            for split in ReasoningSplit::ALL {
                assert!(cases.iter().any(|case| case.kind == kind && case.split == split));
            }
        }
    }

    #[test]
    fn oracle_predictions_score_exactly_one() {
        for profile in [ReasoningProfile::Smoke, ReasoningProfile::Full] {
            let cases = generate_suite(profile, REASONING_SUITE_SEED);
            let predictions = oracle_predictions(&cases);
            let report = score_predictions(&cases, &predictions).unwrap();
            assert_eq!(report.total, cases.len());
            assert_eq!(report.correct, cases.len());
            assert_eq!(report.failure_counts, [0; 6]);
            assert_eq!(report.exact_match_ratio(), 1.0);
        }
    }

    #[test]
    fn intentional_regression_is_visible_by_failure_taxonomy() {
        let cases = generate_suite(ReasoningProfile::Full, REASONING_SUITE_SEED);
        let predictions = intentional_regression_predictions(&cases);
        let report = score_predictions(&cases, &predictions).unwrap();
        assert_eq!(report.correct, 0);
        assert!(report.failure_count(FailureKind::WrongValue) > 0);
        assert!(report.failure_count(FailureKind::WrongLength) > 0);
        assert!(report.failure_count(FailureKind::WrongBinding) > 0);
        assert!(report.failure_count(FailureKind::WrongTransition) > 0);
        assert!(report.failure_count(FailureKind::DistractorCapture) > 0);
    }

    #[test]
    fn malformed_predictions_are_not_textually_forgiven() {
        let cases = generate_suite(ReasoningProfile::Smoke, REASONING_SUITE_SEED);
        let case = cases
            .iter()
            .find(|case| matches!(&case.expected, ReasoningAnswer::Integer(_)))
            .unwrap();
        let correct = case.expected.canonical();
        assert!(score_case(case, &correct).exact);
        let verbose = format!("The answer is {correct}");
        assert_eq!(
            score_case(case, &verbose).failure,
            FailureKind::InvalidFormat
        );
    }

    #[test]
    fn planning_oracle_is_shortest_and_reaches_target() {
        let cases = generate_suite(ReasoningProfile::Full, REASONING_SUITE_SEED);
        for case in cases
            .iter()
            .filter(|case| case.kind == ReasoningKind::FiniteStatePlanning)
        {
            let ReasoningAnswer::Plan {
                actions,
                final_state,
                steps,
            } = &case.expected
            else {
                panic!("planning case has non-plan answer");
            };
            assert_eq!(actions.len(), *steps);
            assert!(score_case(case, &case.expected.canonical()).exact);
        }
    }

    #[test]
    fn difficulty_split_changes_structure_in_every_family() {
        let cases = generate_suite(ReasoningProfile::Full, REASONING_SUITE_SEED);

        let hard_arithmetic = cases
            .iter()
            .filter(|case| {
                case.kind == ReasoningKind::ArithmeticComposition
                    && case.split == ReasoningSplit::EvalDifficulty
            })
            .collect::<Vec<_>>();
        assert!(hard_arithmetic.iter().all(|case| {
            let forced = 2 + case.difficulty as i64;
            case.prompt.contains(&format!("*{forced}"))
        }));

        assert!(cases.iter().filter(|case| {
            case.kind == ReasoningKind::VariableBinding
                && case.split == ReasoningSplit::EvalDifficulty
        }).all(|case| case.prompt.contains("2*")));

        assert!(cases.iter().filter(|case| {
            case.kind == ReasoningKind::FiniteStatePlanning
                && case.split == ReasoningSplit::EvalDifficulty
        }).all(|case| case.prompt.contains("action C")));

        for case in cases.iter().filter(|case| {
            case.kind == ReasoningKind::DistractorRobustness
                && case.split == ReasoningSplit::EvalDifficulty
        }) {
            let ReasoningAnswer::DistractorInteger { correct, distractors } = &case.expected else {
                unreachable!()
            };
            assert!(distractors.iter().any(|value| (value - correct).abs() <= 2));
        }

        // Hard sequence cases always use non-zero acceleration; easy cases use zero.
        for local_index in 0..ReasoningProfile::Full.cases_per_kind_split() {
            let hard_index = 10_000 + 1_000 + local_index;
            let easy_index = 10_000 + local_index;
            assert!(sequence_acceleration(3, REASONING_SUITE_SEED, hard_index as u64) > 0);
            assert_eq!(
                sequence_acceleration(2, REASONING_SUITE_SEED, easy_index as u64),
                0
            );
        }
    }

    #[test]
    fn length_split_does_not_activate_hard_only_structures() {
        let cases = generate_suite(ReasoningProfile::Full, REASONING_SUITE_SEED);
        for case in cases
            .iter()
            .filter(|case| case.split == ReasoningSplit::EvalLength)
        {
            assert_eq!(case.difficulty, 2);
            match &case.expected {
                ReasoningAnswer::Plan { .. } => {
                    assert!(!case.prompt.contains("action C"));
                }
                ReasoningAnswer::DistractorInteger {
                    correct,
                    distractors,
                } => {
                    assert!(distractors
                        .iter()
                        .all(|value| (value - correct).abs() >= 10));
                }
                _ => {}
            }
            if case.kind == ReasoningKind::VariableBinding {
                assert!(!case.prompt.contains("2*"));
            }
        }
    }

    #[test]
    fn length_split_increases_planning_horizon() {
        let cases = generate_suite(ReasoningProfile::Full, REASONING_SUITE_SEED);
        let plan_steps = |split| {
            cases
                .iter()
                .filter(|case| {
                    case.kind == ReasoningKind::FiniteStatePlanning && case.split == split
                })
                .map(|case| match &case.expected {
                    ReasoningAnswer::Plan { steps, .. } => *steps,
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>()
        };
        let train = plan_steps(ReasoningSplit::TrainDifficulty);
        let long = plan_steps(ReasoningSplit::EvalLength);
        assert!(long.iter().copied().min().unwrap() > train.iter().copied().max().unwrap());
    }

    #[test]
    fn registry_definitions_are_stable_and_smoke_full_are_not_interchangeable() {
        let mut registry = EvaluationRegistry::new();
        let smoke = suite_definition(ReasoningProfile::Smoke, REASONING_SUITE_SEED);
        let full = suite_definition(ReasoningProfile::Full, REASONING_SUITE_SEED);
        smoke.validate().unwrap();
        full.validate().unwrap();
        let smoke_ref = registry.register_suite(smoke.clone()).unwrap();
        let full_ref = registry.register_suite(full.clone()).unwrap();
        assert_ne!(smoke_ref, full_ref);
        assert!(!registry.exact_compatible(&smoke_ref, &full_ref).unwrap());
        assert_eq!(
            smoke.definition_fingerprint().unwrap(),
            suite_definition(ReasoningProfile::Smoke, REASONING_SUITE_SEED)
                .definition_fingerprint()
                .unwrap()
        );
    }

    #[test]
    fn prediction_count_mismatch_fails_closed() {
        let cases = generate_suite(ReasoningProfile::Smoke, REASONING_SUITE_SEED);
        assert!(matches!(
            score_predictions(&cases, &[]),
            Err(ReasoningError::PredictionCountMismatch { .. })
        ));
    }
}
