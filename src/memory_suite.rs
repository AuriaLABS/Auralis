use crate::eval_registry::{
    DatasetMetadata, EvaluationMetric, EvaluationSuite, EvaluationTask, MetricDirection,
    ResourceLimits, SeedPolicy, SemVer, TaskKind,
};
use crate::experiment::{deterministic_u64, fingerprint_bytes};
use crate::memory::{
    ExternalMemory, InMemoryExternalMemory, MemoryClass, MemoryError, MemoryQuery, MemoryWrite,
};
use std::error::Error;
use std::fmt;
use std::time::Instant;

pub const MEMORY_SUITE_SCHEMA_VERSION: u32 = 1;
pub const MEMORY_SUITE_SEED: u64 = 135_659_918;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemorySuiteProfile {
    Smoke,
    Full,
}

impl MemorySuiteProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Full => "full",
        }
    }

    pub fn cases_per_kind(self) -> usize {
        match self {
            Self::Smoke => 2,
            Self::Full => 12,
        }
    }

    pub fn capacity_working_set(self) -> usize {
        match self {
            Self::Smoke => 16,
            Self::Full => 128,
        }
    }

    pub fn capacity_points(self) -> &'static [usize] {
        match self {
            Self::Smoke => &[4, 8, 16],
            Self::Full => &[8, 32, 128],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTaskKind {
    ExactRetrieval,
    TemporalOrder,
    ConflictResolution,
    StaleResolution,
    LongHorizonRecall,
}

impl MemoryTaskKind {
    pub const ALL: [Self; 5] = [
        Self::ExactRetrieval,
        Self::TemporalOrder,
        Self::ConflictResolution,
        Self::StaleResolution,
        Self::LongHorizonRecall,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactRetrieval => "exact-retrieval",
            Self::TemporalOrder => "temporal-order",
            Self::ConflictResolution => "conflict-resolution",
            Self::StaleResolution => "stale-resolution",
            Self::LongHorizonRecall => "long-horizon-recall",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryFixtureWrite {
    pub namespace: String,
    pub key: String,
    pub value: f32,
}

impl MemoryFixtureWrite {
    fn canonical(&self) -> String {
        format!(
            "write|namespace={}|key={}|value_bits={:08x}\n",
            escape(&self.namespace),
            escape(&self.key),
            self.value.to_bits()
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryCase {
    pub id: String,
    pub kind: MemoryTaskKind,
    pub capacity: usize,
    pub horizon: usize,
    pub writes: Vec<MemoryFixtureWrite>,
    pub query_namespace: String,
    pub query_key: String,
    pub expected: f32,
}

impl MemoryCase {
    pub fn canonical(&self) -> String {
        let mut out = format!(
            "case|id={}|kind={}|capacity={}|horizon={}|query_namespace={}|query_key={}|expected_bits={:08x}\n",
            escape(&self.id),
            self.kind.as_str(),
            self.capacity,
            self.horizon,
            escape(&self.query_namespace),
            escape(&self.query_key),
            self.expected.to_bits(),
        );
        for write in &self.writes {
            out.push_str(&write.canonical());
        }
        out
    }

    fn first_matching(&self) -> Option<(usize, f32)> {
        self.writes
            .iter()
            .enumerate()
            .find(|(_, write)| {
                write.namespace == self.query_namespace && write.key == self.query_key
            })
            .map(|(index, write)| (index, write.value))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryPrediction {
    pub value: Option<f32>,
    pub record_ids: Vec<u64>,
    pub selected_id: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFailureKind {
    Correct,
    Missing,
    WrongValue,
    WrongOrder,
    ConflictCapture,
    StaleValue,
}

impl MemoryFailureKind {
    pub const FAILURES: [Self; 5] = [
        Self::Missing,
        Self::WrongValue,
        Self::WrongOrder,
        Self::ConflictCapture,
        Self::StaleValue,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::Missing => "missing",
            Self::WrongValue => "wrong-value",
            Self::WrongOrder => "wrong-order",
            Self::ConflictCapture => "conflict-capture",
            Self::StaleValue => "stale-value",
        }
    }

    fn index(self) -> Option<usize> {
        match self {
            Self::Correct => None,
            Self::Missing => Some(0),
            Self::WrongValue => Some(1),
            Self::WrongOrder => Some(2),
            Self::ConflictCapture => Some(3),
            Self::StaleValue => Some(4),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryCaseScore {
    pub exact: bool,
    pub failure: MemoryFailureKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryReport {
    pub total: usize,
    pub correct: usize,
    pub failure_counts: [usize; 5],
    pub per_kind_total: [usize; 5],
    pub per_kind_correct: [usize; 5],
}

impl MemoryReport {
    pub fn exact_match_ratio(&self) -> f64 {
        ratio(self.correct, self.total)
    }

    pub fn failure_count(&self, kind: MemoryFailureKind) -> usize {
        kind.index()
            .map(|index| self.failure_counts[index])
            .unwrap_or(0)
    }

    pub fn kind_exact_match_ratio(&self, kind: MemoryTaskKind) -> f64 {
        let index = kind_index(kind);
        ratio(self.per_kind_correct[index], self.per_kind_total[index])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemorySuiteError {
    PredictionCountMismatch { expected: usize, actual: usize },
    Backend(String),
}

impl fmt::Display for MemorySuiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PredictionCountMismatch { expected, actual } => write!(
                f,
                "prediction count mismatch: expected {expected}, got {actual}"
            ),
            Self::Backend(message) => write!(f, "memory suite backend error: {message}"),
        }
    }
}

impl Error for MemorySuiteError {}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryCaseExecution {
    pub prediction: MemoryPrediction,
    pub query_ns: u64,
    pub heap_bytes: usize,
    pub snapshot_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryExecutionSummary {
    pub report: MemoryReport,
    pub query_ns_total: u64,
    pub max_heap_bytes: usize,
    pub max_snapshot_bytes: usize,
    pub min_long_horizon: usize,
    pub max_long_horizon: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapacityPoint {
    pub capacity: usize,
    pub attempted: usize,
    pub stored: usize,
    pub rejected: usize,
    pub exact_hits: usize,
    pub exact_retrieval: f64,
    pub evicted: usize,
    pub query_ns: u64,
    pub heap_bytes: usize,
    pub snapshot_bytes: usize,
}

pub fn generate_suite(profile: MemorySuiteProfile, seed: u64) -> Vec<MemoryCase> {
    let per_kind = profile.cases_per_kind();
    let mut cases = Vec::with_capacity(MemoryTaskKind::ALL.len() * per_kind);
    for (kind_index, kind) in MemoryTaskKind::ALL.iter().copied().enumerate() {
        for local_index in 0..per_kind {
            let index = kind_index * 10_000 + local_index;
            cases.push(generate_case(profile, kind, seed, index as u64, local_index));
        }
    }
    cases
}

pub fn execute_case(case: &MemoryCase) -> Result<MemoryCaseExecution, MemorySuiteError> {
    let mut memory = InMemoryExternalMemory::new(case.capacity)
        .map_err(|e| MemorySuiteError::Backend(e.to_string()))?;
    for write in &case.writes {
        memory
            .write(MemoryWrite::new(
                write.namespace.clone(),
                write.key.clone(),
                vec![write.value],
                MemoryClass::Persistent,
            ))
            .map_err(|e| MemorySuiteError::Backend(e.to_string()))?;
    }

    let query = MemoryQuery {
        namespace: Some(case.query_namespace.clone()),
        key: Some(case.query_key.clone()),
        class: Some(MemoryClass::Persistent),
        limit: case.capacity,
    };
    let started = Instant::now();
    let records = memory
        .query(&query)
        .map_err(|e| MemorySuiteError::Backend(e.to_string()))?;
    let query_ns = started.elapsed().as_nanos().min(u64::MAX as u128).max(1) as u64;
    let record_ids = records.iter().map(|record| record.id).collect::<Vec<_>>();
    let selected = records.last();
    let prediction = MemoryPrediction {
        value: selected.and_then(|record| record.value.first().copied()),
        selected_id: selected.map(|record| record.id),
        record_ids,
    };
    let snapshot_bytes = memory
        .encode()
        .map_err(|e| MemorySuiteError::Backend(e.to_string()))?
        .len();
    Ok(MemoryCaseExecution {
        prediction,
        query_ns,
        heap_bytes: memory.estimated_heap_bytes(),
        snapshot_bytes,
    })
}

pub fn execute_profile(
    profile: MemorySuiteProfile,
    seed: u64,
) -> Result<MemoryExecutionSummary, MemorySuiteError> {
    let cases = generate_suite(profile, seed);
    let mut predictions = Vec::with_capacity(cases.len());
    let mut query_ns_total = 0u64;
    let mut max_heap_bytes = 0usize;
    let mut max_snapshot_bytes = 0usize;

    for case in &cases {
        let execution = execute_case(case)?;
        query_ns_total = query_ns_total.saturating_add(execution.query_ns);
        max_heap_bytes = max_heap_bytes.max(execution.heap_bytes);
        max_snapshot_bytes = max_snapshot_bytes.max(execution.snapshot_bytes);
        predictions.push(execution.prediction);
    }

    let long_horizons = cases
        .iter()
        .filter(|case| case.kind == MemoryTaskKind::LongHorizonRecall)
        .map(|case| case.horizon)
        .collect::<Vec<_>>();

    Ok(MemoryExecutionSummary {
        report: score_predictions(&cases, &predictions)?,
        query_ns_total,
        max_heap_bytes,
        max_snapshot_bytes,
        min_long_horizon: long_horizons.iter().copied().min().unwrap_or(0),
        max_long_horizon: long_horizons.iter().copied().max().unwrap_or(0),
    })
}

pub fn score_case(case: &MemoryCase, prediction: &MemoryPrediction) -> MemoryCaseScore {
    let Some(actual) = prediction.value else {
        return fail(MemoryFailureKind::Missing);
    };

    if prediction.record_ids.is_empty() || prediction.selected_id.is_none() {
        return fail(MemoryFailureKind::WrongOrder);
    }
    if prediction.record_ids.windows(2).any(|ids| ids[0] >= ids[1]) {
        return fail(MemoryFailureKind::WrongOrder);
    }

    let latest_id = prediction.record_ids.last().copied();
    if actual.to_bits() == case.expected.to_bits() {
        if prediction.selected_id == latest_id {
            return pass();
        }
        return fail(MemoryFailureKind::WrongOrder);
    }

    let first_value = case.first_matching().map(|(_, value)| value);
    match case.kind {
        MemoryTaskKind::TemporalOrder if first_value == Some(actual) => {
            fail(MemoryFailureKind::WrongOrder)
        }
        MemoryTaskKind::ConflictResolution if first_value == Some(actual) => {
            fail(MemoryFailureKind::ConflictCapture)
        }
        MemoryTaskKind::StaleResolution if first_value == Some(actual) => {
            fail(MemoryFailureKind::StaleValue)
        }
        _ => fail(MemoryFailureKind::WrongValue),
    }
}

pub fn score_predictions(
    cases: &[MemoryCase],
    predictions: &[MemoryPrediction],
) -> Result<MemoryReport, MemorySuiteError> {
    if cases.len() != predictions.len() {
        return Err(MemorySuiteError::PredictionCountMismatch {
            expected: cases.len(),
            actual: predictions.len(),
        });
    }

    let mut report = MemoryReport {
        total: cases.len(),
        correct: 0,
        failure_counts: [0; 5],
        per_kind_total: [0; 5],
        per_kind_correct: [0; 5],
    };
    for (case, prediction) in cases.iter().zip(predictions) {
        let index = kind_index(case.kind);
        report.per_kind_total[index] += 1;
        let score = score_case(case, prediction);
        if score.exact {
            report.correct += 1;
            report.per_kind_correct[index] += 1;
        } else if let Some(failure_index) = score.failure.index() {
            report.failure_counts[failure_index] += 1;
        }
    }
    Ok(report)
}

pub fn oracle_predictions(cases: &[MemoryCase]) -> Result<Vec<MemoryPrediction>, MemorySuiteError> {
    cases
        .iter()
        .map(|case| execute_case(case).map(|execution| execution.prediction))
        .collect()
}

pub fn memory_off_predictions(cases: &[MemoryCase]) -> Vec<MemoryPrediction> {
    cases
        .iter()
        .map(|_| MemoryPrediction {
            value: None,
            record_ids: Vec::new(),
            selected_id: None,
        })
        .collect()
}

pub fn intentional_regression_predictions(cases: &[MemoryCase]) -> Vec<MemoryPrediction> {
    cases
        .iter()
        .map(|case| match case.kind {
            MemoryTaskKind::ExactRetrieval => MemoryPrediction {
                value: None,
                record_ids: Vec::new(),
                selected_id: None,
            },
            MemoryTaskKind::TemporalOrder
            | MemoryTaskKind::ConflictResolution
            | MemoryTaskKind::StaleResolution => {
                let (first_index, first_value) = case.first_matching().expect("matching fixture");
                let matching_ids = case
                    .writes
                    .iter()
                    .enumerate()
                    .filter(|(_, write)| {
                        write.namespace == case.query_namespace && write.key == case.query_key
                    })
                    .map(|(index, _)| index as u64)
                    .collect::<Vec<_>>();
                MemoryPrediction {
                    value: Some(first_value),
                    record_ids: matching_ids,
                    selected_id: Some(first_index as u64),
                }
            }
            MemoryTaskKind::LongHorizonRecall => MemoryPrediction {
                value: Some(case.expected + 1.0),
                record_ids: vec![0],
                selected_id: Some(0),
            },
        })
        .collect()
}

pub fn capacity_sweep(
    profile: MemorySuiteProfile,
) -> Result<Vec<CapacityPoint>, MemorySuiteError> {
    let attempted = profile.capacity_working_set();
    let mut out = Vec::new();
    for &capacity in profile.capacity_points() {
        let mut memory = InMemoryExternalMemory::new(capacity)
            .map_err(|e| MemorySuiteError::Backend(e.to_string()))?;
        let mut rejected = 0usize;
        for i in 0..attempted {
            match memory.write(MemoryWrite::new(
                "capacity",
                format!("key-{i:04}"),
                vec![i as f32],
                MemoryClass::Persistent,
            )) {
                Ok(_) => {}
                Err(MemoryError::AtCapacity { .. }) => rejected += 1,
                Err(error) => return Err(MemorySuiteError::Backend(error.to_string())),
            }
        }

        let started = Instant::now();
        let mut exact_hits = 0usize;
        let mut retained_prefix_hits = 0usize;
        for i in 0..attempted {
            let records = memory
                .query(&MemoryQuery::exact("capacity", format!("key-{i:04}")))
                .map_err(|e| MemorySuiteError::Backend(e.to_string()))?;
            if records
                .first()
                .and_then(|record| record.value.first())
                .copied()
                == Some(i as f32)
            {
                exact_hits += 1;
                if i < memory.len() {
                    retained_prefix_hits += 1;
                }
            }
        }
        let operations = attempted.max(1) as u128;
        let query_ns = (started.elapsed().as_nanos() / operations)
            .min(u64::MAX as u128)
            .max(1) as u64;
        let stored = memory.len();
        let evicted = stored.saturating_sub(retained_prefix_hits);
        let snapshot_bytes = memory
            .encode()
            .map_err(|e| MemorySuiteError::Backend(e.to_string()))?
            .len();

        out.push(CapacityPoint {
            capacity,
            attempted,
            stored,
            rejected,
            exact_hits,
            exact_retrieval: ratio(exact_hits, attempted),
            evicted,
            query_ns,
            heap_bytes: memory.estimated_heap_bytes(),
            snapshot_bytes,
        });
    }
    Ok(out)
}

pub fn suite_definition(profile: MemorySuiteProfile, seed: u64) -> EvaluationSuite {
    let cases = generate_suite(profile, seed);
    let canonical = canonical_cases(&cases);
    let profile_name = profile.as_str();
    let tasks = MemoryTaskKind::ALL
        .iter()
        .copied()
        .map(|kind| {
            let subset = cases
                .iter()
                .filter(|case| case.kind == kind)
                .cloned()
                .collect::<Vec<_>>();
            EvaluationTask {
                id: format!("memory.{profile_name}.{}", kind.as_str()),
                version: SemVer::new(1, 0, 0),
                kind: TaskKind::Memory,
                description: task_description(kind).to_string(),
                fixture_fingerprint: fingerprint_bytes(canonical_cases(&subset).as_bytes()),
                fixture_count: subset.len(),
            }
        })
        .collect();

    EvaluationSuite {
        id: format!("memory-external-{profile_name}"),
        version: SemVer::new(1, 0, 0),
        description: format!(
            "deterministic external-memory {profile_name} profile with objective retrieval scoring"
        ),
        dataset: DatasetMetadata {
            id: "auralis-external-memory-eval".to_string(),
            revision: "1".to_string(),
            split: profile_name.to_string(),
            population: "exact retrieval, temporal order, conflict, stale-data and long-horizon external-memory fixtures".to_string(),
            fixture_fingerprint: fingerprint_bytes(canonical.as_bytes()),
            sample_count: cases.len(),
        },
        tasks,
        metrics: memory_metrics(),
        seed_policy: SeedPolicy { seeds: vec![seed] },
        limits: ResourceLimits {
            max_examples: cases.len(),
            max_steps: profile.capacity_working_set(),
            max_tokens: match profile {
                MemorySuiteProfile::Smoke => 16_384,
                MemorySuiteProfile::Full => 262_144,
            },
            max_wall_ms: match profile {
                MemorySuiteProfile::Smoke => 10_000,
                MemorySuiteProfile::Full => 60_000,
            },
        },
    }
}

pub fn canonical_cases(cases: &[MemoryCase]) -> String {
    cases.iter().map(MemoryCase::canonical).collect()
}

fn memory_metrics() -> Vec<EvaluationMetric> {
    let mut metrics = vec![
        EvaluationMetric {
            id: "memory.exact-match".to_string(),
            version: SemVer::new(1, 0, 0),
            unit: "ratio".to_string(),
            direction: MetricDirection::HigherIsBetter,
            description: "fraction of memory cases selecting the exact expected record/value".to_string(),
        },
        EvaluationMetric {
            id: "memory.off.exact-match".to_string(),
            version: SemVer::new(1, 0, 0),
            unit: "ratio".to_string(),
            direction: MetricDirection::ExactTarget,
            description: "memory-off control exact-match; target is zero for memory-dependent fixtures".to_string(),
        },
        EvaluationMetric {
            id: "memory.long-horizon.exact-match".to_string(),
            version: SemVer::new(1, 0, 0),
            unit: "ratio".to_string(),
            direction: MetricDirection::HigherIsBetter,
            description: "exact-match ratio for long-horizon recall fixtures".to_string(),
        },
        EvaluationMetric {
            id: "memory.capacity.retrieval".to_string(),
            version: SemVer::new(1, 0, 0),
            unit: "ratio".to_string(),
            direction: MetricDirection::HigherIsBetter,
            description: "retrievable fraction under an explicit reject-new capacity sweep".to_string(),
        },
        EvaluationMetric {
            id: "memory.query-latency".to_string(),
            version: SemVer::new(1, 0, 0),
            unit: "ns".to_string(),
            direction: MetricDirection::LowerIsBetter,
            description: "descriptive external-memory query latency".to_string(),
        },
        EvaluationMetric {
            id: "memory.heap-bytes".to_string(),
            version: SemVer::new(1, 0, 0),
            unit: "bytes".to_string(),
            direction: MetricDirection::LowerIsBetter,
            description: "estimated external-memory heap footprint".to_string(),
        },
        EvaluationMetric {
            id: "memory.snapshot-bytes".to_string(),
            version: SemVer::new(1, 0, 0),
            unit: "bytes".to_string(),
            direction: MetricDirection::LowerIsBetter,
            description: "serialized persistent-memory snapshot bytes".to_string(),
        },
    ];
    for failure in MemoryFailureKind::FAILURES {
        metrics.push(EvaluationMetric {
            id: format!("memory.failure.{}", failure.as_str()),
            version: SemVer::new(1, 0, 0),
            unit: "count".to_string(),
            direction: MetricDirection::LowerIsBetter,
            description: format!(
                "number of memory cases classified as {}",
                failure.as_str()
            ),
        });
    }
    metrics
}

fn task_description(kind: MemoryTaskKind) -> &'static str {
    match kind {
        MemoryTaskKind::ExactRetrieval => "exact key/namespace retrieval with one authoritative record",
        MemoryTaskKind::TemporalOrder => "latest-record selection from an ordered history for one key",
        MemoryTaskKind::ConflictResolution => "latest authoritative value wins when a key has conflicting values",
        MemoryTaskKind::StaleResolution => "refreshed value must supersede an older stale value after intervening writes",
        MemoryTaskKind::LongHorizonRecall => "early target remains retrievable after a long sequence of unrelated writes",
    }
}

fn generate_case(
    profile: MemorySuiteProfile,
    kind: MemoryTaskKind,
    seed: u64,
    index: u64,
    local_index: usize,
) -> MemoryCase {
    let namespace = format!("memory-suite-{}", kind.as_str());
    let key = format!("target-{index:05}");
    let base = 10.0 + bounded(seed, index, 0, 900) as f32;

    let (writes, expected, horizon) = match kind {
        MemoryTaskKind::ExactRetrieval => {
            let writes = vec![
                fixture(&namespace, format!("noise-a-{index}"), base + 1000.0),
                fixture(&namespace, &key, base),
                fixture(&namespace, format!("noise-b-{index}"), base + 2000.0),
            ];
            (writes, base, 3)
        }
        MemoryTaskKind::TemporalOrder => {
            let writes = vec![
                fixture(&namespace, &key, base),
                fixture(&namespace, &key, base + 1.0),
                fixture(&namespace, &key, base + 2.0),
            ];
            (writes, base + 2.0, 3)
        }
        MemoryTaskKind::ConflictResolution => {
            let writes = vec![
                fixture(&namespace, &key, base),
                fixture("other-source", &key, base + 5000.0),
                fixture(&namespace, &key, base + 100.0),
            ];
            (writes, base + 100.0, 3)
        }
        MemoryTaskKind::StaleResolution => {
            let noise = 4 + bounded(seed, index, 1, 5);
            let mut writes = Vec::with_capacity(noise + 2);
            writes.push(fixture(&namespace, &key, base));
            for i in 0..noise {
                writes.push(fixture(
                    &namespace,
                    format!("noise-{index}-{i:03}"),
                    base + 1000.0 + i as f32,
                ));
            }
            writes.push(fixture(&namespace, &key, base + 200.0));
            let horizon = writes.len();
            (writes, base + 200.0, horizon)
        }
        MemoryTaskKind::LongHorizonRecall => {
            let horizon = match profile {
                MemorySuiteProfile::Smoke => 8 + local_index * 8,
                MemorySuiteProfile::Full => 32 + local_index * 8,
            };
            let mut writes = Vec::with_capacity(horizon + 1);
            writes.push(fixture(&namespace, &key, base));
            for i in 0..horizon {
                writes.push(fixture(
                    &namespace,
                    format!("long-noise-{index}-{i:03}"),
                    base + 3000.0 + i as f32,
                ));
            }
            (writes, base, horizon)
        }
    };

    MemoryCase {
        id: format!("{}:{:05}", kind.as_str(), index),
        kind,
        capacity: writes.len() + 2,
        horizon,
        writes,
        query_namespace: namespace,
        query_key: key,
        expected,
    }
}

fn fixture(
    namespace: impl Into<String>,
    key: impl Into<String>,
    value: f32,
) -> MemoryFixtureWrite {
    MemoryFixtureWrite {
        namespace: namespace.into(),
        key: key.into(),
        value,
    }
}

fn bounded(seed: u64, index: u64, stream: u64, upper: usize) -> usize {
    debug_assert!(upper > 0);
    (deterministic_u64(seed, index, stream) % upper as u64) as usize
}

fn kind_index(kind: MemoryTaskKind) -> usize {
    match kind {
        MemoryTaskKind::ExactRetrieval => 0,
        MemoryTaskKind::TemporalOrder => 1,
        MemoryTaskKind::ConflictResolution => 2,
        MemoryTaskKind::StaleResolution => 3,
        MemoryTaskKind::LongHorizonRecall => 4,
    }
}

fn ratio(correct: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}

fn pass() -> MemoryCaseScore {
    MemoryCaseScore {
        exact: true,
        failure: MemoryFailureKind::Correct,
    }
}

fn fail(failure: MemoryFailureKind) -> MemoryCaseScore {
    MemoryCaseScore {
        exact: false,
        failure,
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
    fn generation_is_deterministic_and_profile_sizes_are_exact() {
        for profile in [MemorySuiteProfile::Smoke, MemorySuiteProfile::Full] {
            let a = generate_suite(profile, MEMORY_SUITE_SEED);
            let b = generate_suite(profile, MEMORY_SUITE_SEED);
            assert_eq!(a, b);
            assert_eq!(a.len(), 5 * profile.cases_per_kind());
            assert_eq!(canonical_cases(&a), canonical_cases(&b));
        }
    }

    #[test]
    fn backend_oracle_passes_every_memory_family() {
        for profile in [MemorySuiteProfile::Smoke, MemorySuiteProfile::Full] {
            let cases = generate_suite(profile, MEMORY_SUITE_SEED);
            let predictions = oracle_predictions(&cases).unwrap();
            let report = score_predictions(&cases, &predictions).unwrap();
            assert_eq!(report.correct, cases.len());
            assert_eq!(report.failure_counts, [0; 5]);
            assert_eq!(report.exact_match_ratio(), 1.0);
            for kind in MemoryTaskKind::ALL {
                assert_eq!(report.kind_exact_match_ratio(kind), 1.0);
            }
        }
    }

    #[test]
    fn temporal_conflict_and_stale_cases_select_latest_matching_record() {
        let cases = generate_suite(MemorySuiteProfile::Full, MEMORY_SUITE_SEED);
        for case in cases.iter().filter(|case| {
            matches!(
                case.kind,
                MemoryTaskKind::TemporalOrder
                    | MemoryTaskKind::ConflictResolution
                    | MemoryTaskKind::StaleResolution
            )
        }) {
            let execution = execute_case(case).unwrap();
            assert_eq!(execution.prediction.value, Some(case.expected));
            assert_eq!(
                execution.prediction.selected_id,
                execution.prediction.record_ids.last().copied()
            );
            assert!(execution.prediction.record_ids.len() >= 2);
            assert!(execution
                .prediction
                .record_ids
                .windows(2)
                .all(|ids| ids[0] < ids[1]));
        }
    }

    #[test]
    fn value_without_retrieval_trace_fails_closed() {
        let case = generate_suite(MemorySuiteProfile::Smoke, MEMORY_SUITE_SEED)
            .into_iter()
            .next()
            .unwrap();
        let forged = MemoryPrediction {
            value: Some(case.expected),
            record_ids: Vec::new(),
            selected_id: None,
        };
        assert_eq!(
            score_case(&case, &forged).failure,
            MemoryFailureKind::WrongOrder
        );

        let forged = MemoryPrediction {
            value: Some(case.expected),
            record_ids: vec![0],
            selected_id: None,
        };
        assert_eq!(
            score_case(&case, &forged).failure,
            MemoryFailureKind::WrongOrder
        );
    }

    #[test]
    fn intentional_regression_exercises_failure_taxonomy() {
        let cases = generate_suite(MemorySuiteProfile::Full, MEMORY_SUITE_SEED);
        let regression = intentional_regression_predictions(&cases);
        let report = score_predictions(&cases, &regression).unwrap();
        assert_eq!(report.correct, 0);
        for failure in MemoryFailureKind::FAILURES {
            assert!(
                report.failure_count(failure) > 0,
                "missing failure category {}",
                failure.as_str()
            );
        }
    }

    #[test]
    fn memory_off_control_is_zero_and_classified_as_missing() {
        let cases = generate_suite(MemorySuiteProfile::Smoke, MEMORY_SUITE_SEED);
        let report = score_predictions(&cases, &memory_off_predictions(&cases)).unwrap();
        assert_eq!(report.correct, 0);
        assert_eq!(report.exact_match_ratio(), 0.0);
        assert_eq!(report.failure_count(MemoryFailureKind::Missing), cases.len());
    }

    #[test]
    fn capacity_sweep_is_reject_new_without_eviction() {
        for profile in [MemorySuiteProfile::Smoke, MemorySuiteProfile::Full] {
            let points = capacity_sweep(profile).unwrap();
            assert_eq!(points.len(), 3);
            let mut previous = 0.0;
            for point in &points {
                assert_eq!(point.stored, point.capacity.min(point.attempted));
                assert_eq!(point.rejected, point.attempted - point.stored);
                assert_eq!(point.exact_hits, point.stored);
                assert_eq!(point.evicted, 0);
                assert!(point.exact_retrieval >= previous);
                assert!(point.query_ns > 0);
                assert!(point.heap_bytes > 0);
                assert!(point.snapshot_bytes > 0);
                previous = point.exact_retrieval;
            }
            assert_eq!(points.last().unwrap().exact_retrieval, 1.0);
        }
    }

    #[test]
    fn full_profile_has_real_long_horizon_recall() {
        let cases = generate_suite(MemorySuiteProfile::Full, MEMORY_SUITE_SEED);
        let horizons = cases
            .iter()
            .filter(|case| case.kind == MemoryTaskKind::LongHorizonRecall)
            .map(|case| case.horizon)
            .collect::<Vec<_>>();
        assert!(horizons.iter().copied().min().unwrap() >= 32);
        assert!(horizons.iter().copied().max().unwrap() >= 120);
        for case in cases
            .iter()
            .filter(|case| case.kind == MemoryTaskKind::LongHorizonRecall)
        {
            assert!(score_case(case, &execute_case(case).unwrap().prediction).exact);
        }
    }

    #[test]
    fn suite_registry_identity_is_stable_and_profiles_differ() {
        let mut registry = EvaluationRegistry::new();
        let smoke = suite_definition(MemorySuiteProfile::Smoke, MEMORY_SUITE_SEED);
        let full = suite_definition(MemorySuiteProfile::Full, MEMORY_SUITE_SEED);
        smoke.validate().unwrap();
        full.validate().unwrap();
        assert_eq!(smoke.metrics.len(), 12);
        let smoke_ref = registry.register_suite(smoke.clone()).unwrap();
        let full_ref = registry.register_suite(full.clone()).unwrap();
        assert_ne!(smoke_ref, full_ref);
        assert!(!registry.exact_compatible(&smoke_ref, &full_ref).unwrap());
        assert_eq!(
            smoke.definition_fingerprint().unwrap(),
            suite_definition(MemorySuiteProfile::Smoke, MEMORY_SUITE_SEED)
                .definition_fingerprint()
                .unwrap()
        );
    }

    #[test]
    fn prediction_count_mismatch_fails_closed() {
        let cases = generate_suite(MemorySuiteProfile::Smoke, MEMORY_SUITE_SEED);
        assert!(matches!(
            score_predictions(&cases, &[]),
            Err(MemorySuiteError::PredictionCountMismatch { .. })
        ));
    }
}
