use crate::experiment::fingerprint_bytes;
use std::error::Error;
use std::fmt;

pub const EVALUATION_REGISTRY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemVer {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl SemVer {
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self { major, minor, patch }
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
    ExactTarget,
}

impl MetricDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HigherIsBetter => "higher-is-better",
            Self::LowerIsBetter => "lower-is-better",
            Self::ExactTarget => "exact-target",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskKind {
    Reasoning,
    Code,
    Memory,
    Agent,
    Multimodal,
    Scale,
    Infrastructure,
}

impl TaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::Code => "code",
            Self::Memory => "memory",
            Self::Agent => "agent",
            Self::Multimodal => "multimodal",
            Self::Scale => "scale",
            Self::Infrastructure => "infrastructure",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetMetadata {
    pub id: String,
    pub revision: String,
    pub split: String,
    pub population: String,
    pub fixture_fingerprint: u64,
    pub sample_count: usize,
}

impl DatasetMetadata {
    fn validate(&self) -> Result<(), RegistryError> {
        nonempty("dataset id", &self.id)?;
        nonempty("dataset revision", &self.revision)?;
        nonempty("dataset split", &self.split)?;
        nonempty("dataset population", &self.population)?;
        if self.sample_count == 0 {
            return Err(RegistryError::InvalidDefinition(
                "dataset sample_count must be > 0",
            ));
        }
        Ok(())
    }

    fn canonical(&self) -> String {
        format!(
            "dataset|id={}|revision={}|split={}|population={}|fixture={:016x}|samples={}\n",
            escape(&self.id),
            escape(&self.revision),
            escape(&self.split),
            escape(&self.population),
            self.fixture_fingerprint,
            self.sample_count,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedPolicy {
    pub seeds: Vec<u64>,
}

impl SeedPolicy {
    fn validate(&self) -> Result<(), RegistryError> {
        if self.seeds.is_empty() {
            return Err(RegistryError::InvalidDefinition(
                "seed policy must contain at least one seed",
            ));
        }
        let mut seen = Vec::new();
        for seed in &self.seeds {
            if seen.contains(seed) {
                return Err(RegistryError::InvalidDefinition(
                    "seed policy contains duplicate seed",
                ));
            }
            seen.push(*seed);
        }
        Ok(())
    }

    fn canonical(&self) -> String {
        let values = self
            .seeds
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!("seeds|values={values}\n")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimits {
    pub max_examples: usize,
    pub max_steps: usize,
    pub max_tokens: usize,
    pub max_wall_ms: u64,
}

impl ResourceLimits {
    fn validate(self) -> Result<(), RegistryError> {
        if self.max_examples == 0
            || self.max_steps == 0
            || self.max_tokens == 0
            || self.max_wall_ms == 0
        {
            return Err(RegistryError::InvalidDefinition(
                "resource limits must all be > 0",
            ));
        }
        Ok(())
    }

    fn canonical(self) -> String {
        format!(
            "limits|examples={}|steps={}|tokens={}|wall_ms={}\n",
            self.max_examples, self.max_steps, self.max_tokens, self.max_wall_ms
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationTask {
    pub id: String,
    pub version: SemVer,
    pub kind: TaskKind,
    pub description: String,
    pub fixture_fingerprint: u64,
    pub fixture_count: usize,
}

impl EvaluationTask {
    fn validate(&self) -> Result<(), RegistryError> {
        nonempty("task id", &self.id)?;
        nonempty("task description", &self.description)?;
        if self.fixture_count == 0 {
            return Err(RegistryError::InvalidDefinition(
                "task fixture_count must be > 0",
            ));
        }
        Ok(())
    }

    fn canonical(&self) -> String {
        format!(
            "task|id={}|version={}|kind={}|description={}|fixture={:016x}|count={}\n",
            escape(&self.id),
            self.version,
            self.kind.as_str(),
            escape(&self.description),
            self.fixture_fingerprint,
            self.fixture_count,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationMetric {
    pub id: String,
    pub version: SemVer,
    pub unit: String,
    pub direction: MetricDirection,
    pub description: String,
}

impl EvaluationMetric {
    fn validate(&self) -> Result<(), RegistryError> {
        nonempty("metric id", &self.id)?;
        nonempty("metric unit", &self.unit)?;
        nonempty("metric description", &self.description)?;
        Ok(())
    }

    fn canonical(&self) -> String {
        format!(
            "metric|id={}|version={}|unit={}|direction={}|description={}\n",
            escape(&self.id),
            self.version,
            escape(&self.unit),
            self.direction.as_str(),
            escape(&self.description),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationSuite {
    pub id: String,
    pub version: SemVer,
    pub description: String,
    pub dataset: DatasetMetadata,
    pub tasks: Vec<EvaluationTask>,
    pub metrics: Vec<EvaluationMetric>,
    pub seed_policy: SeedPolicy,
    pub limits: ResourceLimits,
}

impl EvaluationSuite {
    pub fn validate(&self) -> Result<(), RegistryError> {
        nonempty("suite id", &self.id)?;
        nonempty("suite description", &self.description)?;
        self.dataset.validate()?;
        self.seed_policy.validate()?;
        self.limits.validate()?;
        if self.tasks.is_empty() {
            return Err(RegistryError::InvalidDefinition(
                "suite must contain at least one task",
            ));
        }
        if self.metrics.is_empty() {
            return Err(RegistryError::InvalidDefinition(
                "suite must contain at least one metric",
            ));
        }
        for task in &self.tasks {
            task.validate()?;
        }
        for metric in &self.metrics {
            metric.validate()?;
        }
        unique_versioned_ids(
            "task",
            self.tasks.iter().map(|task| (&task.id, task.version)),
        )?;
        unique_versioned_ids(
            "metric",
            self.metrics
                .iter()
                .map(|metric| (&metric.id, metric.version)),
        )?;
        Ok(())
    }

    pub fn canonical(&self) -> Result<String, RegistryError> {
        self.validate()?;
        let mut out = format!(
            "suite|schema={}|id={}|version={}|description={}\n",
            EVALUATION_REGISTRY_SCHEMA_VERSION,
            escape(&self.id),
            self.version,
            escape(&self.description),
        );
        out.push_str(&self.dataset.canonical());
        out.push_str(&self.seed_policy.canonical());
        out.push_str(&self.limits.canonical());
        for task in &self.tasks {
            out.push_str(&task.canonical());
        }
        for metric in &self.metrics {
            out.push_str(&metric.canonical());
        }
        Ok(out)
    }

    pub fn definition_fingerprint(&self) -> Result<u64, RegistryError> {
        Ok(fingerprint_bytes(self.canonical()?.as_bytes()))
    }

    pub fn exact_ref(&self) -> Result<SuiteRef, RegistryError> {
        Ok(SuiteRef {
            id: self.id.clone(),
            version: self.version,
            definition_fingerprint: self.definition_fingerprint()?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuiteRef {
    pub id: String,
    pub version: SemVer,
    pub definition_fingerprint: u64,
}

impl SuiteRef {
    pub fn line(&self) -> String {
        format!(
            "suite_ref|id={}|version={}|fingerprint={:016x}",
            escape(&self.id),
            self.version,
            self.definition_fingerprint
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationRunDescriptor {
    pub suite: SuiteRef,
    pub code_revision: String,
    pub model_id: String,
    pub checkpoint_fingerprint: Option<u64>,
    pub config_fingerprint: u64,
}

impl EvaluationRunDescriptor {
    pub fn validate(&self, registry: &EvaluationRegistry) -> Result<(), RegistryError> {
        nonempty("run code_revision", &self.code_revision)?;
        nonempty("run model_id", &self.model_id)?;
        registry.resolve_suite(&self.suite)?;
        Ok(())
    }

    pub fn line(&self) -> String {
        format!(
            "eval_run|{}|commit={}|model={}|checkpoint={}|config={:016x}",
            self.suite.line(),
            escape(&self.code_revision),
            escape(&self.model_id),
            self.checkpoint_fingerprint
                .map(|value| format!("{value:016x}"))
                .unwrap_or_else(|| "none".to_string()),
            self.config_fingerprint,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricValue {
    pub metric_id: String,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Baseline {
    pub id: String,
    pub version: SemVer,
    pub capability: String,
    pub created_date: String,
    pub code_revision: String,
    pub model_id: String,
    pub checkpoint_fingerprint: Option<u64>,
    pub config_fingerprint: u64,
    pub suite: SuiteRef,
    pub values: Vec<MetricValue>,
}

impl Baseline {
    pub fn validate(&self, suite: &EvaluationSuite) -> Result<(), RegistryError> {
        nonempty("baseline id", &self.id)?;
        nonempty("baseline capability", &self.capability)?;
        nonempty("baseline created_date", &self.created_date)?;
        nonempty("baseline code_revision", &self.code_revision)?;
        nonempty("baseline model_id", &self.model_id)?;
        if self.values.len() != suite.metrics.len() {
            return Err(RegistryError::MetricSetMismatch {
                expected: suite.metrics.len(),
                actual: self.values.len(),
            });
        }
        let mut seen = Vec::new();
        for value in &self.values {
            if !value.value.is_finite() {
                return Err(RegistryError::NonFiniteMetric(value.metric_id.clone()));
            }
            if seen.iter().any(|id: &String| id == &value.metric_id) {
                return Err(RegistryError::DuplicateMetricValue(value.metric_id.clone()));
            }
            if !suite.metrics.iter().any(|metric| metric.id == value.metric_id) {
                return Err(RegistryError::UnknownMetric(value.metric_id.clone()));
            }
            seen.push(value.metric_id.clone());
        }
        for metric in &suite.metrics {
            if !seen.iter().any(|id| id == &metric.id) {
                return Err(RegistryError::MissingMetric(metric.id.clone()));
            }
        }
        Ok(())
    }

    pub fn canonical(&self) -> Result<String, RegistryError> {
        nonempty("baseline id", &self.id)?;
        let mut out = format!(
            "baseline|schema={}|id={}|version={}|capability={}|date={}|commit={}|model={}|checkpoint={}|config={:016x}|suite_id={}|suite_version={}|suite_fingerprint={:016x}\n",
            EVALUATION_REGISTRY_SCHEMA_VERSION,
            escape(&self.id),
            self.version,
            escape(&self.capability),
            escape(&self.created_date),
            escape(&self.code_revision),
            escape(&self.model_id),
            self.checkpoint_fingerprint
                .map(|value| format!("{value:016x}"))
                .unwrap_or_else(|| "none".to_string()),
            self.config_fingerprint,
            escape(&self.suite.id),
            self.suite.version,
            self.suite.definition_fingerprint,
        );
        let mut values = self.values.clone();
        values.sort_by(|a, b| a.metric_id.cmp(&b.metric_id));
        for value in values {
            if !value.value.is_finite() {
                return Err(RegistryError::NonFiniteMetric(value.metric_id));
            }
            out.push_str(&format!(
                "value|metric={}|value={:.17}\n",
                escape(&value.metric_id),
                value.value,
            ));
        }
        Ok(out)
    }

    pub fn definition_fingerprint(&self) -> Result<u64, RegistryError> {
        Ok(fingerprint_bytes(self.canonical()?.as_bytes()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deprecation {
    pub reason: String,
    pub replacement: Option<SuiteRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SuiteRecord {
    suite: EvaluationSuite,
    fingerprint: u64,
    deprecation: Option<Deprecation>,
}

#[derive(Clone, Debug, PartialEq)]
struct BaselineRecord {
    baseline: Baseline,
    fingerprint: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvaluationRegistry {
    suites: Vec<SuiteRecord>,
    baselines: Vec<BaselineRecord>,
}

impl EvaluationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_suite(&mut self, suite: EvaluationSuite) -> Result<SuiteRef, RegistryError> {
        suite.validate()?;
        let fingerprint = suite.definition_fingerprint()?;
        if let Some(existing) = self
            .suites
            .iter()
            .find(|record| record.suite.id == suite.id && record.suite.version == suite.version)
        {
            if existing.fingerprint != fingerprint {
                return Err(RegistryError::SuiteVersionCollision {
                    id: suite.id,
                    version: suite.version,
                    existing: existing.fingerprint,
                    candidate: fingerprint,
                });
            }
            return existing.suite.exact_ref();
        }
        let suite_ref = SuiteRef {
            id: suite.id.clone(),
            version: suite.version,
            definition_fingerprint: fingerprint,
        };
        self.suites.push(SuiteRecord {
            suite,
            fingerprint,
            deprecation: None,
        });
        Ok(suite_ref)
    }

    pub fn resolve_suite(&self, suite_ref: &SuiteRef) -> Result<&EvaluationSuite, RegistryError> {
        let record = self
            .suites
            .iter()
            .find(|record| {
                record.suite.id == suite_ref.id && record.suite.version == suite_ref.version
            })
            .ok_or_else(|| RegistryError::UnknownSuite {
                id: suite_ref.id.clone(),
                version: suite_ref.version,
            })?;
        if record.fingerprint != suite_ref.definition_fingerprint {
            return Err(RegistryError::SuiteFingerprintMismatch {
                id: suite_ref.id.clone(),
                version: suite_ref.version,
                expected: record.fingerprint,
                actual: suite_ref.definition_fingerprint,
            });
        }
        Ok(&record.suite)
    }

    pub fn suite_versions(&self, id: &str) -> Vec<SuiteRef> {
        self.suites
            .iter()
            .filter(|record| record.suite.id == id)
            .map(|record| SuiteRef {
                id: record.suite.id.clone(),
                version: record.suite.version,
                definition_fingerprint: record.fingerprint,
            })
            .collect()
    }

    pub fn deprecate_suite(
        &mut self,
        suite_ref: &SuiteRef,
        reason: impl Into<String>,
        replacement: Option<SuiteRef>,
    ) -> Result<(), RegistryError> {
        let reason = reason.into();
        nonempty("deprecation reason", &reason)?;
        self.resolve_suite(suite_ref)?;
        if let Some(ref replacement_ref) = replacement {
            self.resolve_suite(replacement_ref)?;
            if replacement_ref == suite_ref {
                return Err(RegistryError::InvalidDefinition(
                    "suite cannot deprecate itself in favor of itself",
                ));
            }
        }
        let record = self
            .suites
            .iter_mut()
            .find(|record| {
                record.suite.id == suite_ref.id && record.suite.version == suite_ref.version
            })
            .expect("suite was resolved above");
        record.deprecation = Some(Deprecation {
            reason,
            replacement,
        });
        Ok(())
    }

    pub fn deprecation(&self, suite_ref: &SuiteRef) -> Result<Option<&Deprecation>, RegistryError> {
        self.resolve_suite(suite_ref)?;
        Ok(self
            .suites
            .iter()
            .find(|record| {
                record.suite.id == suite_ref.id && record.suite.version == suite_ref.version
            })
            .and_then(|record| record.deprecation.as_ref()))
    }

    pub fn register_baseline(&mut self, baseline: Baseline) -> Result<u64, RegistryError> {
        let suite = self.resolve_suite(&baseline.suite)?;
        baseline.validate(suite)?;
        let fingerprint = baseline.definition_fingerprint()?;
        if let Some(existing) = self
            .baselines
            .iter()
            .find(|record| {
                record.baseline.id == baseline.id && record.baseline.version == baseline.version
            })
        {
            if existing.fingerprint != fingerprint {
                return Err(RegistryError::BaselineVersionCollision {
                    id: baseline.id,
                    version: baseline.version,
                    existing: existing.fingerprint,
                    candidate: fingerprint,
                });
            }
            return Ok(existing.fingerprint);
        }
        self.baselines.push(BaselineRecord {
            baseline,
            fingerprint,
        });
        Ok(fingerprint)
    }

    pub fn baseline(&self, id: &str, version: SemVer) -> Option<&Baseline> {
        self.baselines
            .iter()
            .find(|record| record.baseline.id == id && record.baseline.version == version)
            .map(|record| &record.baseline)
    }

    pub fn baseline_versions(&self, id: &str) -> Vec<SemVer> {
        self.baselines
            .iter()
            .filter(|record| record.baseline.id == id)
            .map(|record| record.baseline.version)
            .collect()
    }

    pub fn baselines_for_suite(
        &self,
        suite_ref: &SuiteRef,
    ) -> Result<Vec<&Baseline>, RegistryError> {
        self.resolve_suite(suite_ref)?;
        Ok(self
            .baselines
            .iter()
            .filter(|record| record.baseline.suite == *suite_ref)
            .map(|record| &record.baseline)
            .collect())
    }

    pub fn exact_compatible(&self, a: &SuiteRef, b: &SuiteRef) -> Result<bool, RegistryError> {
        self.resolve_suite(a)?;
        self.resolve_suite(b)?;
        Ok(a == b)
    }

    pub fn suite_count(&self) -> usize {
        self.suites.len()
    }

    pub fn baseline_count(&self) -> usize {
        self.baselines.len()
    }
}

pub fn smoke_fixture_bytes() -> &'static [u8] {
    b"auralis-eval-registry-smoke-v1\ncase=identity\nanswer=1\n"
}

pub fn smoke_suite() -> EvaluationSuite {
    let fixture = smoke_fixture_bytes();
    EvaluationSuite {
        id: "registry-smoke".to_string(),
        version: SemVer::new(1, 0, 0),
        description: "offline deterministic registry contract smoke suite".to_string(),
        dataset: DatasetMetadata {
            id: "registry-smoke-fixture".to_string(),
            revision: "1".to_string(),
            split: "smoke".to_string(),
            population: "one deterministic infrastructure fixture".to_string(),
            fixture_fingerprint: fingerprint_bytes(fixture),
            sample_count: 1,
        },
        tasks: vec![EvaluationTask {
            id: "identity".to_string(),
            version: SemVer::new(1, 0, 0),
            kind: TaskKind::Infrastructure,
            description: "return the pinned fixture answer".to_string(),
            fixture_fingerprint: fingerprint_bytes(fixture),
            fixture_count: 1,
        }],
        metrics: vec![EvaluationMetric {
            id: "exact_match".to_string(),
            version: SemVer::new(1, 0, 0),
            unit: "ratio".to_string(),
            direction: MetricDirection::HigherIsBetter,
            description: "fraction of exact expected answers".to_string(),
        }],
        seed_policy: SeedPolicy { seeds: vec![659_918] },
        limits: ResourceLimits {
            max_examples: 1,
            max_steps: 1,
            max_tokens: 16,
            max_wall_ms: 1_000,
        },
    }
}

pub fn smoke_baseline(suite: SuiteRef, version: SemVer) -> Baseline {
    Baseline {
        id: "registry-smoke-dense-reference".to_string(),
        version,
        capability: "infrastructure".to_string(),
        created_date: "2026-09-22".to_string(),
        code_revision: "registry-smoke".to_string(),
        model_id: "none".to_string(),
        checkpoint_fingerprint: None,
        config_fingerprint: fingerprint_bytes(b"registry-smoke-config-v1"),
        suite,
        values: vec![MetricValue {
            metric_id: "exact_match".to_string(),
            value: 1.0,
        }],
    }
}

fn nonempty(what: &'static str, value: &str) -> Result<(), RegistryError> {
    if value.trim().is_empty() {
        Err(RegistryError::EmptyField(what))
    } else {
        Ok(())
    }
}

fn unique_versioned_ids<'a>(
    what: &'static str,
    values: impl Iterator<Item = (&'a String, SemVer)>,
) -> Result<(), RegistryError> {
    let mut seen = Vec::<(String, SemVer)>::new();
    for (id, version) in values {
        if seen.iter().any(|(seen_id, seen_version)| {
            seen_id == id && *seen_version == version
        }) {
            return Err(RegistryError::DuplicateVersionedId {
                what,
                id: id.clone(),
                version,
            });
        }
        seen.push((id.clone(), version));
    }
    Ok(())
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

#[derive(Clone, Debug, PartialEq)]
pub enum RegistryError {
    EmptyField(&'static str),
    InvalidDefinition(&'static str),
    DuplicateVersionedId {
        what: &'static str,
        id: String,
        version: SemVer,
    },
    SuiteVersionCollision {
        id: String,
        version: SemVer,
        existing: u64,
        candidate: u64,
    },
    SuiteFingerprintMismatch {
        id: String,
        version: SemVer,
        expected: u64,
        actual: u64,
    },
    UnknownSuite {
        id: String,
        version: SemVer,
    },
    MetricSetMismatch {
        expected: usize,
        actual: usize,
    },
    UnknownMetric(String),
    MissingMetric(String),
    DuplicateMetricValue(String),
    NonFiniteMetric(String),
    BaselineVersionCollision {
        id: String,
        version: SemVer,
        existing: u64,
        candidate: u64,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must be non-empty"),
            Self::InvalidDefinition(message) => write!(f, "invalid evaluation definition: {message}"),
            Self::DuplicateVersionedId { what, id, version } => {
                write!(f, "duplicate {what} id/version: {id}@{version}")
            }
            Self::SuiteVersionCollision {
                id,
                version,
                existing,
                candidate,
            } => write!(
                f,
                "suite {id}@{version} definition changed without a version bump: existing={existing:016x} candidate={candidate:016x}"
            ),
            Self::SuiteFingerprintMismatch {
                id,
                version,
                expected,
                actual,
            } => write!(
                f,
                "suite {id}@{version} fingerprint mismatch: expected={expected:016x} actual={actual:016x}"
            ),
            Self::UnknownSuite { id, version } => write!(f, "unknown suite {id}@{version}"),
            Self::MetricSetMismatch { expected, actual } => write!(
                f,
                "baseline metric count mismatch: expected {expected}, got {actual}"
            ),
            Self::UnknownMetric(id) => write!(f, "baseline references unknown metric {id}"),
            Self::MissingMetric(id) => write!(f, "baseline is missing metric {id}"),
            Self::DuplicateMetricValue(id) => write!(f, "baseline duplicates metric {id}"),
            Self::NonFiniteMetric(id) => write!(f, "baseline metric {id} is non-finite"),
            Self::BaselineVersionCollision {
                id,
                version,
                existing,
                candidate,
            } => write!(
                f,
                "baseline {id}@{version} changed without a version bump: existing={existing:016x} candidate={candidate:016x}"
            ),
        }
    }
}

impl Error for RegistryError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered_smoke() -> (EvaluationRegistry, SuiteRef) {
        let mut registry = EvaluationRegistry::new();
        let suite_ref = registry.register_suite(smoke_suite()).unwrap();
        (registry, suite_ref)
    }

    #[test]
    fn smoke_suite_has_stable_exact_reference() {
        let suite = smoke_suite();
        suite.validate().unwrap();
        let a = suite.exact_ref().unwrap();
        let b = smoke_suite().exact_ref().unwrap();
        assert_eq!(a, b);
        assert_eq!(a.id, "registry-smoke");
        assert_eq!(a.version, SemVer::new(1, 0, 0));
        assert_ne!(a.definition_fingerprint, 0);
        assert_eq!(
            suite.dataset.fixture_fingerprint,
            fingerprint_bytes(smoke_fixture_bytes())
        );
    }

    #[test]
    fn dataset_or_metric_drift_requires_a_new_suite_version() {
        let (mut registry, original) = registered_smoke();

        let mut dataset_drift = smoke_suite();
        dataset_drift.dataset.revision = "2".to_string();
        assert!(matches!(
            registry.register_suite(dataset_drift),
            Err(RegistryError::SuiteVersionCollision { .. })
        ));

        let mut metric_drift = smoke_suite();
        metric_drift.metrics[0].description = "changed scoring semantics".to_string();
        assert!(matches!(
            registry.register_suite(metric_drift),
            Err(RegistryError::SuiteVersionCollision { .. })
        ));

        let mut bumped = smoke_suite();
        bumped.version = SemVer::new(1, 1, 0);
        bumped.dataset.revision = "2".to_string();
        let new_ref = registry.register_suite(bumped).unwrap();
        assert_ne!(new_ref, original);
        assert_eq!(registry.suite_versions("registry-smoke").len(), 2);
        assert!(!registry.exact_compatible(&original, &new_ref).unwrap());
    }

    #[test]
    fn exact_suite_fingerprint_prevents_silent_reinterpretation() {
        let (registry, suite_ref) = registered_smoke();
        let mut wrong = suite_ref.clone();
        wrong.definition_fingerprint ^= 1;
        assert!(matches!(
            registry.resolve_suite(&wrong),
            Err(RegistryError::SuiteFingerprintMismatch { .. })
        ));
    }

    #[test]
    fn old_baselines_remain_queryable_after_new_versions_and_deprecation() {
        let (mut registry, v1_ref) = registered_smoke();
        let v1 = smoke_baseline(v1_ref.clone(), SemVer::new(1, 0, 0));
        let v1_fp = registry.register_baseline(v1).unwrap();

        let mut suite_v2 = smoke_suite();
        suite_v2.version = SemVer::new(2, 0, 0);
        suite_v2.description = "second smoke definition".to_string();
        let v2_ref = registry.register_suite(suite_v2).unwrap();
        registry
            .deprecate_suite(
                &v1_ref,
                "superseded by explicit v2 smoke definition",
                Some(v2_ref.clone()),
            )
            .unwrap();

        let v2 = smoke_baseline(v2_ref.clone(), SemVer::new(2, 0, 0));
        registry.register_baseline(v2).unwrap();

        assert_eq!(registry.baseline_count(), 2);
        assert_eq!(
            registry
                .baseline("registry-smoke-dense-reference", SemVer::new(1, 0, 0))
                .unwrap()
                .definition_fingerprint()
                .unwrap(),
            v1_fp
        );
        assert_eq!(registry.baseline_versions("registry-smoke-dense-reference").len(), 2);
        let deprecation = registry.deprecation(&v1_ref).unwrap().unwrap();
        assert_eq!(deprecation.replacement.as_ref(), Some(&v2_ref));
        assert_eq!(registry.baselines_for_suite(&v1_ref).unwrap().len(), 1);
    }

    #[test]
    fn baseline_must_cover_exact_metric_set_and_exact_suite() {
        let (mut registry, suite_ref) = registered_smoke();

        let mut missing = smoke_baseline(suite_ref.clone(), SemVer::new(1, 0, 0));
        missing.values.clear();
        assert!(matches!(
            registry.register_baseline(missing),
            Err(RegistryError::MetricSetMismatch { .. })
        ));

        let mut unknown = smoke_baseline(suite_ref.clone(), SemVer::new(1, 0, 0));
        unknown.values[0].metric_id = "other".to_string();
        assert!(matches!(
            registry.register_baseline(unknown),
            Err(RegistryError::UnknownMetric(_))
        ));

        let mut wrong_ref = suite_ref.clone();
        wrong_ref.definition_fingerprint ^= 0x55;
        let wrong = smoke_baseline(wrong_ref, SemVer::new(1, 0, 0));
        assert!(matches!(
            registry.register_baseline(wrong),
            Err(RegistryError::SuiteFingerprintMismatch { .. })
        ));
    }

    #[test]
    fn baseline_version_collision_preserves_history() {
        let (mut registry, suite_ref) = registered_smoke();
        let baseline = smoke_baseline(suite_ref.clone(), SemVer::new(1, 0, 0));
        let first = registry.register_baseline(baseline.clone()).unwrap();
        assert_eq!(registry.register_baseline(baseline).unwrap(), first);

        let mut changed = smoke_baseline(suite_ref, SemVer::new(1, 0, 0));
        changed.values[0].value = 0.5;
        assert!(matches!(
            registry.register_baseline(changed),
            Err(RegistryError::BaselineVersionCollision { .. })
        ));
        assert_eq!(registry.baseline_count(), 1);
    }

    #[test]
    fn run_descriptor_requires_registered_exact_suite() {
        let (registry, suite_ref) = registered_smoke();
        let run = EvaluationRunDescriptor {
            suite: suite_ref,
            code_revision: "fedf075c".to_string(),
            model_id: "auralis-tiny".to_string(),
            checkpoint_fingerprint: Some(0x1234),
            config_fingerprint: 0x5678,
        };
        run.validate(&registry).unwrap();
        assert!(run.line().contains("suite_ref|id=registry-smoke|version=1.0.0"));
    }

    #[test]
    fn duplicate_task_or_metric_versions_fail_closed() {
        let mut suite = smoke_suite();
        suite.tasks.push(suite.tasks[0].clone());
        assert!(matches!(
            suite.validate(),
            Err(RegistryError::DuplicateVersionedId { what: "task", .. })
        ));

        let mut suite = smoke_suite();
        suite.metrics.push(suite.metrics[0].clone());
        assert!(matches!(
            suite.validate(),
            Err(RegistryError::DuplicateVersionedId { what: "metric", .. })
        ));
    }
}
