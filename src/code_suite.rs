use crate::eval_registry::{
    DatasetMetadata, EvaluationMetric, EvaluationSuite, EvaluationTask, MetricDirection,
    ResourceLimits, SeedPolicy, SemVer, TaskKind,
};
use crate::experiment::{deterministic_u64, fingerprint_bytes};
use crate::sec_path;
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub const CODE_SUITE_SCHEMA_VERSION: u32 = 1;
pub const CODE_SUITE_SEED: u64 = 131_659_918;
static NEXT_SANDBOX: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeSuiteProfile {
    Smoke,
    Full,
}

impl CodeSuiteProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Full => "full",
        }
    }

    pub fn cases_per_kind(self) -> usize {
        match self {
            Self::Smoke => 1,
            Self::Full => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeTaskKind {
    Comprehension,
    Compile,
    UnitTests,
    BugFix,
}

impl CodeTaskKind {
    pub const ALL: [Self; 4] = [
        Self::Comprehension,
        Self::Compile,
        Self::UnitTests,
        Self::BugFix,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Comprehension => "comprehension",
            Self::Compile => "compile",
            Self::UnitTests => "unit-tests",
            Self::BugFix => "bug-fix",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeExecLimits {
    pub max_source_bytes: usize,
    pub max_patch_bytes: usize,
    pub max_compile_output_bytes: usize,
    pub compile_timeout_ms: u64,
    pub test_timeout_ms: u64,
}

impl CodeExecLimits {
    pub fn for_profile(_profile: CodeSuiteProfile) -> Self {
        Self {
            max_source_bytes: 32 * 1024,
            max_patch_bytes: 256,
            max_compile_output_bytes: 256 * 1024,
            compile_timeout_ms: 10_000,
            test_timeout_ms: 5_000,
        }
    }

    fn validate(self) -> Result<(), CodeSuiteError> {
        if self.max_source_bytes == 0
            || self.max_patch_bytes == 0
            || self.max_compile_output_bytes == 0
            || self.compile_timeout_ms == 0
            || self.test_timeout_ms == 0
        {
            return Err(CodeSuiteError::Harness(
                "code execution limits must all be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodePatch {
    pub path: String,
    pub expected_old: String,
    pub replacement: String,
}

impl CodePatch {
    pub fn canonical(&self) -> String {
        format!(
            "patch|path={}|old={}|replacement={}\n",
            escape(&self.path),
            escape(&self.expected_old),
            escape(&self.replacement)
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeCase {
    pub id: String,
    pub kind: CodeTaskKind,
    pub prompt: String,
    pub base_source: String,
    pub patch_target: String,
    pub oracle_replacement: String,
    pub expected_tests: usize,
}

impl CodeCase {
    pub fn oracle_patch(&self) -> CodePatch {
        CodePatch {
            path: "main.rs".to_string(),
            expected_old: self.patch_target.clone(),
            replacement: self.oracle_replacement.clone(),
        }
    }

    pub fn canonical(&self) -> String {
        format!(
            "case|id={}|kind={}|prompt={}|target={}|oracle={}|tests={}\nsource={}\n",
            escape(&self.id),
            self.kind.as_str(),
            escape(&self.prompt),
            escape(&self.patch_target),
            escape(&self.oracle_replacement),
            self.expected_tests,
            escape(&self.base_source),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileErrorKind {
    Syntax,
    TypeMismatch,
    BorrowCheck,
    MissingItem,
    Other,
}

impl CompileErrorKind {
    pub const ALL: [Self; 5] = [
        Self::Syntax,
        Self::TypeMismatch,
        Self::BorrowCheck,
        Self::MissingItem,
        Self::Other,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Syntax => "syntax",
            Self::TypeMismatch => "type-mismatch",
            Self::BorrowCheck => "borrow-check",
            Self::MissingItem => "missing-item",
            Self::Other => "other",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Syntax => 0,
            Self::TypeMismatch => 1,
            Self::BorrowCheck => 2,
            Self::MissingItem => 3,
            Self::Other => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeFailureKind {
    Correct,
    PatchRejected,
    SandboxViolation,
    CompileError,
    CompileTimeout,
    TestFailure,
    TestTimeout,
}

impl CodeFailureKind {
    pub const FAILURES: [Self; 6] = [
        Self::PatchRejected,
        Self::SandboxViolation,
        Self::CompileError,
        Self::CompileTimeout,
        Self::TestFailure,
        Self::TestTimeout,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::PatchRejected => "patch-rejected",
            Self::SandboxViolation => "sandbox-violation",
            Self::CompileError => "compile-error",
            Self::CompileTimeout => "compile-timeout",
            Self::TestFailure => "test-failure",
            Self::TestTimeout => "test-timeout",
        }
    }

    fn index(self) -> Option<usize> {
        match self {
            Self::Correct => None,
            Self::PatchRejected => Some(0),
            Self::SandboxViolation => Some(1),
            Self::CompileError => Some(2),
            Self::CompileTimeout => Some(3),
            Self::TestFailure => Some(4),
            Self::TestTimeout => Some(5),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeCaseResult {
    pub failure: CodeFailureKind,
    pub compile_error: Option<CompileErrorKind>,
    pub compile_passed: bool,
    pub test_passed: bool,
    pub compile_ns: u64,
    pub test_ns: u64,
}

impl CodeCaseResult {
    pub fn exact(&self) -> bool {
        self.failure == CodeFailureKind::Correct
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CodeReport {
    pub total: usize,
    pub exact: usize,
    pub compile_passed: usize,
    pub test_passed: usize,
    pub failure_counts: [usize; 6],
    pub compile_error_counts: [usize; 5],
    pub per_kind_total: [usize; 4],
    pub per_kind_exact: [usize; 4],
    pub compile_ns_total: u64,
    pub test_ns_total: u64,
}

impl CodeReport {
    pub fn exact_ratio(&self) -> f64 {
        ratio(self.exact, self.total)
    }

    pub fn compile_ratio(&self) -> f64 {
        ratio(self.compile_passed, self.total)
    }

    pub fn test_ratio(&self) -> f64 {
        ratio(self.test_passed, self.total)
    }

    pub fn failure_count(&self, kind: CodeFailureKind) -> usize {
        kind.index()
            .map(|index| self.failure_counts[index])
            .unwrap_or(0)
    }

    pub fn compile_error_count(&self, kind: CompileErrorKind) -> usize {
        self.compile_error_counts[kind.index()]
    }

    pub fn kind_exact_ratio(&self, kind: CodeTaskKind) -> f64 {
        let index = kind_index(kind);
        ratio(self.per_kind_exact[index], self.per_kind_total[index])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodeSuiteError {
    PredictionCountMismatch { expected: usize, actual: usize },
    Harness(String),
}

impl fmt::Display for CodeSuiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PredictionCountMismatch { expected, actual } => {
                write!(f, "submission count mismatch: expected {expected}, got {actual}")
            }
            Self::Harness(message) => write!(f, "code-suite harness error: {message}"),
        }
    }
}

impl Error for CodeSuiteError {}

#[derive(Debug)]
struct ProcessResult {
    success: bool,
    timed_out: bool,
    elapsed_ns: u64,
}

struct SandboxGuard {
    root: PathBuf,
}

impl Drop for SandboxGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

pub fn generate_suite(profile: CodeSuiteProfile, seed: u64) -> Vec<CodeCase> {
    let per_kind = profile.cases_per_kind();
    let mut cases = Vec::with_capacity(CodeTaskKind::ALL.len() * per_kind);
    for (kind_i, kind) in CodeTaskKind::ALL.iter().copied().enumerate() {
        for local in 0..per_kind {
            let index = kind_i * 10_000 + local;
            cases.push(generate_case(kind, seed, index as u64, local));
        }
    }
    cases
}

pub fn oracle_submissions(cases: &[CodeCase]) -> Vec<CodePatch> {
    cases.iter().map(CodeCase::oracle_patch).collect()
}

pub fn intentional_regression_submissions(cases: &[CodeCase]) -> Vec<CodePatch> {
    cases
        .iter()
        .map(|case| {
            let replacement = match case.kind {
                CodeTaskKind::Comprehension => "0".to_string(),
                CodeTaskKind::Compile => "missing_symbol".to_string(),
                CodeTaskKind::UnitTests | CodeTaskKind::BugFix => case.patch_target.clone(),
            };
            CodePatch {
                path: "main.rs".into(),
                expected_old: case.patch_target.clone(),
                replacement,
            }
        })
        .collect()
}

pub fn apply_patch(
    case: &CodeCase,
    patch: &CodePatch,
    limits: CodeExecLimits,
) -> Result<String, CodeFailureKind> {
    if limits.validate().is_err() {
        return Err(CodeFailureKind::SandboxViolation);
    }
    if patch.path != "main.rs" {
        return Err(CodeFailureKind::PatchRejected);
    }
    let virtual_root = Path::new("/auralis-code-suite");
    let confined = sec_path::confine(virtual_root, Path::new(&patch.path))
        .map_err(|_| CodeFailureKind::PatchRejected)?;
    if confined != virtual_root.join("main.rs") {
        return Err(CodeFailureKind::PatchRejected);
    }
    if patch.expected_old != case.patch_target
        || patch.expected_old.is_empty()
        || case.base_source.matches(&patch.expected_old).count() != 1
    {
        return Err(CodeFailureKind::PatchRejected);
    }
    validate_replacement(&patch.replacement, limits)?;
    let patched = case
        .base_source
        .replacen(&patch.expected_old, &patch.replacement, 1);
    if patched.len() > limits.max_source_bytes {
        return Err(CodeFailureKind::SandboxViolation);
    }
    Ok(patched)
}

pub fn evaluate_submission(
    case: &CodeCase,
    patch: &CodePatch,
    limits: CodeExecLimits,
) -> Result<CodeCaseResult, CodeSuiteError> {
    limits.validate()?;
    let source = match apply_patch(case, patch, limits) {
        Ok(source) => source,
        Err(failure) => {
            return Ok(CodeCaseResult {
                failure,
                compile_error: None,
                compile_passed: false,
                test_passed: false,
                compile_ns: 0,
                test_ns: 0,
            })
        }
    };

    let (guard, source_path, binary_path, stderr_path) = create_sandbox(case, &source)?;
    let rustc = find_rustc()?;
    let stderr_file = File::create(&stderr_path)
        .map_err(|e| CodeSuiteError::Harness(format!("create compiler stderr: {e}")))?;
    let mut compile = Command::new(rustc);
    compile
        .current_dir(&guard.root)
        .arg("--edition=2021")
        .arg("--test")
        .arg(&source_path)
        .arg("-Awarnings")
        .arg("-C")
        .arg("opt-level=0")
        .arg("-C")
        .arg("debuginfo=0")
        .arg("--error-format=short")
        .arg("-o")
        .arg(&binary_path)
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file))
        .env("TZ", "UTC")
        .env("LC_ALL", "C")
        .env("LANG", "C");

    let compile_result = run_with_timeout(
        &mut compile,
        Duration::from_millis(limits.compile_timeout_ms),
    )?;
    if compile_result.timed_out {
        return Ok(CodeCaseResult {
            failure: CodeFailureKind::CompileTimeout,
            compile_error: None,
            compile_passed: false,
            test_passed: false,
            compile_ns: compile_result.elapsed_ns,
            test_ns: 0,
        });
    }
    if !compile_result.success {
        let stderr = read_limited(&stderr_path, limits.max_compile_output_bytes)?;
        return Ok(CodeCaseResult {
            failure: CodeFailureKind::CompileError,
            compile_error: Some(classify_compile_error(&stderr)),
            compile_passed: false,
            test_passed: false,
            compile_ns: compile_result.elapsed_ns,
            test_ns: 0,
        });
    }

    let mut test = Command::new(&binary_path);
    test.current_dir(&guard.root)
        .arg("--quiet")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .env("TZ", "UTC")
        .env("LC_ALL", "C")
        .env("LANG", "C");
    let test_result =
        run_with_timeout(&mut test, Duration::from_millis(limits.test_timeout_ms))?;
    if test_result.timed_out {
        return Ok(CodeCaseResult {
            failure: CodeFailureKind::TestTimeout,
            compile_error: None,
            compile_passed: true,
            test_passed: false,
            compile_ns: compile_result.elapsed_ns,
            test_ns: test_result.elapsed_ns,
        });
    }

    let passed = test_result.success;
    Ok(CodeCaseResult {
        failure: if passed {
            CodeFailureKind::Correct
        } else {
            CodeFailureKind::TestFailure
        },
        compile_error: None,
        compile_passed: true,
        test_passed: passed,
        compile_ns: compile_result.elapsed_ns,
        test_ns: test_result.elapsed_ns,
    })
}

pub fn evaluate_submissions(
    cases: &[CodeCase],
    submissions: &[CodePatch],
    limits: CodeExecLimits,
) -> Result<CodeReport, CodeSuiteError> {
    if cases.len() != submissions.len() {
        return Err(CodeSuiteError::PredictionCountMismatch {
            expected: cases.len(),
            actual: submissions.len(),
        });
    }

    let mut report = CodeReport {
        total: cases.len(),
        exact: 0,
        compile_passed: 0,
        test_passed: 0,
        failure_counts: [0; 6],
        compile_error_counts: [0; 5],
        per_kind_total: [0; 4],
        per_kind_exact: [0; 4],
        compile_ns_total: 0,
        test_ns_total: 0,
    };

    for (case, submission) in cases.iter().zip(submissions) {
        let result = evaluate_submission(case, submission, limits)?;
        let kind = kind_index(case.kind);
        report.per_kind_total[kind] += 1;
        report.compile_ns_total = report.compile_ns_total.saturating_add(result.compile_ns);
        report.test_ns_total = report.test_ns_total.saturating_add(result.test_ns);
        if result.compile_passed {
            report.compile_passed += 1;
        }
        if result.test_passed {
            report.test_passed += 1;
        }
        if result.exact() {
            report.exact += 1;
            report.per_kind_exact[kind] += 1;
        } else if let Some(index) = result.failure.index() {
            report.failure_counts[index] += 1;
        }
        if let Some(error) = result.compile_error {
            report.compile_error_counts[error.index()] += 1;
        }
    }
    Ok(report)
}

pub fn suite_definition(profile: CodeSuiteProfile, seed: u64) -> EvaluationSuite {
    let cases = generate_suite(profile, seed);
    let limits = CodeExecLimits::for_profile(profile);
    let canonical = canonical_cases(&cases);
    let profile_name = profile.as_str();
    let tasks = CodeTaskKind::ALL
        .iter()
        .copied()
        .map(|kind| {
            let subset = cases
                .iter()
                .filter(|case| case.kind == kind)
                .cloned()
                .collect::<Vec<_>>();
            EvaluationTask {
                id: format!("code.{profile_name}.{}", kind.as_str()),
                version: SemVer::new(1, 0, 0),
                kind: TaskKind::Code,
                description: task_description(kind).to_string(),
                fixture_fingerprint: fingerprint_bytes(canonical_cases(&subset).as_bytes()),
                fixture_count: subset.len(),
            }
        })
        .collect();

    EvaluationSuite {
        id: format!("code-rust-exec-{profile_name}"),
        version: SemVer::new(1, 0, 0),
        description: format!(
            "deterministic executable Rust code {profile_name} profile with compile/test scoring"
        ),
        dataset: DatasetMetadata {
            id: "auralis-rust-code-eval".to_string(),
            revision: "1".to_string(),
            split: profile_name.to_string(),
            population: "standalone Rust expression-patch fixtures for comprehension, compilation, unit tests and bug fixes".to_string(),
            fixture_fingerprint: fingerprint_bytes(canonical.as_bytes()),
            sample_count: cases.len(),
        },
        tasks,
        metrics: code_metrics(),
        seed_policy: SeedPolicy { seeds: vec![seed] },
        limits: ResourceLimits {
            max_examples: cases.len(),
            max_steps: 2,
            max_tokens: (limits.max_source_bytes / 4).saturating_mul(cases.len()).max(1),
            max_wall_ms: (limits.compile_timeout_ms + limits.test_timeout_ms)
                .saturating_mul(cases.len() as u64),
        },
    }
}

pub fn canonical_cases(cases: &[CodeCase]) -> String {
    cases.iter().map(CodeCase::canonical).collect()
}

pub fn classify_compile_error(stderr: &str) -> CompileErrorKind {
    if stderr.contains("E0308") {
        CompileErrorKind::TypeMismatch
    } else if ["E0382", "E0499", "E0502", "E0596"]
        .iter()
        .any(|code| stderr.contains(code))
    {
        CompileErrorKind::BorrowCheck
    } else if ["E0412", "E0425", "E0433"]
        .iter()
        .any(|code| stderr.contains(code))
    {
        CompileErrorKind::MissingItem
    } else if stderr.contains("expected")
        || stderr.contains("unexpected")
        || stderr.contains("unclosed delimiter")
    {
        CompileErrorKind::Syntax
    } else {
        CompileErrorKind::Other
    }
}

fn validate_replacement(
    replacement: &str,
    limits: CodeExecLimits,
) -> Result<(), CodeFailureKind> {
    if replacement.trim().is_empty() || replacement.len() > limits.max_patch_bytes {
        return Err(CodeFailureKind::PatchRejected);
    }
    if ["//", "/*", "*/", "unsafe", "extern", "include", "env!"]
        .iter()
        .any(|needle| replacement.contains(needle))
    {
        return Err(CodeFailureKind::SandboxViolation);
    }
    if !replacement.chars().all(|ch| {
        ch.is_ascii_alphanumeric()
            || ch == '_'
            || ch == ' '
            || ch == '\t'
            || matches!(
                ch,
                '+' | '-' | '*' | '/' | '%' | '<' | '>' | '=' | '!' | '&' | '|'
                    | '(' | ')' | '.' | ','
            )
    }) {
        return Err(CodeFailureKind::SandboxViolation);
    }
    Ok(())
}

fn create_sandbox(
    case: &CodeCase,
    source: &str,
) -> Result<(SandboxGuard, PathBuf, PathBuf, PathBuf), CodeSuiteError> {
    let sequence = NEXT_SANDBOX.fetch_add(1, Ordering::Relaxed);
    let fingerprint = fingerprint_bytes(case.canonical().as_bytes());
    let root = std::env::temp_dir().join(format!(
        "auralis-code-suite-{}-{fingerprint:016x}-{sequence}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root)
            .map_err(|e| CodeSuiteError::Harness(format!("clean sandbox: {e}")))?;
    }
    fs::create_dir_all(&root)
        .map_err(|e| CodeSuiteError::Harness(format!("create sandbox: {e}")))?;

    let source_path = sec_path::confine(&root, Path::new("main.rs"))
        .map_err(CodeSuiteError::Harness)?;
    let binary_name = format!("suite-tests{}", std::env::consts::EXE_SUFFIX);
    let binary_path =
        sec_path::confine(&root, Path::new(&binary_name)).map_err(CodeSuiteError::Harness)?;
    let stderr_path = sec_path::confine(&root, Path::new("compile.stderr"))
        .map_err(CodeSuiteError::Harness)?;
    let hardened = format!("#![forbid(unsafe_code)]\n{source}");
    fs::write(&source_path, hardened)
        .map_err(|e| CodeSuiteError::Harness(format!("write sandbox source: {e}"))?;

    Ok((
        SandboxGuard { root },
        source_path,
        binary_path,
        stderr_path,
    ))
}

fn find_rustc() -> Result<PathBuf, CodeSuiteError> {
    let executable = format!("rustc{}", std::env::consts::EXE_SUFFIX);
    if let Some(raw) = std::env::var_os("RUSTC") {
        let candidate = PathBuf::from(raw);
        if candidate.is_file() {
            return Ok(candidate);
        }
        if candidate.components().count() == 1 {
            if let Some(found) = find_on_path(&candidate) {
                return Ok(found);
            }
        }
    }
    find_on_path(Path::new(&executable)).ok_or_else(|| {
        CodeSuiteError::Harness("rustc executable not found on PATH".into())
    })
}

fn find_on_path(executable: &Path) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join(executable);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn run_with_timeout(
    command: &mut Command,
    timeout: Duration,
) -> Result<ProcessResult, CodeSuiteError> {
    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|e| CodeSuiteError::Harness(format!("spawn process: {e}")))?;
    loop {
        match child
            .try_wait()
            .map_err(|e| CodeSuiteError::Harness(format!("poll process: {e}")))?
        {
            Some(status) => {
                return Ok(ProcessResult {
                    success: status.success(),
                    timed_out: false,
                    elapsed_ns: elapsed_ns(started),
                })
            }
            None if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(ProcessResult {
                    success: false,
                    timed_out: true,
                    elapsed_ns: elapsed_ns(started),
                });
            }
            None => thread::sleep(Duration::from_millis(5)),
        }
    }
}

fn read_limited(path: &Path, max_bytes: usize) -> Result<String, CodeSuiteError> {
    let bytes = fs::read(path)
        .map_err(|e| CodeSuiteError::Harness(format!("read compiler output: {e}")))?;
    if bytes.len() > max_bytes {
        return Err(CodeSuiteError::Harness(format!(
            "compiler output exceeded limit: {} > {max_bytes}",
            bytes.len()
        )));
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128).max(1) as u64
}

fn code_metrics() -> Vec<EvaluationMetric> {
    let mut metrics = vec![
        metric("code.exact-pass", "ratio", MetricDirection::HigherIsBetter, "fraction of submissions that compile and pass all protected tests"),
        metric("code.compile-pass", "ratio", MetricDirection::HigherIsBetter, "fraction of submissions that compile"),
        metric("code.test-pass", "ratio", MetricDirection::HigherIsBetter, "fraction of submissions that compile and pass the protected test binary"),
        metric("code.compile-latency", "ns", MetricDirection::LowerIsBetter, "descriptive rustc compilation latency"),
        metric("code.test-latency", "ns", MetricDirection::LowerIsBetter, "descriptive protected-test execution latency"),
    ];
    for failure in CodeFailureKind::FAILURES {
        metrics.push(metric(
            &format!("code.failure.{}", failure.as_str()),
            "count",
            MetricDirection::LowerIsBetter,
            &format!("number of submissions classified as {}", failure.as_str()),
        ));
    }
    for error in CompileErrorKind::ALL {
        metrics.push(metric(
            &format!("code.compile-error.{}", error.as_str()),
            "count",
            MetricDirection::LowerIsBetter,
            &format!("number of compiler failures classified as {}", error.as_str()),
        ));
    }
    metrics
}

fn metric(id: &str, unit: &str, direction: MetricDirection, description: &str) -> EvaluationMetric {
    EvaluationMetric {
        id: id.to_string(),
        version: SemVer::new(1, 0, 0),
        unit: unit.to_string(),
        direction,
        description: description.to_string(),
    }
}

fn task_description(kind: CodeTaskKind) -> &'static str {
    match kind {
        CodeTaskKind::Comprehension => "derive the executable answer implied by a small Rust function",
        CodeTaskKind::Compile => "repair one declared expression so the fixture compiles and passes its protected test",
        CodeTaskKind::UnitTests => "repair an implementation expression to satisfy protected unit tests",
        CodeTaskKind::BugFix => "repair a boundary bug using an expression-only patch while preserving protected regression tests",
    }
}

fn generate_case(kind: CodeTaskKind, seed: u64, index: u64, local: usize) -> CodeCase {
    let suffix = index;
    match kind {
        CodeTaskKind::Comprehension => {
            let factor = 2 + bounded(seed, index, 0, 5) as i32;
            let offset = 1 + bounded(seed, index, 1, 7) as i32;
            let input = 3 + bounded(seed, index, 2, 9) as i32;
            let answer = input * factor + offset;
            let placeholder = 900_000_000i32 + local as i32;
            let target = placeholder.to_string();
            CodeCase {
                id: format!("comprehension:{suffix:05}"),
                kind,
                prompt: "Replace the declared ANSWER expression with the value computed by the function.".into(),
                base_source: format!(
                    "fn compute_{suffix}(x: i32) -> i32 {{ x * {factor} + {offset} }}\nconst INPUT: i32 = {input};\nconst ANSWER: i32 = {placeholder};\n#[test]\nfn comprehension_{suffix}() {{ assert_eq!(ANSWER, compute_{suffix}(INPUT)); }}\n"
                ),
                patch_target: target,
                oracle_replacement: answer.to_string(),
                expected_tests: 1,
            }
        }
        CodeTaskKind::Compile => {
            let a = 4 + bounded(seed, index, 3, 8) as i32;
            let b = 2 + bounded(seed, index, 4, 8) as i32;
            let hole = format!("COMPILE_HOLE_{suffix}");
            CodeCase {
                id: format!("compile:{suffix:05}"),
                kind,
                prompt: "Repair the declared expression so this standalone Rust fixture compiles and its protected test passes.".into(),
                base_source: format!(
                    "fn combine_{suffix}(a: i32, b: i32) -> i32 {{ a + {hole} }}\n#[test]\nfn compile_{suffix}() {{ assert_eq!(combine_{suffix}({a}, {b}), {}); }}\n",
                    a + b
                ),
                patch_target: hole,
                oracle_replacement: "b".into(),
                expected_tests: 1,
            }
        }
        CodeTaskKind::UnitTests => {
            let sentinel = 120_000 + local as i32;
            let target = format!("x - {sentinel}");
            CodeCase {
                id: format!("unit-tests:{suffix:05}"),
                kind,
                prompt: "Repair normalize so negative inputs clamp to zero and non-negative inputs are unchanged.".into(),
                base_source: format!(
                    "fn normalize_{suffix}(x: i32) -> i32 {{ {target} }}\n#[test]\nfn negative_{suffix}() {{ assert_eq!(normalize_{suffix}(-7), 0); }}\n#[test]\nfn positive_{suffix}() {{ assert_eq!(normalize_{suffix}(9), 9); }}\n"
                ),
                patch_target: target,
                oracle_replacement: "x.max(0)".into(),
                expected_tests: 2,
            }
        }
        CodeTaskKind::BugFix => CodeCase {
            id: format!("bug-fix:{suffix:05}"),
            kind,
            prompt: "Fix the index boundary predicate without changing the protected regression tests.".into(),
            base_source: format!(
                "fn valid_index_{suffix}(len: usize, idx: usize) -> bool {{ idx <= len }}\n#[test]\nfn inside_{suffix}() {{ assert!(valid_index_{suffix}(5, 4)); }}\n#[test]\nfn boundary_{suffix}() {{ assert!(!valid_index_{suffix}(5, 5)); }}\n#[test]\nfn beyond_{suffix}() {{ assert!(!valid_index_{suffix}(5, 6)); }}\n"
            ),
            patch_target: "idx <= len".into(),
            oracle_replacement: "idx < len".into(),
            expected_tests: 3,
        },
    }
}

fn bounded(seed: u64, index: u64, stream: u64, upper: usize) -> usize {
    debug_assert!(upper > 0);
    (deterministic_u64(seed, index, stream) % upper as u64) as usize
}

fn kind_index(kind: CodeTaskKind) -> usize {
    match kind {
        CodeTaskKind::Comprehension => 0,
        CodeTaskKind::Compile => 1,
        CodeTaskKind::UnitTests => 2,
        CodeTaskKind::BugFix => 3,
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
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
        for profile in [CodeSuiteProfile::Smoke, CodeSuiteProfile::Full] {
            let a = generate_suite(profile, CODE_SUITE_SEED);
            let b = generate_suite(profile, CODE_SUITE_SEED);
            assert_eq!(a, b);
            assert_eq!(a.len(), 4 * profile.cases_per_kind());
            assert_eq!(canonical_cases(&a), canonical_cases(&b));
        }
    }

    #[test]
    fn patch_validation_confines_path_target_and_expression_dialect() {
        let case = generate_suite(CodeSuiteProfile::Smoke, CODE_SUITE_SEED)
            .into_iter()
            .next()
            .unwrap();
        let limits = CodeExecLimits::for_profile(CodeSuiteProfile::Smoke);

        let mut patch = case.oracle_patch();
        patch.path = "../main.rs".into();
        assert_eq!(
            apply_patch(&case, &patch, limits),
            Err(CodeFailureKind::PatchRejected)
        );

        let mut patch = case.oracle_patch();
        patch.expected_old = "not-the-target".into();
        assert_eq!(
            apply_patch(&case, &patch, limits),
            Err(CodeFailureKind::PatchRejected)
        );

        let mut patch = case.oracle_patch();
        patch.replacement = "0; #[test] fn cheat() {}".into();
        assert_eq!(
            apply_patch(&case, &patch, limits),
            Err(CodeFailureKind::SandboxViolation)
        );

        let applied = apply_patch(&case, &case.oracle_patch(), limits).unwrap();
        assert!(applied.contains(&case.oracle_replacement));
        assert!(applied.contains("#[test]"));
    }

    #[test]
    fn executable_compile_and_test_scoring_is_objective() {
        let cases = generate_suite(CodeSuiteProfile::Smoke, CODE_SUITE_SEED);
        let limits = CodeExecLimits::for_profile(CodeSuiteProfile::Smoke);

        let compile_case = cases
            .iter()
            .find(|case| case.kind == CodeTaskKind::Compile)
            .unwrap();
        let pass = evaluate_submission(compile_case, &compile_case.oracle_patch(), limits).unwrap();
        assert!(pass.exact());
        assert!(pass.compile_passed && pass.test_passed);

        let bad_compile = intentional_regression_submissions(std::slice::from_ref(compile_case));
        let fail = evaluate_submission(compile_case, &bad_compile[0], limits).unwrap();
        assert_eq!(fail.failure, CodeFailureKind::CompileError);
        assert_eq!(fail.compile_error, Some(CompileErrorKind::MissingItem));

        let bug_case = cases
            .iter()
            .find(|case| case.kind == CodeTaskKind::BugFix)
            .unwrap();
        let bad_bug = intentional_regression_submissions(std::slice::from_ref(bug_case));
        let fail = evaluate_submission(bug_case, &bad_bug[0], limits).unwrap();
        assert_eq!(fail.failure, CodeFailureKind::TestFailure);
        assert!(fail.compile_passed);
        assert!(!fail.test_passed);
    }

    #[test]
    fn compile_error_taxonomy_is_stable() {
        assert_eq!(classify_compile_error("error[E0308]: mismatched types"), CompileErrorKind::TypeMismatch);
        assert_eq!(classify_compile_error("error[E0502]: cannot borrow"), CompileErrorKind::BorrowCheck);
        assert_eq!(classify_compile_error("error[E0425]: cannot find value"), CompileErrorKind::MissingItem);
        assert_eq!(classify_compile_error("error: expected expression"), CompileErrorKind::Syntax);
        assert_eq!(classify_compile_error("some future compiler diagnostic"), CompileErrorKind::Other);
    }

    #[test]
    fn registry_identity_and_resource_limits_are_stable() {
        let mut registry = EvaluationRegistry::new();
        let smoke = suite_definition(CodeSuiteProfile::Smoke, CODE_SUITE_SEED);
        let full = suite_definition(CodeSuiteProfile::Full, CODE_SUITE_SEED);
        smoke.validate().unwrap();
        full.validate().unwrap();
        assert_eq!(smoke.metrics.len(), 16);
        assert_eq!(full.metrics.len(), 16);
        assert_eq!(smoke.limits.max_steps, 2);
        assert!(smoke.limits.max_wall_ms > 0);
        assert!(full.limits.max_wall_ms > smoke.limits.max_wall_ms);
        let smoke_ref = registry.register_suite(smoke.clone()).unwrap();
        let full_ref = registry.register_suite(full.clone()).unwrap();
        assert_ne!(smoke_ref, full_ref);
        assert!(!registry.exact_compatible(&smoke_ref, &full_ref).unwrap());
        assert_eq!(
            smoke.definition_fingerprint().unwrap(),
            suite_definition(CodeSuiteProfile::Smoke, CODE_SUITE_SEED)
                .definition_fingerprint()
                .unwrap()
        );
    }

    #[test]
    fn submission_count_mismatch_fails_closed() {
        let cases = generate_suite(CodeSuiteProfile::Smoke, CODE_SUITE_SEED);
        assert!(matches!(
            evaluate_submissions(
                &cases,
                &[],
                CodeExecLimits::for_profile(CodeSuiteProfile::Smoke)
            ),
            Err(CodeSuiteError::PredictionCountMismatch { .. })
        ));
    }
}
