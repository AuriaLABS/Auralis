//! Machine-readable benchmark result contract.
//!
//! Benchmarks keep raw timings instead of publishing only a summary. The
//! summary is derived deterministically so another agent can recompute it from
//! the same record and compare reference/candidate runs without hidden state.

#[derive(Clone, Debug, PartialEq)]
pub struct BenchSummary {
    pub median_ns: f64,
    pub q1_ns: f64,
    pub q3_ns: f64,
    pub iqr_ns: f64,
    pub min_ns: u64,
    pub max_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchRunRecord {
    pub benchmark_id: String,
    pub backend: String,
    pub hardware: String,
    pub toolchain: String,
    pub config_fingerprint: u64,
    pub warmup_iterations: usize,
    pub repetitions: usize,
    pub raw_ns: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchComparison {
    pub benchmark_id: String,
    pub reference_backend: String,
    pub candidate_backend: String,
    pub reference_median_ns: f64,
    pub candidate_median_ns: f64,
    pub candidate_over_reference: f64,
    pub speedup: f64,
}

impl BenchRunRecord {
    pub fn validate(&self) -> Result<(), String> {
        if self.benchmark_id.trim().is_empty() {
            return Err("benchmark_id must not be empty".into());
        }
        if self.backend.trim().is_empty() {
            return Err("backend must not be empty".into());
        }
        if self.hardware.trim().is_empty() {
            return Err("hardware must not be empty".into());
        }
        if self.toolchain.trim().is_empty() {
            return Err("toolchain must not be empty".into());
        }
        if self.repetitions == 0 || self.raw_ns.is_empty() {
            return Err("benchmark requires at least one measured repetition".into());
        }
        if self.repetitions != self.raw_ns.len() {
            return Err(format!(
                "repetitions/raw timing mismatch: repetitions={} raw={}",
                self.repetitions,
                self.raw_ns.len()
            ));
        }
        if self.raw_ns.iter().any(|&ns| ns == 0) {
            return Err("raw timings must be positive nanoseconds".into());
        }
        Ok(())
    }

    pub fn summary(&self) -> Result<BenchSummary, String> {
        self.validate()?;
        let mut sorted = self.raw_ns.clone();
        sorted.sort_unstable();
        let median_ns = median(&sorted);
        let (lower, upper) = halves(&sorted);
        let q1_ns = median(lower);
        let q3_ns = median(upper);
        Ok(BenchSummary {
            median_ns,
            q1_ns,
            q3_ns,
            iqr_ns: q3_ns - q1_ns,
            min_ns: sorted[0],
            max_ns: *sorted.last().unwrap(),
        })
    }

    pub fn json(&self) -> Result<String, String> {
        let s = self.summary()?;
        let raw = self
            .raw_ns
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        Ok(format!(
            concat!(
                "{{\"benchmark_id\":\"{}\",\"backend\":\"{}\",",
                "\"hardware\":\"{}\",\"toolchain\":\"{}\",",
                "\"config_fingerprint\":\"{:016x}\",",
                "\"warmup_iterations\":{},\"repetitions\":{},",
                "\"unit\":\"ns\",\"raw_ns\":[{}],",
                "\"median_ns\":{},\"q1_ns\":{},\"q3_ns\":{},",
                "\"iqr_ns\":{},\"min_ns\":{},\"max_ns\":{}}}"
            ),
            json_escape(&self.benchmark_id),
            json_escape(&self.backend),
            json_escape(&self.hardware),
            json_escape(&self.toolchain),
            self.config_fingerprint,
            self.warmup_iterations,
            self.repetitions,
            raw,
            f64_json(s.median_ns),
            f64_json(s.q1_ns),
            f64_json(s.q3_ns),
            f64_json(s.iqr_ns),
            s.min_ns,
            s.max_ns,
        ))
    }

    pub fn csv_header() -> &'static str {
        "benchmark_id,backend,hardware,toolchain,config_fingerprint,warmup_iterations,repetitions,unit,raw_ns,median_ns,q1_ns,q3_ns,iqr_ns,min_ns,max_ns\n"
    }

    pub fn csv_row(&self) -> Result<String, String> {
        let s = self.summary()?;
        let raw = self
            .raw_ns
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(";");
        Ok(format!(
            "{},{},{},{},{:016x},{},{},ns,{},{},{},{},{},{},{}\n",
            csv_escape(&self.benchmark_id),
            csv_escape(&self.backend),
            csv_escape(&self.hardware),
            csv_escape(&self.toolchain),
            self.config_fingerprint,
            self.warmup_iterations,
            self.repetitions,
            csv_escape(&raw),
            f64_json(s.median_ns),
            f64_json(s.q1_ns),
            f64_json(s.q3_ns),
            f64_json(s.iqr_ns),
            s.min_ns,
            s.max_ns,
        ))
    }
}

impl BenchComparison {
    pub fn compare(reference: &BenchRunRecord, candidate: &BenchRunRecord) -> Result<Self, String> {
        reference.validate()?;
        candidate.validate()?;
        if reference.benchmark_id != candidate.benchmark_id {
            return Err("cannot compare different benchmark ids".into());
        }
        if reference.config_fingerprint != candidate.config_fingerprint {
            return Err("cannot compare different config fingerprints".into());
        }
        if reference.hardware != candidate.hardware {
            return Err("cannot compare different hardware".into());
        }
        if reference.toolchain != candidate.toolchain {
            return Err("cannot compare different toolchains".into());
        }
        if reference.warmup_iterations != candidate.warmup_iterations {
            return Err("cannot compare different warmup iteration counts".into());
        }
        if reference.repetitions != candidate.repetitions {
            return Err("cannot compare different repetition counts".into());
        }
        let reference_median_ns = reference.summary()?.median_ns;
        let candidate_median_ns = candidate.summary()?.median_ns;
        let candidate_over_reference = candidate_median_ns / reference_median_ns;
        Ok(Self {
            benchmark_id: reference.benchmark_id.clone(),
            reference_backend: reference.backend.clone(),
            candidate_backend: candidate.backend.clone(),
            reference_median_ns,
            candidate_median_ns,
            candidate_over_reference,
            speedup: reference_median_ns / candidate_median_ns,
        })
    }

    pub fn json(&self) -> String {
        format!(
            concat!(
                "{{\"benchmark_id\":\"{}\",\"reference_backend\":\"{}\",",
                "\"candidate_backend\":\"{}\",\"reference_median_ns\":{},",
                "\"candidate_median_ns\":{},\"candidate_over_reference\":{},",
                "\"speedup\":{}}}"
            ),
            json_escape(&self.benchmark_id),
            json_escape(&self.reference_backend),
            json_escape(&self.candidate_backend),
            f64_json(self.reference_median_ns),
            f64_json(self.candidate_median_ns),
            f64_json(self.candidate_over_reference),
            f64_json(self.speedup),
        )
    }
}

fn halves(sorted: &[u64]) -> (&[u64], &[u64]) {
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (&sorted[..mid], &sorted[mid..])
    } else if sorted.len() == 1 {
        (sorted, sorted)
    } else {
        (&sorted[..mid], &sorted[mid + 1..])
    }
}

fn median(sorted: &[u64]) -> f64 {
    debug_assert!(!sorted.is_empty());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
    } else {
        sorted[mid] as f64
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c <= '\u{001f}' => {
                out.push_str("\\u");
                let value = c as u32;
                const HEX: &[u8; 16] = b"0123456789abcdef";
                for shift in [12, 8, 4, 0] {
                    out.push(HEX[((value >> shift) & 0xf) as usize] as char);
                }
            }
            _ => out.push(c),
        }
    }
    out
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn f64_json(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.9}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(backend: &str, raw_ns: &[u64]) -> BenchRunRecord {
        BenchRunRecord {
            benchmark_id: "matmul/tiny".into(),
            backend: backend.into(),
            hardware: "test-cpu".into(),
            toolchain: "rustc-test".into(),
            config_fingerprint: 0x1234,
            warmup_iterations: 3,
            repetitions: raw_ns.len(),
            raw_ns: raw_ns.to_vec(),
        }
    }

    #[test]
    fn summary_preserves_raw_data_and_is_deterministic() {
        let r = run("reference", &[30, 10, 50, 20, 40]);
        let original = r.raw_ns.clone();
        let s = r.summary().unwrap();
        assert_eq!(r.raw_ns, original);
        assert_eq!(s.median_ns, 30.0);
        assert_eq!(s.q1_ns, 15.0);
        assert_eq!(s.q3_ns, 45.0);
        assert_eq!(s.iqr_ns, 30.0);
        assert_eq!(s.min_ns, 10);
        assert_eq!(s.max_ns, 50);
        assert_eq!(r.summary().unwrap(), s);
    }

    #[test]
    fn even_sample_median_and_iqr_are_explicit() {
        let s = run("reference", &[40, 10, 30, 20]).summary().unwrap();
        assert_eq!(s.median_ns, 25.0);
        assert_eq!(s.q1_ns, 15.0);
        assert_eq!(s.q3_ns, 35.0);
        assert_eq!(s.iqr_ns, 20.0);
    }

    #[test]
    fn single_sample_is_supported() {
        let s = run("reference", &[17]).summary().unwrap();
        assert_eq!(s.median_ns, 17.0);
        assert_eq!(s.q1_ns, 17.0);
        assert_eq!(s.q3_ns, 17.0);
        assert_eq!(s.iqr_ns, 0.0);
    }

    #[test]
    fn invalid_records_fail_closed() {
        let mut r = run("reference", &[10, 20]);
        r.repetitions = 3;
        assert!(r.validate().is_err());
        r.repetitions = 2;
        r.raw_ns[0] = 0;
        assert!(r.validate().is_err());

        let mut r = run("reference", &[10, 20]);
        r.hardware.clear();
        assert_eq!(r.validate().unwrap_err(), "hardware must not be empty");

        let mut r = run("reference", &[10, 20]);
        r.toolchain = "   ".into();
        assert_eq!(r.validate().unwrap_err(), "toolchain must not be empty");
    }

    #[test]
    fn compare_requires_same_benchmark_and_config() {
        let reference = run("reference", &[100, 110, 90]);
        let candidate = run("candidate", &[50, 55, 45]);
        let c = BenchComparison::compare(&reference, &candidate).unwrap();
        assert_eq!(c.reference_median_ns, 100.0);
        assert_eq!(c.candidate_median_ns, 50.0);
        assert_eq!(c.candidate_over_reference, 0.5);
        assert_eq!(c.speedup, 2.0);

        let mut wrong = candidate.clone();
        wrong.config_fingerprint += 1;
        assert!(BenchComparison::compare(&reference, &wrong).is_err());
        wrong = candidate.clone();
        wrong.benchmark_id = "other".into();
        assert!(BenchComparison::compare(&reference, &wrong).is_err());
    }

    #[test]
    fn json_and_csv_keep_raw_timings_and_metadata() {
        let r = run("reference", &[30, 10, 20]);
        let json = r.json().unwrap();
        assert!(json.contains("\"raw_ns\":[30,10,20]"));
        assert!(json.contains("\"config_fingerprint\":\"0000000000001234\""));
        let csv = r.csv_row().unwrap();
        assert!(csv.contains("30;10;20"));
        assert!(csv.contains("0000000000001234"));
        assert!(BenchRunRecord::csv_header().contains("median_ns"));
    }

    #[test]
    fn json_escapes_all_ascii_control_characters() {
        let mut r = run("reference", &[10, 20, 30]);
        r.hardware = "cpu\u{0001}node".into();
        let json = r.json().unwrap();
        assert!(json.contains("cpu\\u0001node"));
        assert!(!json.contains('\u{0001}'));
    }

    #[test]
    fn csv_quotes_carriage_returns() {
        let mut r = run("reference", &[10, 20, 30]);
        r.hardware = "cpu\rnode".into();
        let csv = r.csv_row().unwrap();
        assert!(csv.contains("\"cpu\rnode\""));
    }

    #[test]
    fn compare_rejects_cross_environment_or_protocol_runs() {
        let reference = run("reference", &[100, 110, 90]);
        let mut candidate = run("candidate", &[80, 85, 75]);

        candidate.hardware = "other-cpu".into();
        assert_eq!(
            BenchComparison::compare(&reference, &candidate).unwrap_err(),
            "cannot compare different hardware"
        );

        candidate = run("candidate", &[80, 85, 75]);
        candidate.toolchain = "other-rustc".into();
        assert_eq!(
            BenchComparison::compare(&reference, &candidate).unwrap_err(),
            "cannot compare different toolchains"
        );

        candidate = run("candidate", &[80, 85, 75]);
        candidate.warmup_iterations += 1;
        assert_eq!(
            BenchComparison::compare(&reference, &candidate).unwrap_err(),
            "cannot compare different warmup iteration counts"
        );

        candidate = run("candidate", &[80, 85, 75, 82]);
        assert_eq!(
            BenchComparison::compare(&reference, &candidate).unwrap_err(),
            "cannot compare different repetition counts"
        );
    }

    #[test]
    fn comparison_json_is_machine_readable_shape() {
        let c = BenchComparison::compare(
            &run("reference", &[100, 100, 100]),
            &run("candidate", &[80, 80, 80]),
        )
        .unwrap();
        let json = c.json();
        assert!(json.contains("\"candidate_over_reference\":0.800000000"));
        assert!(json.contains("\"speedup\":1.250000000"));
    }
}
