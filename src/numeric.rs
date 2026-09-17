//! Optional numerical diagnostics for Engine.
//!
//! Scanning is explicit and off the production hot path until a caller opts in.
//! Mode-off therefore adds no overhead: nothing in `model`/`training` calls this
//! module yet (wiring into forward is a later slice of #90 so it does not collide with #27).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Activation,
    Logits,
    Loss,
    Gradient,
    Parameter,
    AdamMoment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    NaN { index: usize },
    Infinity { index: usize, negative: bool },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scan {
    pub len: usize,
    pub nans: usize,
    pub infs: usize,
    pub first: Option<Fault>,
    pub max_abs: f32,
}

impl Scan {
    pub fn is_finite(&self) -> bool {
        self.first.is_none()
    }
}

/// Opt-in gate. Default off; `scan` is a no-op so callers can sit on the hot path later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Diagnostics {
    pub enabled: bool,
}

impl Diagnostics {
    pub fn off() -> Self {
        Self { enabled: false }
    }

    pub fn on() -> Self {
        Self { enabled: true }
    }

    pub fn scan(&self, values: &[f32]) -> Option<Scan> {
        if !self.enabled {
            return None;
        }
        Some(scan_f32(values))
    }
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self::off()
    }
}

pub fn scan_f32(values: &[f32]) -> Scan {
    let mut scan = Scan {
        len: values.len(),
        nans: 0,
        infs: 0,
        first: None,
        max_abs: 0.0,
    };
    for (index, &value) in values.iter().enumerate() {
        if value.is_nan() {
            scan.nans += 1;
            if scan.first.is_none() {
                scan.first = Some(Fault::NaN { index });
            }
            continue;
        }
        if value.is_infinite() {
            scan.infs += 1;
            if scan.first.is_none() {
                scan.first = Some(Fault::Infinity {
                    index,
                    negative: value.is_sign_negative(),
                });
            }
            continue;
        }
        scan.max_abs = scan.max_abs.max(value.abs());
    }
    scan
}

pub fn l2_norm(values: &[f32]) -> Option<f32> {
    let scan = scan_f32(values);
    if !scan.is_finite() {
        return None;
    }
    let mut acc = 0.0f32;
    for &value in values {
        acc += value * value;
    }
    Some(acc.sqrt())
}

pub fn explain(stage: Stage, name: &str, scan: &Scan) -> String {
    match scan.first {
        None => format!("{name} ({stage:?}): finite n={} max_abs={:.6}", scan.len, scan.max_abs),
        Some(Fault::NaN { index }) => format!(
            "{name} ({stage:?}): first NaN at index {index} (nans={} infs={} n={})",
            scan.nans, scan.infs, scan.len
        ),
        Some(Fault::Infinity { index, negative }) => format!(
            "{name} ({stage:?}): first {}Inf at index {index} (nans={} infs={} n={})",
            if negative { "-" } else { "+" },
            scan.nans,
            scan.infs,
            scan.len
        ),
    }
}

pub fn first_corrupt_stage<'a, I>(stages: I) -> Option<(Stage, &'a str, Scan)>
where
    I: IntoIterator<Item = (Stage, &'a str, &'a [f32])>,
{
    for (stage, name, values) in stages {
        let scan = scan_f32(values);
        if !scan.is_finite() {
            return Some((stage, name, scan));
        }
    }
    None
}

/// Parse a whitespace list. Tokens: number, `nan`, `inf`, `+inf`, `-inf`.
pub fn parse_tokens(text: &str) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.split_whitespace().enumerate() {
        let tok = raw.trim();
        let value = match tok.to_ascii_lowercase().as_str() {
            "nan" => f32::NAN,
            "inf" | "+inf" | "infinity" | "+infinity" => f32::INFINITY,
            "-inf" | "-infinity" => f32::NEG_INFINITY,
            other => other
                .parse::<f32>()
                .map_err(|_| format!("invalid f32 token {i}: {tok}"))?,
        };
        out.push(value);
    }
    if out.is_empty() {
        return Err("no f32 tokens".into());
    }
    Ok(out)
}

/// Fixture: activations clean, logits contain a NaN, grads clean.
pub fn nan_at_logits_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        vec![0.1, 0.2],
        vec![0.0, f32::NAN],
        vec![1.0, 1.0],
    )
}

/// Fixture: first fault is -Inf on a gradient slice.
pub fn neg_inf_at_gradient_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (vec![0.5], vec![0.25], vec![f32::NEG_INFINITY, 0.0])
}

pub fn report_first_corrupt(
    stages: &[(Stage, &str, &[f32])],
) -> Result<String, String> {
    match first_corrupt_stage(stages.iter().copied()) {
        Some((stage, name, scan)) => Ok(explain(stage, name, &scan)),
        None => Err("pipeline is finite".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_slice_is_clean() {
        let scan = scan_f32(&[0.0, -1.5, 2.0]);
        assert!(scan.is_finite());
        assert_eq!(scan.max_abs, 2.0);
        assert_eq!(l2_norm(&[3.0, 4.0]), Some(5.0));
    }

    #[test]
    fn nan_is_the_first_fault() {
        let scan = scan_f32(&[1.0, f32::NAN, f32::INFINITY]);
        assert_eq!(scan.first, Some(Fault::NaN { index: 1 }));
        assert_eq!(scan.nans, 1);
        assert_eq!(scan.infs, 1);
        assert!(l2_norm(&[1.0, f32::NAN]).is_none());
    }

    #[test]
    fn signed_infinity_is_reported() {
        let scan = scan_f32(&[1.0, f32::NEG_INFINITY]);
        assert_eq!(
            scan.first,
            Some(Fault::Infinity {
                index: 1,
                negative: true
            })
        );
    }

    #[test]
    fn first_corrupt_stage_stops_at_logits() {
        let acts = [0.1f32, 0.2];
        let logits = [0.0f32, f32::NAN];
        let grads = [1.0f32, 1.0];
        let hit = first_corrupt_stage([
            (Stage::Activation, "h", acts.as_slice()),
            (Stage::Logits, "logits", logits.as_slice()),
            (Stage::Gradient, "dW", grads.as_slice()),
        ])
        .expect("logits should trip");
        assert_eq!(hit.0, Stage::Logits);
        assert_eq!(hit.1, "logits");
        assert!(explain(hit.0, hit.1, &hit.2).contains("NaN"));
    }

    #[test]
    fn clean_pipeline_returns_none() {
        let none = first_corrupt_stage([
            (Stage::Loss, "loss", [1.2f32].as_slice()),
            (Stage::AdamMoment, "m", [0.01f32, 0.0].as_slice()),
        ]);
        assert!(none.is_none());
    }

    #[test]
    fn diagnostics_off_is_a_noop() {
        let dirty = [1.0f32, f32::NAN];
        assert!(Diagnostics::off().scan(&dirty).is_none());
        assert!(Diagnostics::default().scan(&dirty).is_none());
    }

    #[test]
    fn diagnostics_on_reports_first_fault() {
        let dirty = [1.0f32, f32::NAN];
        let scan = Diagnostics::on().scan(&dirty).expect("enabled");
        assert_eq!(scan.first, Some(Fault::NaN { index: 1 }));
    }

    #[test]
    fn parse_tokens_accepts_sentinels() {
        let vals = parse_tokens("1.0 nan inf -inf").unwrap();
        assert_eq!(vals.len(), 4);
        assert!(vals[1].is_nan());
        assert!(vals[2].is_infinite() && vals[2].is_sign_positive());
        assert!(vals[3].is_infinite() && vals[3].is_sign_negative());
    }

    #[test]
    fn parse_tokens_rejects_junk() {
        assert!(parse_tokens("1.0 nope").is_err());
        assert!(parse_tokens("   ").is_err());
    }

    #[test]
    fn named_nan_fixture_trips_logits() {
        let (acts, logits, grads) = nan_at_logits_fixture();
        let report = report_first_corrupt(&[
            (Stage::Activation, "h", acts.as_slice()),
            (Stage::Logits, "logits", logits.as_slice()),
            (Stage::Gradient, "dW", grads.as_slice()),
        ])
        .unwrap();
        assert!(report.contains("logits"));
        assert!(report.contains("NaN"));
    }

    #[test]
    fn named_neg_inf_fixture_trips_gradient() {
        let (acts, logits, grads) = neg_inf_at_gradient_fixture();
        let report = report_first_corrupt(&[
            (Stage::Activation, "h", acts.as_slice()),
            (Stage::Logits, "logits", logits.as_slice()),
            (Stage::Gradient, "dW", grads.as_slice()),
        ])
        .unwrap();
        assert!(report.contains("dW"));
        assert!(report.contains("-Inf"));
    }
}
