//! Optional numerical diagnostics for Engine.
//!
//! Scanning is explicit and off the production hot path until a caller opts in.
//! Mode-off therefore adds no overhead: nothing in `model`/`training` calls this
//! module yet (wiring is a later slice of #90 so it does not collide with #27).

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
}
