//! Numerical diagnostics over training state slices.
//!
//! This module deliberately stays independent from `model` and `training` so
//! callers can opt in without changing engine semantics. When diagnostics are
//! disabled the entry point returns before scanning any slice.

use crate::numeric::{Diagnostics, Fault, Scan, Stage};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientSummary {
    pub len: usize,
    pub l2: Option<f32>,
    pub max_abs: f32,
    pub finite: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateSliceSummary {
    pub stage: Stage,
    pub name: &'static str,
    pub scan: Scan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingStateSummary {
    pub gradient: GradientSummary,
    pub slices: Vec<StateSliceSummary>,
}

impl TrainingStateSummary {
    pub fn first_fault(&self) -> Option<StateSliceSummary> {
        self.slices.iter().copied().find(|entry| !entry.scan.is_finite())
    }
}

pub fn summarize_training_state(
    diagnostics: Diagnostics,
    gradients: &[f32],
    parameters: &[f32],
    adam_m: &[f32],
    adam_v: &[f32],
) -> Option<TrainingStateSummary> {
    if !diagnostics.enabled {
        return None;
    }

    let grad_scan = diagnostics.scan(gradients).expect("enabled diagnostics");
    let parameter_scan = diagnostics.scan(parameters).expect("enabled diagnostics");
    let adam_m_scan = diagnostics.scan(adam_m).expect("enabled diagnostics");
    let adam_v_scan = diagnostics.scan(adam_v).expect("enabled diagnostics");

    let gradient_l2 = if grad_scan.is_finite() {
        let mut acc = 0.0f32;
        for &value in gradients {
            acc += value * value;
        }
        Some(acc.sqrt())
    } else {
        None
    };

    let gradient = GradientSummary {
        len: gradients.len(),
        l2: gradient_l2,
        max_abs: grad_scan.max_abs,
        finite: grad_scan.is_finite(),
    };

    Some(TrainingStateSummary {
        gradient,
        slices: vec![
            StateSliceSummary {
                stage: Stage::Gradient,
                name: "gradients",
                scan: grad_scan,
            },
            StateSliceSummary {
                stage: Stage::Parameter,
                name: "parameters",
                scan: parameter_scan,
            },
            StateSliceSummary {
                stage: Stage::AdamMoment,
                name: "adam_m",
                scan: adam_m_scan,
            },
            StateSliceSummary {
                stage: Stage::AdamMoment,
                name: "adam_v",
                scan: adam_v_scan,
            },
        ],
    })
}

pub fn fault_context(summary: &TrainingStateSummary) -> Option<String> {
    let hit = summary.first_fault()?;
    let detail = match hit.scan.first {
        Some(Fault::NaN { index }) => format!("NaN at index {index}"),
        Some(Fault::Infinity { index, negative }) => format!(
            "{}Inf at index {index}",
            if negative { "-" } else { "+" }
        ),
        None => return None,
    };
    Some(format!(
        "{} ({:?}): {detail}; nans={} infs={} len={}",
        hit.name, hit.stage, hit.scan.nans, hit.scan.infs, hit.scan.len
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_off_returns_before_state_scan() {
        let dirty = [f32::NAN];
        assert!(summarize_training_state(
            Diagnostics::off(),
            &dirty,
            &dirty,
            &dirty,
            &dirty,
        )
        .is_none());
    }

    #[test]
    fn finite_state_reports_gradient_norm() {
        let summary = summarize_training_state(
            Diagnostics::on(),
            &[3.0, 4.0],
            &[1.0, -2.0],
            &[0.1, 0.2],
            &[0.01, 0.02],
        )
        .unwrap();
        assert_eq!(summary.gradient.len, 2);
        assert_eq!(summary.gradient.l2, Some(5.0));
        assert_eq!(summary.gradient.max_abs, 4.0);
        assert!(summary.gradient.finite);
        assert!(summary.first_fault().is_none());
        assert!(fault_context(&summary).is_none());
    }

    #[test]
    fn gradient_corruption_is_reported_first() {
        let summary = summarize_training_state(
            Diagnostics::on(),
            &[1.0, f32::NAN],
            &[f32::INFINITY],
            &[0.0],
            &[0.0],
        )
        .unwrap();
        let hit = summary.first_fault().unwrap();
        assert_eq!(hit.stage, Stage::Gradient);
        assert_eq!(hit.name, "gradients");
        assert_eq!(hit.scan.first, Some(Fault::NaN { index: 1 }));
        assert_eq!(summary.gradient.l2, None);
        assert!(!summary.gradient.finite);
        assert!(fault_context(&summary).unwrap().contains("NaN at index 1"));
    }

    #[test]
    fn parameter_corruption_precedes_adam_faults() {
        let summary = summarize_training_state(
            Diagnostics::on(),
            &[1.0],
            &[0.0, f32::NEG_INFINITY],
            &[f32::NAN],
            &[f32::INFINITY],
        )
        .unwrap();
        let hit = summary.first_fault().unwrap();
        assert_eq!(hit.stage, Stage::Parameter);
        assert_eq!(hit.name, "parameters");
        assert!(fault_context(&summary).unwrap().contains("-Inf at index 1"));
    }

    #[test]
    fn adam_m_precedes_adam_v_when_parameters_are_clean() {
        let summary = summarize_training_state(
            Diagnostics::on(),
            &[1.0],
            &[2.0],
            &[0.0, f32::NAN],
            &[0.0, f32::NEG_INFINITY],
        )
        .unwrap();
        let hit = summary.first_fault().unwrap();
        assert_eq!(hit.stage, Stage::AdamMoment);
        assert_eq!(hit.name, "adam_m");
    }
}
