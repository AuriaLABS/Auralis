//! Reusable numerical equivalence helpers for Engine reference-vs-candidate checks.
//!
//! Correctness checks intentionally live outside candidate kernels so an
//! optimization does not get to define its own acceptance criteria.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    pub abs: f32,
    pub rel: f32,
}

impl Tolerance {
    pub const EXACT: Self = Self { abs: 0.0, rel: 0.0 };

    pub const fn new(abs: f32, rel: f32) -> Self {
        Self { abs, rel }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub len: usize,
    pub mismatches: usize,
    pub first_mismatch: Option<usize>,
    pub max_abs_error: f32,
    pub max_rel_error: f32,
}

impl Comparison {
    pub fn is_equivalent(&self) -> bool {
        self.mismatches == 0
    }
}

/// Compare two same-shaped `f32` slices using an explicit absolute/relative
/// tolerance contract.
///
/// A value passes when `abs_error <= abs + rel * max(|reference|, 1e-8)`.
/// NaN never compares equivalent. Infinities compare equivalent only when they
/// are exactly equal (same sign).
pub fn compare_f32_slices(
    reference: &[f32],
    candidate: &[f32],
    tolerance: Tolerance,
) -> Comparison {
    assert_eq!(reference.len(), candidate.len(), "equivalence shape mismatch");
    assert!(tolerance.abs >= 0.0, "absolute tolerance must be non-negative");
    assert!(tolerance.rel >= 0.0, "relative tolerance must be non-negative");

    let mut result = Comparison {
        len: reference.len(),
        mismatches: 0,
        first_mismatch: None,
        max_abs_error: 0.0,
        max_rel_error: 0.0,
    };

    for (index, (&expected, &actual)) in reference.iter().zip(candidate).enumerate() {
        let equivalent = if expected.is_nan() || actual.is_nan() {
            false
        } else if expected.is_infinite() || actual.is_infinite() {
            expected == actual
        } else {
            let abs_error = (actual - expected).abs();
            let denom = expected.abs().max(1e-8);
            let rel_error = abs_error / denom;
            result.max_abs_error = result.max_abs_error.max(abs_error);
            result.max_rel_error = result.max_rel_error.max(rel_error);
            abs_error <= tolerance.abs + tolerance.rel * denom
        };

        if !equivalent {
            result.mismatches += 1;
            if result.first_mismatch.is_none() {
                result.first_mismatch = Some(index);
            }
        }
    }

    result
}

pub fn assert_f32_slices_equivalent(
    label: &str,
    reference: &[f32],
    candidate: &[f32],
    tolerance: Tolerance,
) {
    let comparison = compare_f32_slices(reference, candidate, tolerance);
    assert!(
        comparison.is_equivalent(),
        "{label}: {} / {} values differ; first mismatch={:?}; max_abs_error={:.9e}; max_rel_error={:.9e}; tolerance=(abs={:.9e}, rel={:.9e})",
        comparison.mismatches,
        comparison.len,
        comparison.first_mismatch,
        comparison.max_abs_error,
        comparison.max_rel_error,
        tolerance.abs,
        tolerance.rel,
    );
}

#[cfg(test)]
mod tests {
    use super::{compare_f32_slices, Tolerance};

    #[test]
    fn exact_mode_accepts_identical_values() {
        let values = [0.0, -0.0, 1.25, -7.0, f32::INFINITY, f32::NEG_INFINITY];
        let report = compare_f32_slices(&values, &values, Tolerance::EXACT);
        assert!(report.is_equivalent());
        assert_eq!(report.mismatches, 0);
    }

    #[test]
    fn exact_mode_detects_artificial_perturbation() {
        let reference = [1.0, 2.0, 3.0, 4.0];
        let candidate = [1.0, 2.0, 3.000_001, 4.0];
        let report = compare_f32_slices(&reference, &candidate, Tolerance::EXACT);
        assert_eq!(report.mismatches, 1);
        assert_eq!(report.first_mismatch, Some(2));
        assert!(report.max_abs_error > 0.0);
    }

    #[test]
    fn tolerance_accepts_small_rounding_difference() {
        let reference = [100.0, 0.001, -2.0];
        let candidate = [100.000_5, 0.001_000_5, -2.000_001];
        let report = compare_f32_slices(
            &reference,
            &candidate,
            Tolerance::new(1e-6, 1e-5),
        );
        assert!(report.is_equivalent(), "{report:?}");
    }

    #[test]
    fn tolerance_rejects_material_difference() {
        let reference = [1.0, 2.0, 3.0];
        let candidate = [1.0, 2.1, 3.0];
        let report = compare_f32_slices(
            &reference,
            &candidate,
            Tolerance::new(1e-6, 1e-5),
        );
        assert_eq!(report.mismatches, 1);
        assert_eq!(report.first_mismatch, Some(1));
    }

    #[test]
    fn nan_is_never_silently_accepted() {
        let reference = [1.0, f32::NAN];
        let candidate = [1.0, f32::NAN];
        let report = compare_f32_slices(&reference, &candidate, Tolerance::new(1.0, 1.0));
        assert_eq!(report.mismatches, 1);
        assert_eq!(report.first_mismatch, Some(1));
    }

    #[test]
    fn infinity_requires_exact_sign() {
        let reference = [f32::INFINITY, f32::NEG_INFINITY];
        let candidate = [f32::INFINITY, f32::INFINITY];
        let report = compare_f32_slices(&reference, &candidate, Tolerance::new(1.0, 1.0));
        assert_eq!(report.mismatches, 1);
        assert_eq!(report.first_mismatch, Some(1));
    }
}
