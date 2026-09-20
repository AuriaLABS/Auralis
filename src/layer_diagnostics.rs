//! Opt-in per-tensor statistics for Brain layer instrumentation.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactHistogram {
    pub negative_large: usize,
    pub negative_small: usize,
    pub near_zero: usize,
    pub positive_small: usize,
    pub positive_large: usize,
    pub non_finite: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorStats {
    pub len: usize,
    pub mean: Option<f32>,
    pub l2: Option<f32>,
    pub max_abs: f32,
    pub histogram: Option<CompactHistogram>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerHooks {
    pub enabled: bool,
    pub histogram: bool,
    pub adjacent_cosine: bool,
}

impl LayerHooks {
    pub const fn off() -> Self {
        Self {
            enabled: false,
            histogram: false,
            adjacent_cosine: false,
        }
    }

    pub const fn compact() -> Self {
        Self {
            enabled: true,
            histogram: false,
            adjacent_cosine: false,
        }
    }

    pub const fn full() -> Self {
        Self {
            enabled: true,
            histogram: true,
            adjacent_cosine: true,
        }
    }
}

impl Default for LayerHooks {
    fn default() -> Self {
        Self::off()
    }
}

pub fn summarize_tensor(values: &[f32], include_histogram: bool) -> TensorStats {
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut finite = true;
    let mut histogram = CompactHistogram {
        negative_large: 0,
        negative_small: 0,
        near_zero: 0,
        positive_small: 0,
        positive_large: 0,
        non_finite: 0,
    };

    for &value in values {
        if !value.is_finite() {
            finite = false;
            histogram.non_finite += 1;
            continue;
        }
        sum += value as f64;
        sum_sq += (value as f64) * (value as f64);
        max_abs = max_abs.max(value.abs());

        if include_histogram {
            if value < -1.0 {
                histogram.negative_large += 1;
            } else if value < -1e-6 {
                histogram.negative_small += 1;
            } else if value <= 1e-6 {
                histogram.near_zero += 1;
            } else if value <= 1.0 {
                histogram.positive_small += 1;
            } else {
                histogram.positive_large += 1;
            }
        }
    }

    let (mean, l2) = if finite {
        let mean = if values.is_empty() {
            0.0
        } else {
            (sum / values.len() as f64) as f32
        };
        (Some(mean), Some(sum_sq.sqrt() as f32))
    } else {
        (None, None)
    };

    TensorStats {
        len: values.len(),
        mean,
        l2,
        max_abs,
        histogram: include_histogram.then_some(histogram),
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        dot += (x as f64) * (y as f64);
        aa += (x as f64) * (x as f64);
        bb += (y as f64) * (y as f64);
    }
    if aa == 0.0 || bb == 0.0 {
        return None;
    }
    Some((dot / (aa.sqrt() * bb.sqrt())) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_report_mean_l2_max_and_histogram() {
        let stats = summarize_tensor(&[-2.0, -0.5, 0.0, 0.25, 3.0], true);
        assert_eq!(stats.len, 5);
        assert_eq!(stats.mean, Some(0.15));
        assert!((stats.l2.unwrap() - 3.6496575).abs() < 1e-6);
        assert_eq!(stats.max_abs, 3.0);
        assert_eq!(
            stats.histogram.unwrap(),
            CompactHistogram {
                negative_large: 1,
                negative_small: 1,
                near_zero: 1,
                positive_small: 1,
                positive_large: 1,
                non_finite: 0,
            }
        );
    }

    #[test]
    fn non_finite_values_disable_mean_and_l2() {
        let stats = summarize_tensor(&[1.0, f32::NAN], true);
        assert_eq!(stats.mean, None);
        assert_eq!(stats.l2, None);
        assert_eq!(stats.histogram.unwrap().non_finite, 1);
    }

    #[test]
    fn cosine_handles_equal_orthogonal_and_invalid_vectors() {
        assert!((cosine_similarity(&[1.0, 2.0], &[1.0, 2.0]).unwrap() - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap().abs() < 1e-6);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), None);
        assert_eq!(cosine_similarity(&[f32::NAN], &[1.0]), None);
    }

    #[test]
    fn off_hooks_are_zero_cost_contract() {
        assert_eq!(
            LayerHooks::off(),
            LayerHooks {
                enabled: false,
                histogram: false,
                adjacent_cosine: false,
            }
        );
    }
}
