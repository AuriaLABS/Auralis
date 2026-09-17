//! Experimental SIMD dispatch for Engine E3.5.
//!
//! SIMD is never required to execute Auralis. `Portable` always remains
//! available and `Auto` only selects AVX2 after runtime feature detection and
//! for shapes with measured material wins. This module stays isolated from the
//! production model path until equivalence and benchmarks justify promotion.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdBackend {
    Portable,
    Auto,
    Avx2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedSimdBackend {
    Portable,
    Avx2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdCapabilities {
    pub avx2: bool,
}

pub fn capabilities() -> SimdCapabilities {
    SimdCapabilities {
        avx2: avx2_supported(),
    }
}

pub fn avx2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub fn resolve_backend(
    backend: SimdBackend,
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<ResolvedSimdBackend, &'static str> {
    match backend {
        SimdBackend::Portable => Ok(ResolvedSimdBackend::Portable),
        SimdBackend::Auto => {
            // Only shapes with repeatable material wins in E3.5 runs are
            // promoted. 32x32x96 stayed positive but dropped to ~4.7% on the
            // second runner, so it remains portable in Auto until more evidence.
            let measured_material_win = matches!((rows, inner, cols), (32, 32, 32) | (32, 96, 32));
            if avx2_supported() && measured_material_win {
                Ok(ResolvedSimdBackend::Avx2)
            } else {
                Ok(ResolvedSimdBackend::Portable)
            }
        }
        SimdBackend::Avx2 => {
            if avx2_supported() {
                Ok(ResolvedSimdBackend::Avx2)
            } else {
                Err("AVX2 requested but not supported by this CPU")
            }
        }
    }
}

pub fn matmul_simd_into(
    backend: SimdBackend,
    a: &[f32],
    rows: usize,
    inner: usize,
    b: &[f32],
    cols: usize,
    out: &mut [f32],
) -> Result<ResolvedSimdBackend, &'static str> {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(b.len(), inner * cols);
    assert_eq!(out.len(), rows * cols);

    let resolved = resolve_backend(backend, rows, inner, cols)?;
    match resolved {
        ResolvedSimdBackend::Portable => {
            crate::kernels::matmul_reference_into(a, rows, inner, b, cols, out);
        }
        ResolvedSimdBackend::Avx2 => {
            out.fill(0.0);
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: `resolve_backend` verifies AVX2 support at runtime.
                // Slice lengths were checked above, and the implementation only
                // performs unaligned loads/stores inside those validated ranges.
                unsafe { matmul_avx2_into(a, rows, inner, b, cols, out) };
            }
            #[cfg(not(target_arch = "x86_64"))]
            unreachable!("AVX2 cannot resolve on a non-x86_64 target");
        }
    }
    Ok(resolved)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_avx2_into(
    a: &[f32],
    rows: usize,
    inner: usize,
    b: &[f32],
    cols: usize,
    out: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };

    for i in 0..rows {
        let a_row = &a[i * inner..(i + 1) * inner];
        let out_row = &mut out[i * cols..(i + 1) * cols];
        for k in 0..inner {
            let av = a_row[k];
            let b_row = &b[k * cols..(k + 1) * cols];
            let av8 = _mm256_set1_ps(av);
            let mut j = 0usize;
            while j + 8 <= cols {
                // SAFETY: loop bound guarantees eight in-range contiguous f32s
                // for both source and destination. Unaligned intrinsics are used.
                let dst = unsafe { _mm256_loadu_ps(out_row.as_ptr().add(j)) };
                let weights = unsafe { _mm256_loadu_ps(b_row.as_ptr().add(j)) };
                let product = _mm256_mul_ps(av8, weights);
                let sum = _mm256_add_ps(dst, product);
                unsafe { _mm256_storeu_ps(out_row.as_mut_ptr().add(j), sum) };
                j += 8;
            }
            while j < cols {
                out_row[j] += av * b_row[j];
                j += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        avx2_supported, matmul_simd_into, resolve_backend, ResolvedSimdBackend, SimdBackend,
    };
    use crate::kernels::matmul_reference_into;

    fn data(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let raw = ((i * 47 + salt * 29 + i / 3) % 137) as f32;
                (raw - 68.0) / 43.0
            })
            .collect()
    }

    fn assert_exact(backend: SimdBackend, rows: usize, inner: usize, cols: usize) {
        let a = data(rows * inner, 3);
        let b = data(inner * cols, 11);
        let mut reference = vec![f32::NAN; rows * cols];
        let mut candidate = vec![f32::NAN; rows * cols];
        matmul_reference_into(&a, rows, inner, &b, cols, &mut reference);
        matmul_simd_into(backend, &a, rows, inner, &b, cols, &mut candidate).unwrap();
        assert_eq!(candidate, reference, "shape={rows}x{inner}x{cols}");
    }

    #[test]
    fn portable_is_always_exact_reference() {
        for shape in [(1, 1, 1), (4, 3, 5), (9, 13, 37), (32, 32, 100)] {
            assert_exact(SimdBackend::Portable, shape.0, shape.1, shape.2);
        }
    }

    #[test]
    fn auto_selects_avx2_only_for_repeatable_material_wins() {
        let selected = if avx2_supported() {
            ResolvedSimdBackend::Avx2
        } else {
            ResolvedSimdBackend::Portable
        };
        for shape in [(32, 32, 32), (32, 96, 32)] {
            assert_eq!(
                resolve_backend(SimdBackend::Auto, shape.0, shape.1, shape.2).unwrap(),
                selected
            );
        }
        for shape in [
            (32, 32, 96),
            (32, 32, 100),
            (128, 128, 128),
            (7, 8, 3),
            (16, 24, 71),
        ] {
            assert_eq!(
                resolve_backend(SimdBackend::Auto, shape.0, shape.1, shape.2).unwrap(),
                ResolvedSimdBackend::Portable
            );
        }
    }

    #[test]
    fn auto_has_safe_fallback_and_exact_output() {
        for shape in [
            (7, 8, 3),
            (16, 24, 71),
            (32, 32, 32),
            (32, 32, 96),
            (32, 96, 32),
            (32, 32, 100),
            (128, 128, 128),
        ] {
            assert_exact(SimdBackend::Auto, shape.0, shape.1, shape.2);
        }
    }

    #[test]
    fn explicit_avx2_is_exact_when_available() {
        if avx2_supported() {
            for shape in [(8, 8, 8), (9, 13, 37), (32, 32, 96), (32, 32, 100)] {
                assert_exact(SimdBackend::Avx2, shape.0, shape.1, shape.2);
            }
        } else {
            assert!(resolve_backend(SimdBackend::Avx2, 8, 8, 8).is_err());
        }
    }
}
