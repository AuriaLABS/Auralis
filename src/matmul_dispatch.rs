//! Explicit CPU matmul dispatch for Engine E2 experiments.
//!
//! `Auto` is intentionally conservative: it selects the blocked candidate only
//! for shapes with measured wins and falls back to the reference kernel for all
//! other shapes. This keeps dispatch auditable and prevents extrapolating beyond
//! the benchmark corpus.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulBackend {
    Reference,
    Blocked32,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedMatmulBackend {
    Reference,
    Blocked32,
}

pub fn resolve_backend(
    backend: MatmulBackend,
    rows: usize,
    inner: usize,
    cols: usize,
) -> ResolvedMatmulBackend {
    match backend {
        MatmulBackend::Reference => ResolvedMatmulBackend::Reference,
        MatmulBackend::Blocked32 => ResolvedMatmulBackend::Blocked32,
        MatmulBackend::Auto => match (rows, inner, cols) {
            // Measured wins in E2 candidate run 35148207755.
            (32, 32, 32) | (32, 32, 96) | (32, 96, 32) => {
                ResolvedMatmulBackend::Blocked32
            }
            // Known regression: 32x32x100 candidate/reference = 1.1736.
            // Any unknown shape also falls back to reference until measured.
            _ => ResolvedMatmulBackend::Reference,
        },
    }
}

pub fn matmul_dispatch_into(
    backend: MatmulBackend,
    a: &[f32],
    rows: usize,
    inner: usize,
    b: &[f32],
    cols: usize,
    out: &mut [f32],
) -> ResolvedMatmulBackend {
    let resolved = resolve_backend(backend, rows, inner, cols);
    match resolved {
        ResolvedMatmulBackend::Reference => {
            crate::kernels::matmul_reference_into(a, rows, inner, b, cols, out)
        }
        ResolvedMatmulBackend::Blocked32 => {
            crate::cpu_matmul::matmul_blocked_32_into(a, rows, inner, b, cols, out)
        }
    }
    resolved
}

#[cfg(test)]
mod tests {
    use super::{matmul_dispatch_into, resolve_backend, MatmulBackend, ResolvedMatmulBackend};
    use crate::kernels::matmul_reference_into;

    fn data(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let raw = ((i * 37 + salt * 19 + i / 5) % 131) as f32;
                (raw - 65.0) / 41.0
            })
            .collect()
    }

    #[test]
    fn auto_selects_only_measured_wins() {
        for shape in [(32, 32, 32), (32, 32, 96), (32, 96, 32)] {
            assert_eq!(
                resolve_backend(MatmulBackend::Auto, shape.0, shape.1, shape.2),
                ResolvedMatmulBackend::Blocked32
            );
        }

        for shape in [(32, 32, 100), (16, 16, 16), (1, 1, 1), (64, 64, 64)] {
            assert_eq!(
                resolve_backend(MatmulBackend::Auto, shape.0, shape.1, shape.2),
                ResolvedMatmulBackend::Reference
            );
        }
    }

    #[test]
    fn explicit_backends_are_never_overridden() {
        assert_eq!(
            resolve_backend(MatmulBackend::Reference, 32, 32, 32),
            ResolvedMatmulBackend::Reference
        );
        assert_eq!(
            resolve_backend(MatmulBackend::Blocked32, 32, 32, 100),
            ResolvedMatmulBackend::Blocked32
        );
    }

    #[test]
    fn dispatch_is_exact_against_reference_for_measured_and_fallback_shapes() {
        for (rows, inner, cols) in [
            (32, 32, 32),
            (32, 32, 96),
            (32, 96, 32),
            (32, 32, 100),
            (9, 13, 37),
        ] {
            let a = data(rows * inner, 3);
            let b = data(inner * cols, 11);
            let mut reference = vec![f32::NAN; rows * cols];
            let mut auto = vec![f32::NAN; rows * cols];
            matmul_reference_into(&a, rows, inner, &b, cols, &mut reference);
            matmul_dispatch_into(
                MatmulBackend::Auto,
                &a,
                rows,
                inner,
                &b,
                cols,
                &mut auto,
            );
            assert_eq!(auto, reference, "shape={rows}x{inner}x{cols}");
        }
    }
}
