//! Portable CPU matmul candidates for Engine E2 experiments.
//!
//! These kernels are intentionally separate from the production path until
//! side-by-side benchmarks and independent equivalence checks justify promotion.

/// Cache-friendly tiled `A * B` candidate.
///
/// Matrices use contiguous row-major storage. The candidate tiles only the
/// output-column traversal while preserving each output cell's `k=0..inner`
/// accumulation order, so finite deterministic inputs are expected to match the
/// scalar/reference kernel bit-for-bit.
pub fn matmul_blocked_cols_into(
    a: &[f32],
    rows: usize,
    inner: usize,
    b: &[f32],
    cols: usize,
    out: &mut [f32],
    tile_cols: usize,
) {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(b.len(), inner * cols);
    assert_eq!(out.len(), rows * cols);
    assert!(tile_cols > 0, "tile_cols must be positive");

    out.fill(0.0);

    for i in 0..rows {
        let a_row = &a[i * inner..(i + 1) * inner];
        let out_row = &mut out[i * cols..(i + 1) * cols];
        for k in 0..inner {
            let av = a_row[k];
            let b_row = &b[k * cols..(k + 1) * cols];
            let mut start = 0usize;
            while start < cols {
                let end = (start + tile_cols).min(cols);
                let out_tile = &mut out_row[start..end];
                let b_tile = &b_row[start..end];
                for (dst, &weight) in out_tile.iter_mut().zip(b_tile) {
                    *dst += av * weight;
                }
                start = end;
            }
        }
    }
}

/// Small dispatcher for the first E2 candidate sweep.
///
/// Keeping tile selection explicit makes benchmarks reproducible and avoids
/// hiding architecture-specific heuristics in the correctness contract.
pub fn matmul_blocked_32_into(
    a: &[f32],
    rows: usize,
    inner: usize,
    b: &[f32],
    cols: usize,
    out: &mut [f32],
) {
    matmul_blocked_cols_into(a, rows, inner, b, cols, out, 32);
}

#[cfg(test)]
mod tests {
    use super::{matmul_blocked_32_into, matmul_blocked_cols_into};
    use crate::kernels::matmul_reference_into;

    fn data(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let raw = ((i * 41 + salt * 23 + i / 7) % 127) as f32;
                (raw - 63.0) / 37.0
            })
            .collect()
    }

    fn assert_exact(rows: usize, inner: usize, cols: usize, tile_cols: usize) {
        let a = data(rows * inner, 3);
        let b = data(inner * cols, 11);
        let mut reference = vec![f32::NAN; rows * cols];
        let mut candidate = vec![f32::NAN; rows * cols];

        matmul_reference_into(&a, rows, inner, &b, cols, &mut reference);
        matmul_blocked_cols_into(
            &a,
            rows,
            inner,
            &b,
            cols,
            &mut candidate,
            tile_cols,
        );

        assert_eq!(candidate, reference);
    }

    #[test]
    fn blocked_candidate_matches_reference_on_irregular_shapes() {
        for (rows, inner, cols) in [
            (1, 1, 1),
            (4, 3, 5),
            (7, 8, 3),
            (9, 13, 37),
            (32, 32, 32),
            (32, 32, 96),
            (32, 96, 32),
            (32, 32, 100),
        ] {
            for tile in [1, 4, 16, 32, 64] {
                assert_exact(rows, inner, cols, tile);
            }
        }
    }

    #[test]
    fn fixed_tile_dispatch_matches_reference() {
        let rows = 16;
        let inner = 24;
        let cols = 71;
        let a = data(rows * inner, 17);
        let b = data(inner * cols, 29);
        let mut reference = vec![0.0; rows * cols];
        let mut candidate = vec![0.0; rows * cols];

        matmul_reference_into(&a, rows, inner, &b, cols, &mut reference);
        matmul_blocked_32_into(&a, rows, inner, &b, cols, &mut candidate);
        assert_eq!(candidate, reference);
    }

    #[test]
    #[should_panic(expected = "tile_cols must be positive")]
    fn zero_tile_is_rejected() {
        let mut out = [0.0f32; 1];
        matmul_blocked_cols_into(&[1.0], 1, 1, &[1.0], 1, &mut out, 0);
    }
}
