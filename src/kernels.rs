//! CPU kernels used by Engine experiments.
//!
//! Reference implementations stay intentionally simple. Candidate kernels must
//! prove numerical equivalence and measured value before replacing a reference
//! path in the model.

/// Reference forward matrix multiplication `A * B` into an existing buffer.
///
/// The loop order matches the current model implementation: `i -> k -> j`.
pub fn matmul_reference_into(
    a: &[f32],
    rows: usize,
    inner: usize,
    b: &[f32],
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(b.len(), inner * cols);
    assert_eq!(out.len(), rows * cols);
    out.fill(0.0);

    for i in 0..rows {
        for k in 0..inner {
            let av = a[i * inner + k];
            for j in 0..cols {
                out[i * cols + j] += av * b[k * cols + j];
            }
        }
    }
}

/// Forward `A * B` using pre-sliced contiguous rows.
///
/// Every output element still accumulates `k=0..inner` in exactly the same
/// order as [`matmul_reference_into`]. Only address calculation/bounds-check
/// structure changes, so bit-for-bit equality is expected.
pub fn matmul_row_slices_into(
    a: &[f32],
    rows: usize,
    inner: usize,
    b: &[f32],
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(b.len(), inner * cols);
    assert_eq!(out.len(), rows * cols);
    out.fill(0.0);

    for i in 0..rows {
        let a_row = &a[i * inner..(i + 1) * inner];
        let out_row = &mut out[i * cols..(i + 1) * cols];
        for k in 0..inner {
            let av = a_row[k];
            let b_row = &b[k * cols..(k + 1) * cols];
            for (dst, &weight) in out_row.iter_mut().zip(b_row) {
                *dst += av * weight;
            }
        }
    }
}

/// Reference gradient for the right-hand matrix in `A * B`.
///
/// Adds `A^T * dY` into `dB`. This preserves the original scalar loop order.
pub fn matmul_grad_b_reference(
    a: &[f32],
    rows: usize,
    inner: usize,
    dy: &[f32],
    cols: usize,
    db: &mut [f32],
) {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(dy.len(), rows * cols);
    assert_eq!(db.len(), inner * cols);

    for k in 0..inner {
        for j in 0..cols {
            let mut s = 0.0f32;
            for i in 0..rows {
                s += a[i * inner + k] * dy[i * cols + j];
            }
            db[k * cols + j] += s;
        }
    }
}

/// Cache-friendly `A^T * dY` for a zeroed `dB` buffer.
///
/// For every output cell, rows are still accumulated in ascending `i` order,
/// exactly like [`matmul_grad_b_reference`]. The loop nest is reordered across
/// independent output cells so `dY` and each `dB` row are traversed contiguously.
///
/// Engine's backward workspace clears every parameter-gradient buffer before a
/// sample, so this zeroed-output contract matches the hot path we want to test.
pub fn matmul_grad_b_rowwise_zeroed(
    a: &[f32],
    rows: usize,
    inner: usize,
    dy: &[f32],
    cols: usize,
    db: &mut [f32],
) {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(dy.len(), rows * cols);
    assert_eq!(db.len(), inner * cols);
    debug_assert!(db.iter().all(|&x| x == 0.0));

    for i in 0..rows {
        let a_row = &a[i * inner..(i + 1) * inner];
        let dy_row = &dy[i * cols..(i + 1) * cols];
        for k in 0..inner {
            let av = a_row[k];
            let db_row = &mut db[k * cols..(k + 1) * cols];
            for j in 0..cols {
                db_row[j] += av * dy_row[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        matmul_grad_b_reference, matmul_grad_b_rowwise_zeroed, matmul_reference_into,
        matmul_row_slices_into,
    };

    fn data(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let raw = ((i * 37 + salt * 19 + i / 3) % 101) as f32;
                (raw - 50.0) / 31.0
            })
            .collect()
    }

    fn assert_grad_b_exact(rows: usize, inner: usize, cols: usize) {
        let a = data(rows * inner, 1);
        let dy = data(rows * cols, 7);
        let mut reference = vec![0.0; inner * cols];
        let mut rowwise = vec![0.0; inner * cols];

        matmul_grad_b_reference(&a, rows, inner, &dy, cols, &mut reference);
        matmul_grad_b_rowwise_zeroed(&a, rows, inner, &dy, cols, &mut rowwise);

        assert_eq!(rowwise, reference);
    }

    fn assert_matmul_exact(rows: usize, inner: usize, cols: usize) {
        let a = data(rows * inner, 3);
        let b = data(inner * cols, 13);
        let mut reference = vec![f32::NAN; rows * cols];
        let mut sliced = vec![f32::NAN; rows * cols];

        matmul_reference_into(&a, rows, inner, &b, cols, &mut reference);
        matmul_row_slices_into(&a, rows, inner, &b, cols, &mut sliced);

        assert_eq!(sliced, reference);
    }

    #[test]
    fn rowwise_grad_b_kernel_matches_reference_bit_for_bit() {
        for (rows, inner, cols) in [
            (1, 1, 1),
            (4, 3, 5),
            (7, 8, 3),
            (32, 32, 32),
            (32, 32, 96),
            (32, 96, 32),
            (32, 32, 100),
        ] {
            assert_grad_b_exact(rows, inner, cols);
        }
    }

    #[test]
    fn row_slice_matmul_matches_reference_bit_for_bit() {
        for (rows, inner, cols) in [
            (1, 1, 1),
            (4, 3, 5),
            (7, 8, 3),
            (32, 32, 32),
            (32, 32, 96),
            (32, 96, 32),
            (32, 32, 100),
        ] {
            assert_matmul_exact(rows, inner, cols);
        }
    }

    #[test]
    fn reference_grad_b_keeps_additive_contract() {
        let rows = 3;
        let inner = 2;
        let cols = 4;
        let a = data(rows * inner, 2);
        let dy = data(rows * cols, 5);
        let mut once = vec![0.0; inner * cols];
        matmul_grad_b_reference(&a, rows, inner, &dy, cols, &mut once);

        let mut twice = once.clone();
        matmul_grad_b_reference(&a, rows, inner, &dy, cols, &mut twice);
        for (a, b) in twice.iter().zip(&once) {
            assert_eq!(*a, *b + *b);
        }
    }
}
