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

/// Reference `dY * B^T` into an existing output buffer.
pub fn matmul_b_t_reference_into(
    dy: &[f32],
    rows: usize,
    out_cols: usize,
    b: &[f32],
    result_cols: usize,
    out: &mut [f32],
) {
    assert_eq!(dy.len(), rows * out_cols);
    assert_eq!(b.len(), result_cols * out_cols);
    assert_eq!(out.len(), rows * result_cols);
    for i in 0..rows {
        for k in 0..result_cols {
            let mut s = 0.0f32;
            for j in 0..out_cols {
                s += dy[i * out_cols + j] * b[k * out_cols + j];
            }
            out[i * result_cols + k] = s;
        }
    }
}

/// Reference additive `dY * B^T` used when several branches feed one buffer.
pub fn matmul_b_t_reference_add_into(
    dy: &[f32],
    rows: usize,
    out_cols: usize,
    b: &[f32],
    result_cols: usize,
    out: &mut [f32],
) {
    assert_eq!(dy.len(), rows * out_cols);
    assert_eq!(b.len(), result_cols * out_cols);
    assert_eq!(out.len(), rows * result_cols);
    for i in 0..rows {
        for k in 0..result_cols {
            let mut s = 0.0f32;
            for j in 0..out_cols {
                s += dy[i * out_cols + j] * b[k * out_cols + j];
            }
            out[i * result_cols + k] += s;
        }
    }
}

/// Slice-based `dY * B^T` with the exact same per-cell `j` accumulation order.
pub fn matmul_b_t_row_slices_into(
    dy: &[f32],
    rows: usize,
    out_cols: usize,
    b: &[f32],
    result_cols: usize,
    out: &mut [f32],
) {
    assert_eq!(dy.len(), rows * out_cols);
    assert_eq!(b.len(), result_cols * out_cols);
    assert_eq!(out.len(), rows * result_cols);

    for i in 0..rows {
        let dy_row = &dy[i * out_cols..(i + 1) * out_cols];
        let out_row = &mut out[i * result_cols..(i + 1) * result_cols];
        for k in 0..result_cols {
            let b_row = &b[k * out_cols..(k + 1) * out_cols];
            let mut s = 0.0f32;
            for (&grad, &weight) in dy_row.iter().zip(b_row) {
                s += grad * weight;
            }
            out_row[k] = s;
        }
    }
}

/// Additive slice-based `dY * B^T`, preserving the exact `j` sum order.
pub fn matmul_b_t_row_slices_add_into(
    dy: &[f32],
    rows: usize,
    out_cols: usize,
    b: &[f32],
    result_cols: usize,
    out: &mut [f32],
) {
    assert_eq!(dy.len(), rows * out_cols);
    assert_eq!(b.len(), result_cols * out_cols);
    assert_eq!(out.len(), rows * result_cols);

    for i in 0..rows {
        let dy_row = &dy[i * out_cols..(i + 1) * out_cols];
        let out_row = &mut out[i * result_cols..(i + 1) * result_cols];
        for k in 0..result_cols {
            let b_row = &b[k * out_cols..(k + 1) * out_cols];
            let mut s = 0.0f32;
            for (&grad, &weight) in dy_row.iter().zip(b_row) {
                s += grad * weight;
            }
            out_row[k] += s;
        }
    }
}

/// Reference causal multi-head attention forward pass matching `model.rs`.
pub fn attention_forward_reference_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
    out: &mut [f32],
    probs: &mut [f32],
) {
    assert_eq!(q.len(), t * d);
    assert_eq!(k.len(), t * d);
    assert_eq!(v.len(), t * d);
    assert_eq!(out.len(), t * d);
    assert_eq!(probs.len(), n_head * t * t);
    assert!(n_head > 0 && d % n_head == 0);
    out.fill(0.0);
    probs.fill(0.0);

    let hd = d / n_head;
    let scale = 1.0 / (hd as f32).sqrt();
    for h in 0..n_head {
        let hoff = h * hd;
        for i in 0..t {
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..=i {
                let mut s = 0.0;
                for z in 0..hd {
                    s += q[i * d + hoff + z] * k[j * d + hoff + z];
                }
                s *= scale;
                let idx = (h * t + i) * t + j;
                probs[idx] = s;
                max_score = max_score.max(s);
            }
            let mut sum = 0.0;
            for j in 0..=i {
                let idx = (h * t + i) * t + j;
                let e = (probs[idx] - max_score).exp();
                probs[idx] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-20);
            for j in 0..=i {
                let pidx = (h * t + i) * t + j;
                probs[pidx] *= inv;
                let p = probs[pidx];
                for z in 0..hd {
                    out[i * d + hoff + z] += p * v[j * d + hoff + z];
                }
            }
        }
    }
}

/// Slice-based causal attention preserving every `z` and `j` accumulation order.
pub fn attention_forward_row_slices_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
    out: &mut [f32],
    probs: &mut [f32],
) {
    assert_eq!(q.len(), t * d);
    assert_eq!(k.len(), t * d);
    assert_eq!(v.len(), t * d);
    assert_eq!(out.len(), t * d);
    assert_eq!(probs.len(), n_head * t * t);
    assert!(n_head > 0 && d % n_head == 0);
    out.fill(0.0);
    probs.fill(0.0);

    let hd = d / n_head;
    let scale = 1.0 / (hd as f32).sqrt();
    for h in 0..n_head {
        let hoff = h * hd;
        for i in 0..t {
            let q_head = &q[i * d + hoff..i * d + hoff + hd];
            let prob_start = (h * t + i) * t;
            let prob_row = &mut probs[prob_start..prob_start + t];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..=i {
                let k_head = &k[j * d + hoff..j * d + hoff + hd];
                let mut s = 0.0f32;
                for (&qv, &kv) in q_head.iter().zip(k_head) {
                    s += qv * kv;
                }
                s *= scale;
                prob_row[j] = s;
                max_score = max_score.max(s);
            }

            let mut sum = 0.0f32;
            for score in &mut prob_row[..=i] {
                let e = (*score - max_score).exp();
                *score = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-20);
            let out_head = &mut out[i * d + hoff..i * d + hoff + hd];
            for j in 0..=i {
                prob_row[j] *= inv;
                let p = prob_row[j];
                let v_head = &v[j * d + hoff..j * d + hoff + hd];
                for (dst, &vv) in out_head.iter_mut().zip(v_head) {
                    *dst += p * vv;
                }
            }
        }
    }
}

/// Reference causal multi-head attention backward pass matching `model.rs`.
pub fn attention_backward_reference_into(
    dout: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    dp: &mut [f32],
) {
    assert_eq!(dout.len(), t * d);
    assert_eq!(q.len(), t * d);
    assert_eq!(k.len(), t * d);
    assert_eq!(v.len(), t * d);
    assert_eq!(probs.len(), n_head * t * t);
    assert_eq!(dq.len(), t * d);
    assert_eq!(dk.len(), t * d);
    assert_eq!(dv.len(), t * d);
    assert_eq!(dp.len(), t);
    assert!(n_head > 0 && d % n_head == 0);
    dq.fill(0.0);
    dk.fill(0.0);
    dv.fill(0.0);
    dp.fill(0.0);

    let hd = d / n_head;
    let scale = 1.0 / (hd as f32).sqrt();
    for h in 0..n_head {
        let hoff = h * hd;
        for i in 0..t {
            dp[..=i].fill(0.0);
            for j in 0..=i {
                let p = probs[(h * t + i) * t + j];
                for z in 0..hd {
                    let go = dout[i * d + hoff + z];
                    dp[j] += go * v[j * d + hoff + z];
                    dv[j * d + hoff + z] += p * go;
                }
            }
            let mut dot = 0.0;
            for j in 0..=i {
                dot += dp[j] * probs[(h * t + i) * t + j];
            }
            for j in 0..=i {
                let p = probs[(h * t + i) * t + j];
                let ds = p * (dp[j] - dot) * scale;
                for z in 0..hd {
                    let qi = q[i * d + hoff + z];
                    let kj = k[j * d + hoff + z];
                    dq[i * d + hoff + z] += ds * kj;
                    dk[j * d + hoff + z] += ds * qi;
                }
            }
        }
    }
}

/// Slice-based attention backward preserving the exact `h -> i -> j -> z` order.
pub fn attention_backward_row_slices_into(
    dout: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    t: usize,
    d: usize,
    n_head: usize,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    dp: &mut [f32],
) {
    assert_eq!(dout.len(), t * d);
    assert_eq!(q.len(), t * d);
    assert_eq!(k.len(), t * d);
    assert_eq!(v.len(), t * d);
    assert_eq!(probs.len(), n_head * t * t);
    assert_eq!(dq.len(), t * d);
    assert_eq!(dk.len(), t * d);
    assert_eq!(dv.len(), t * d);
    assert_eq!(dp.len(), t);
    assert!(n_head > 0 && d % n_head == 0);
    dq.fill(0.0);
    dk.fill(0.0);
    dv.fill(0.0);
    dp.fill(0.0);

    let hd = d / n_head;
    let scale = 1.0 / (hd as f32).sqrt();
    for h in 0..n_head {
        let hoff = h * hd;
        for i in 0..t {
            dp[..=i].fill(0.0);
            let dout_head = &dout[i * d + hoff..i * d + hoff + hd];
            let q_head = &q[i * d + hoff..i * d + hoff + hd];
            let prob_start = (h * t + i) * t;
            let prob_row = &probs[prob_start..prob_start + t];

            for j in 0..=i {
                let p = prob_row[j];
                let v_head = &v[j * d + hoff..j * d + hoff + hd];
                let dv_head = &mut dv[j * d + hoff..j * d + hoff + hd];
                for z in 0..hd {
                    let go = dout_head[z];
                    dp[j] += go * v_head[z];
                    dv_head[z] += p * go;
                }
            }

            let mut dot = 0.0f32;
            for j in 0..=i {
                dot += dp[j] * prob_row[j];
            }

            let dq_head = &mut dq[i * d + hoff..i * d + hoff + hd];
            for j in 0..=i {
                let p = prob_row[j];
                let ds = p * (dp[j] - dot) * scale;
                let k_head = &k[j * d + hoff..j * d + hoff + hd];
                let dk_head = &mut dk[j * d + hoff..j * d + hoff + hd];
                for z in 0..hd {
                    let qi = q_head[z];
                    let kj = k_head[z];
                    dq_head[z] += ds * kj;
                    dk_head[z] += ds * qi;
                }
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
        attention_backward_reference_into, attention_backward_row_slices_into,
        attention_forward_reference_into, attention_forward_row_slices_into,
        matmul_b_t_reference_add_into, matmul_b_t_reference_into,
        matmul_b_t_row_slices_add_into, matmul_b_t_row_slices_into,
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

    fn assert_matmul_b_t_exact(rows: usize, out_cols: usize, result_cols: usize) {
        let dy = data(rows * out_cols, 5);
        let b = data(result_cols * out_cols, 23);
        let mut reference = vec![f32::NAN; rows * result_cols];
        let mut sliced = vec![f32::NAN; rows * result_cols];
        matmul_b_t_reference_into(&dy, rows, out_cols, &b, result_cols, &mut reference);
        matmul_b_t_row_slices_into(&dy, rows, out_cols, &b, result_cols, &mut sliced);
        assert_eq!(sliced, reference);

        let initial = data(rows * result_cols, 31);
        let mut reference_add = initial.clone();
        let mut sliced_add = initial;
        matmul_b_t_reference_add_into(
            &dy,
            rows,
            out_cols,
            &b,
            result_cols,
            &mut reference_add,
        );
        matmul_b_t_row_slices_add_into(
            &dy,
            rows,
            out_cols,
            &b,
            result_cols,
            &mut sliced_add,
        );
        assert_eq!(sliced_add, reference_add);
    }

    fn assert_attention_forward_exact(t: usize, d: usize, n_head: usize) {
        let q = data(t * d, 37);
        let k = data(t * d, 41);
        let v = data(t * d, 43);
        let mut reference_out = vec![f32::NAN; t * d];
        let mut sliced_out = vec![f32::NAN; t * d];
        let mut reference_probs = vec![f32::NAN; n_head * t * t];
        let mut sliced_probs = vec![f32::NAN; n_head * t * t];

        attention_forward_reference_into(
            &q,
            &k,
            &v,
            t,
            d,
            n_head,
            &mut reference_out,
            &mut reference_probs,
        );
        attention_forward_row_slices_into(
            &q,
            &k,
            &v,
            t,
            d,
            n_head,
            &mut sliced_out,
            &mut sliced_probs,
        );

        assert_eq!(sliced_probs, reference_probs);
        assert_eq!(sliced_out, reference_out);
    }

    fn assert_attention_backward_exact(t: usize, d: usize, n_head: usize) {
        let q = data(t * d, 37);
        let k = data(t * d, 41);
        let v = data(t * d, 43);
        let dout = data(t * d, 47);
        let mut forward_out = vec![0.0; t * d];
        let mut probs = vec![0.0; n_head * t * t];
        attention_forward_reference_into(
            &q,
            &k,
            &v,
            t,
            d,
            n_head,
            &mut forward_out,
            &mut probs,
        );

        let mut reference_dq = vec![f32::NAN; t * d];
        let mut reference_dk = vec![f32::NAN; t * d];
        let mut reference_dv = vec![f32::NAN; t * d];
        let mut reference_dp = vec![f32::NAN; t];
        let mut sliced_dq = vec![f32::NAN; t * d];
        let mut sliced_dk = vec![f32::NAN; t * d];
        let mut sliced_dv = vec![f32::NAN; t * d];
        let mut sliced_dp = vec![f32::NAN; t];

        attention_backward_reference_into(
            &dout,
            &q,
            &k,
            &v,
            &probs,
            t,
            d,
            n_head,
            &mut reference_dq,
            &mut reference_dk,
            &mut reference_dv,
            &mut reference_dp,
        );
        attention_backward_row_slices_into(
            &dout,
            &q,
            &k,
            &v,
            &probs,
            t,
            d,
            n_head,
            &mut sliced_dq,
            &mut sliced_dk,
            &mut sliced_dv,
            &mut sliced_dp,
        );

        assert_eq!(sliced_dq, reference_dq);
        assert_eq!(sliced_dk, reference_dk);
        assert_eq!(sliced_dv, reference_dv);
        assert_eq!(sliced_dp, reference_dp);
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
    fn row_slice_matmul_b_t_matches_reference_bit_for_bit() {
        for (rows, out_cols, result_cols) in [
            (1, 1, 1),
            (4, 3, 5),
            (7, 8, 3),
            (32, 100, 32),
            (32, 32, 96),
            (32, 96, 32),
            (32, 32, 32),
        ] {
            assert_matmul_b_t_exact(rows, out_cols, result_cols);
        }
    }

    #[test]
    fn row_slice_attention_forward_matches_reference_bit_for_bit() {
        for (t, d, n_head) in [(1, 4, 1), (4, 8, 2), (7, 8, 2), (32, 32, 4)] {
            assert_attention_forward_exact(t, d, n_head);
        }
    }

    #[test]
    fn row_slice_attention_backward_matches_reference_bit_for_bit() {
        for (t, d, n_head) in [(1, 4, 1), (4, 8, 2), (7, 8, 2), (32, 32, 4)] {
            assert_attention_backward_exact(t, d, n_head);
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
