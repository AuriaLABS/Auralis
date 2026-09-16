use auralis::equivalence::{assert_f32_slices_equivalent, compare_f32_slices, Tolerance};
use auralis::kernels::{
    matmul_b_t_reference_into, matmul_b_t_row_slices_into, matmul_reference_into,
    matmul_row_slices_into,
};

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 29 + salt * 17 + i / 5) % 97) as f32;
            (raw - 48.0) / 23.0
        })
        .collect()
}

#[test]
fn forward_reference_and_slice_candidate_match_exactly() {
    for (rows, inner, cols) in [
        (1, 1, 1),
        (3, 5, 2),
        (7, 8, 3),
        (32, 32, 32),
        (32, 32, 96),
        (32, 96, 32),
        (32, 32, 100),
    ] {
        let a = data(rows * inner, 11);
        let b = data(inner * cols, 23);
        let mut reference = vec![f32::NAN; rows * cols];
        let mut candidate = vec![f32::NAN; rows * cols];

        matmul_reference_into(&a, rows, inner, &b, cols, &mut reference);
        matmul_row_slices_into(&a, rows, inner, &b, cols, &mut candidate);

        assert_f32_slices_equivalent(
            "forward matmul reference vs slices",
            &reference,
            &candidate,
            Tolerance::EXACT,
        );
    }
}

#[test]
fn backward_input_reference_and_slice_candidate_match_exactly() {
    for (rows, out_cols, result_cols) in [
        (1, 1, 1),
        (3, 5, 2),
        (7, 8, 3),
        (32, 100, 32),
        (32, 32, 96),
        (32, 96, 32),
    ] {
        let dy = data(rows * out_cols, 31);
        let b = data(result_cols * out_cols, 47);
        let mut reference = vec![f32::NAN; rows * result_cols];
        let mut candidate = vec![f32::NAN; rows * result_cols];

        matmul_b_t_reference_into(
            &dy,
            rows,
            out_cols,
            &b,
            result_cols,
            &mut reference,
        );
        matmul_b_t_row_slices_into(
            &dy,
            rows,
            out_cols,
            &b,
            result_cols,
            &mut candidate,
        );

        assert_f32_slices_equivalent(
            "backward input matmul reference vs slices",
            &reference,
            &candidate,
            Tolerance::EXACT,
        );
    }
}

#[test]
fn verifier_detects_an_intentional_candidate_regression() {
    let reference = [0.25, -1.0, 2.5, 8.0];
    let mut candidate = reference;
    candidate[2] += 1e-3;

    let comparison = compare_f32_slices(&reference, &candidate, Tolerance::EXACT);
    assert!(!comparison.is_equivalent());
    assert_eq!(comparison.mismatches, 1);
    assert_eq!(comparison.first_mismatch, Some(2));
}

#[test]
#[should_panic(expected = "equivalence shape mismatch")]
fn verifier_rejects_shape_mismatch() {
    let _ = compare_f32_slices(&[1.0, 2.0], &[1.0], Tolerance::EXACT);
}
