use auralis::equivalence::{assert_f32_slices_equivalent, compare_f32_slices, Tolerance};
use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn fixture_config() -> Config {
    Config {
        vocab: 11,
        n_embd: 8,
        n_head: 2,
        n_layer: 2,
        block: 6,
        n_ff: 16,
    }
}

fn fixture_model(seed: u64) -> Gpt {
    let mut rng = StdRng::seed_from_u64(seed);
    Gpt::new(fixture_config(), &mut rng)
}

#[test]
fn same_seed_model_loss_and_gradients_are_bit_exact() {
    let a = fixture_model(0xA11CE_2901);
    let b = fixture_model(0xA11CE_2901);
    let x = [0, 1, 2, 3, 4, 5];
    let y = [1, 2, 3, 4, 5, 6];

    assert_f32_slices_equivalent(
        "same-seed parameters",
        &a.collect_params(),
        &b.collect_params(),
        Tolerance::EXACT,
    );

    let loss_a = a.loss(&x, &y);
    let loss_b = b.loss(&x, &y);
    assert_eq!(loss_a.to_bits(), loss_b.to_bits());

    let mut grad_a = vec![0.0; a.collect_params().len()];
    let mut grad_b = vec![0.0; b.collect_params().len()];
    let backward_loss_a = a.backward_into(&x, &y, &mut grad_a);
    let backward_loss_b = b.backward_into(&x, &y, &mut grad_b);

    assert_eq!(backward_loss_a.to_bits(), backward_loss_b.to_bits());
    assert_eq!(loss_a.to_bits(), backward_loss_a.to_bits());
    assert_f32_slices_equivalent(
        "same-seed gradients",
        &grad_a,
        &grad_b,
        Tolerance::EXACT,
    );
}

#[test]
fn one_adam_step_produces_bit_exact_parameter_and_state_transitions() {
    let model = fixture_model(0xA11CE_2902);
    let x = [5, 4, 3, 2, 1, 0];
    let y = [4, 3, 2, 1, 0, 10];
    let mut grads = vec![0.0; model.collect_params().len()];
    model.backward_into(&x, &y, &mut grads);

    let mut params_a = model.collect_params();
    let mut params_b = model.collect_params();
    let mut adam_a = Adam::new(params_a.len(), 3e-3);
    let mut adam_b = Adam::new(params_b.len(), 3e-3);

    adam_a.step(&mut params_a, &grads);
    adam_b.step(&mut params_b, &grads);

    assert_f32_slices_equivalent(
        "Adam parameter transition",
        &params_a,
        &params_b,
        Tolerance::EXACT,
    );

    let (lr_a, t_a, m_a, v_a) = adam_a.export();
    let (lr_b, t_b, m_b, v_b) = adam_b.export();
    assert_eq!(lr_a.to_bits(), lr_b.to_bits());
    assert_eq!(t_a, t_b);
    assert_f32_slices_equivalent("Adam m state", m_a, m_b, Tolerance::EXACT);
    assert_f32_slices_equivalent("Adam v state", v_a, v_b, Tolerance::EXACT);
}

#[test]
fn verifier_detects_gradient_and_optimizer_state_perturbations() {
    let model = fixture_model(0xA11CE_2903);
    let x = [0, 2, 4, 6, 8, 10];
    let y = [1, 3, 5, 7, 9, 0];
    let mut grads = vec![0.0; model.collect_params().len()];
    model.backward_into(&x, &y, &mut grads);

    let mut bad_grads = grads.clone();
    let index = bad_grads
        .iter()
        .position(|value| value.is_finite())
        .expect("fixture must contain gradients");
    bad_grads[index] = f32::from_bits(bad_grads[index].to_bits() ^ 1);
    let report = compare_f32_slices(&grads, &bad_grads, Tolerance::EXACT);
    assert_eq!(report.first_mismatch, Some(index));
    assert_eq!(report.mismatches, 1);

    let mut params = model.collect_params();
    let mut adam = Adam::new(params.len(), 3e-3);
    adam.step(&mut params, &grads);
    let (_, _, m, v) = adam.export();

    let mut bad_m = m.to_vec();
    bad_m[index] = f32::from_bits(bad_m[index].to_bits() ^ 1);
    assert!(!compare_f32_slices(m, &bad_m, Tolerance::EXACT).is_equivalent());

    let mut bad_v = v.to_vec();
    bad_v[index] = f32::from_bits(bad_v[index].to_bits() ^ 1);
    assert!(!compare_f32_slices(v, &bad_v, Tolerance::EXACT).is_equivalent());
}
