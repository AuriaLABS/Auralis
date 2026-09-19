use auralis::equivalence::{compare_f32_slices, Tolerance};
use auralis::model::{Config, Gpt};
use rand::rngs::StdRng;
use rand::SeedableRng;

const LOSS_TOLERANCE: Tolerance = Tolerance::new(1e-6, 1e-6);

fn cross_entropy_from_logits(logits: &[f32], targets: &[usize], vocab: usize) -> f32 {
    assert_eq!(logits.len(), targets.len() * vocab);
    let mut loss = 0.0f32;

    for (row_index, &target) in targets.iter().enumerate() {
        assert!(target < vocab);
        let row = &logits[row_index * vocab..(row_index + 1) * vocab];
        let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &logit in row {
            sum += (logit - maxv).exp();
        }
        let p = ((row[target] - maxv).exp() / sum.max(1e-20)).max(1e-20);
        loss -= p.ln();
    }

    loss / targets.len() as f32
}

fn fixture_cases() -> [(Config, u64); 3] {
    [
        (
            Config {
                vocab: 7,
                n_embd: 4,
                n_head: 1,
                n_layer: 1,
                block: 3,
                n_ff: 8,
            },
            0xA11CE_2904,
        ),
        (
            Config {
                vocab: 11,
                n_embd: 8,
                n_head: 2,
                n_layer: 2,
                block: 5,
                n_ff: 16,
            },
            0xA11CE_2905,
        ),
        (
            Config {
                vocab: 13,
                n_embd: 12,
                n_head: 3,
                n_layer: 3,
                block: 6,
                n_ff: 24,
            },
            0xA11CE_2906,
        ),
    ]
}

#[test]
fn public_inference_logits_agree_with_cacheful_training_loss() {
    for (cfg, seed) in fixture_cases() {
        let mut rng = StdRng::seed_from_u64(seed);
        let gpt = Gpt::new(cfg, &mut rng);
        let tokens: Vec<usize> = (0..=cfg.block)
            .map(|i| (i * 5 + 1) % cfg.vocab)
            .collect();

        for len in 1..=cfg.block {
            let x = &tokens[..len];
            let y = &tokens[1..=len];
            let logits = gpt.logits(x);
            let external_loss = cross_entropy_from_logits(&logits, y, cfg.vocab);
            let eval_loss = gpt.loss(x, y);
            let mut grads = vec![0.0; gpt.collect_params().len()];
            let training_loss = gpt.backward_into(x, y, &mut grads);

            assert_eq!(
                external_loss.to_bits(),
                eval_loss.to_bits(),
                "external CE must reproduce public eval loss exactly: seed={seed:#x} len={len}"
            );

            let report = compare_f32_slices(
                &[external_loss],
                &[training_loss],
                LOSS_TOLERANCE,
            );
            assert!(
                report.is_equivalent(),
                "inference/training loss diverged: seed={seed:#x} heads={} layers={} len={len} external={external_loss:.9} training={training_loss:.9} report={report:?}",
                cfg.n_head,
                cfg.n_layer,
            );
            assert!(grads.iter().all(|g| g.is_finite()));
        }
    }
}

#[test]
fn public_logits_are_exactly_reproducible_for_same_seed() {
    for (cfg, seed) in fixture_cases() {
        let mut rng_a = StdRng::seed_from_u64(seed);
        let mut rng_b = StdRng::seed_from_u64(seed);
        let a = Gpt::new(cfg, &mut rng_a);
        let b = Gpt::new(cfg, &mut rng_b);
        let tokens: Vec<usize> = (0..cfg.block)
            .map(|i| (i * 3 + 2) % cfg.vocab)
            .collect();

        for len in 1..=cfg.block {
            let logits_a = a.logits(&tokens[..len]);
            let logits_b = b.logits(&tokens[..len]);
            let report = compare_f32_slices(&logits_a, &logits_b, Tolerance::EXACT);
            assert!(
                report.is_equivalent(),
                "same-seed public logits diverged: seed={seed:#x} len={len} report={report:?}"
            );
        }
    }
}

#[test]
fn external_fixture_detects_perturbed_inference_logits() {
    let cfg = Config {
        vocab: 7,
        n_embd: 4,
        n_head: 1,
        n_layer: 1,
        block: 3,
        n_ff: 8,
    };
    let mut rng = StdRng::seed_from_u64(0xA11CE_29FF);
    let gpt = Gpt::new(cfg, &mut rng);
    let x = [0, 1, 2];
    let y = [1, 2, 3];

    let mut logits = gpt.logits(&x);
    let mut grads = vec![0.0; gpt.collect_params().len()];
    let training_loss = gpt.backward_into(&x, &y, &mut grads);

    logits[2 * cfg.vocab + y[2]] += 0.5;
    let perturbed_loss = cross_entropy_from_logits(&logits, &y, cfg.vocab);
    let report = compare_f32_slices(
        &[training_loss],
        &[perturbed_loss],
        LOSS_TOLERANCE,
    );

    assert!(
        !report.is_equivalent(),
        "fixture failed to detect a deliberate inference-logit perturbation"
    );
}
