use auralis::checkpoint;
use auralis::gradcheck;
use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use auralis::tokenizer::{AnyTok, CharTokenizer};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs;

#[test]
fn analytical_backward_matches_finite_differences() {
    let cfg = Config {
        vocab: 7,
        n_embd: 8,
        n_head: 2,
        n_layer: 1,
        block: 4,
        n_ff: 16,
    };
    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let gpt = Gpt::new(cfg, &mut rng);
    let x = [0, 1, 2, 3];
    let y = [1, 2, 3, 4];

    let report = gradcheck::check_random_params(&gpt, &x, &y, 24, 1e-3, &mut rng);
    assert!(
        report.ok_with(1e-3, 0.25),
        "gradient check failed: checked={} max_abs={} max_rel={} mean_rel={}",
        report.checked,
        report.max_abs_err,
        report.max_rel_err,
        report.mean_rel_err
    );
}

#[test]
fn tiny_transformer_actually_learns() {
    let cfg = Config {
        vocab: 4,
        n_embd: 8,
        n_head: 2,
        n_layer: 1,
        block: 8,
        n_ff: 16,
    };
    let mut rng = StdRng::seed_from_u64(7);
    let mut gpt = Gpt::new(cfg, &mut rng);
    let x = [0, 1, 2, 3, 0, 1, 2, 3];
    let y = [1, 2, 3, 0, 1, 2, 3, 0];

    let mut params = gpt.collect_params();
    let mut adam = Adam::new(params.len(), 1e-2);
    let mut grads = vec![0.0; params.len()];
    let initial = gpt.backward_into(&x, &y, &mut grads);

    for _ in 0..80 {
        grads.fill(0.0);
        let _ = gpt.backward_into(&x, &y, &mut grads);
        let norm = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        if norm > 1.0 {
            let scale = 1.0 / norm;
            for g in &mut grads {
                *g *= scale;
            }
        }
        adam.step(&mut params, &grads);
        gpt.write_params(&params);
    }

    grads.fill(0.0);
    let final_loss = gpt.backward_into(&x, &y, &mut grads);
    assert!(
        final_loss < initial * 0.70,
        "model did not learn enough: initial={initial:.6} final={final_loss:.6}"
    );
}

#[test]
fn checkpoint_roundtrip_preserves_model_tokenizer_and_adam() {
    let tok = AnyTok::Char(CharTokenizer::fit("abc abc\n"));
    let cfg = Config {
        vocab: tok.vocab_size(),
        n_embd: 8,
        n_head: 2,
        n_layer: 1,
        block: 4,
        n_ff: 16,
    };
    let mut rng = StdRng::seed_from_u64(99);
    let mut gpt = Gpt::new(cfg, &mut rng);
    let mut params = gpt.collect_params();
    let mut grads = vec![0.0; params.len()];
    let ids = tok.encode("abc a");
    let x = &ids[..4];
    let y = &ids[1..5];
    let _ = gpt.backward_into(x, y, &mut grads);

    let mut adam = Adam::new(params.len(), 3e-3);
    adam.step(&mut params, &grads);
    gpt.write_params(&params);

    let path = std::env::temp_dir().join(format!(
        "auralis-genesis-checkpoint-{}.bin",
        std::process::id()
    ));
    checkpoint::save_full(&path, &gpt, &tok, Some(&adam)).unwrap();
    let (loaded, loaded_tok, loaded_adam) = checkpoint::load_full(&path).unwrap();
    let _ = fs::remove_file(&path);

    assert_eq!(loaded.cfg, gpt.cfg);
    assert_eq!(loaded.collect_params(), gpt.collect_params());
    assert_eq!(loaded_tok.decode(&loaded_tok.encode("abc a")), "abc a");

    let loaded_adam = loaded_adam.expect("AURLIS03 must include Adam state");
    let (lr_a, t_a, m_a, v_a) = adam.export();
    let (lr_b, t_b, m_b, v_b) = loaded_adam.export();
    assert_eq!(lr_a, lr_b);
    assert_eq!(t_a, t_b);
    assert_eq!(m_a, m_b);
    assert_eq!(v_a, v_b);
}
