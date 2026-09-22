use auralis::checkpoint;
use auralis::gradcheck;
use auralis::model::{Config, Gpt, NormalizationKind};
use auralis::position::PositionKind;
use auralis::optim::Adam;
use auralis::tokenizer::{AnyTok, CharTokenizer};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs;
use std::io::ErrorKind;

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

#[test]
fn grouped_kv_checkpoint_roundtrip_infers_compact_layout_and_adam() {
    let tok = AnyTok::Char(CharTokenizer::fit("abc abc\n"));
    let cfg = Config {
        vocab: tok.vocab_size(),
        n_embd: 8,
        n_head: 4,
        n_layer: 1,
        block: 4,
        n_ff: 16,
    };
    let mut rng = StdRng::seed_from_u64(140);
    let mut gpt = Gpt::new_with_attention_heads(
        cfg,
        NormalizationKind::LayerNorm,
        PositionKind::LearnedAbsolute,
        1,
        &mut rng,
    );
    let mut params = gpt.collect_params();
    let mut grads = vec![0.0; params.len()];
    let ids = tok.encode("abc a");
    let _ = gpt.backward_into(&ids[..4], &ids[1..5], &mut grads);

    let mut adam = Adam::new(params.len(), 3e-3);
    adam.step(&mut params, &grads);
    gpt.write_params(&params);

    let path = std::env::temp_dir().join(format!(
        "auralis-genesis-mqa-checkpoint-{}.bin",
        std::process::id()
    ));
    checkpoint::save_full(&path, &gpt, &tok, Some(&adam)).unwrap();
    let (loaded, _, loaded_adam) = checkpoint::load_full(&path).unwrap();
    let _ = fs::remove_file(&path);

    assert_eq!(loaded.cfg, cfg);
    assert_eq!(loaded.n_kv_head(), 1);
    assert_eq!(loaded.collect_params(), gpt.collect_params());
    let loaded_adam = loaded_adam.unwrap();
    assert_eq!(loaded_adam.export(), adam.export());
}

#[test]
fn incompatible_checkpoint_is_an_error_not_a_panic() {
    // Minimal AURLIS02 header with a valid model/tokenizer configuration but
    // a deliberately impossible parameter count. The loader must reject it
    // before attempting to write the vector into the model.
    let path = std::env::temp_dir().join(format!(
        "auralis-genesis-invalid-checkpoint-{}.bin",
        std::process::id()
    ));

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"AURLIS02");
    for n in [5u32, 8, 2, 1, 4, 16] {
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    bytes.extend_from_slice(&0u32.to_le_bytes()); // char tokenizer
    let vocab = "abc\n ".as_bytes();
    bytes.extend_from_slice(&(vocab.len() as u32).to_le_bytes());
    bytes.extend_from_slice(vocab);
    bytes.extend_from_slice(&1u32.to_le_bytes()); // wrong parameter count
    fs::write(&path, bytes).unwrap();

    let err = match checkpoint::load_full(&path) {
        Ok(_) => panic!("incompatible checkpoint was accepted"),
        Err(err) => err,
    };
    let _ = fs::remove_file(&path);

    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("parameter count mismatch"));
}
