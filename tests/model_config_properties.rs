use auralis::model::{Config, Gpt};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn valid_heads(n_embd: usize) -> Vec<usize> {
    [1usize, 2, 4, 8]
        .into_iter()
        .filter(|&heads| heads <= n_embd && n_embd % heads == 0)
        .collect()
}

fn generated_valid_configs() -> Vec<Config> {
    let embds = [4usize, 8, 12, 16];
    let vocabs = [2usize, 5, 11, 17];
    let mut configs = Vec::new();

    // Deterministic pseudo-property sweep. The case index is enough to reproduce
    // every failure without an external property-testing dependency.
    for case in 0..64usize {
        let n_embd = embds[(case * 5 + 1) % embds.len()];
        let heads = valid_heads(n_embd);
        let n_head = heads[(case * 3 + 2) % heads.len()];
        configs.push(Config {
            vocab: vocabs[(case * 7 + 3) % vocabs.len()],
            n_embd,
            n_head,
            n_layer: 1 + (case % 3),
            block: 1 + ((case * 11) % 6),
            n_ff: 3 + ((case * 13) % 23),
        });
    }
    configs
}

#[test]
fn generated_valid_configs_run_forward_and_generation() {
    for (case, cfg) in generated_valid_configs().into_iter().enumerate() {
        let seed = 0xA11CE_u64 ^ case as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let model = Gpt::new(cfg, &mut rng);

        let len = cfg.block.min(4);
        let x: Vec<usize> = (0..len).map(|i| (i * 3 + case) % cfg.vocab).collect();
        let y: Vec<usize> = (0..len)
            .map(|i| (i * 5 + case + 1) % cfg.vocab)
            .collect();
        let loss = model.loss(&x, &y);
        assert!(
            loss.is_finite() && loss >= 0.0,
            "case={case} cfg={cfg:?} seed={seed:#x} loss={loss}"
        );

        let mut generated = vec![x[0]];
        let mut sample_rng = StdRng::seed_from_u64(seed ^ 0x5EED);
        model.generate(&mut generated, 3, 0.0, &mut sample_rng);
        assert_eq!(generated.len(), 4, "case={case} cfg={cfg:?}");
        assert!(
            generated.iter().all(|&token| token < cfg.vocab),
            "case={case} cfg={cfg:?} generated={generated:?}"
        );
    }
}

#[test]
fn same_seed_and_config_reproduce_parameters_and_loss() {
    for (case, cfg) in generated_valid_configs().into_iter().take(16).enumerate() {
        let seed = 0xC0FFEE_u64 + case as u64;
        let mut rng_a = StdRng::seed_from_u64(seed);
        let mut rng_b = StdRng::seed_from_u64(seed);
        let a = Gpt::new(cfg, &mut rng_a);
        let b = Gpt::new(cfg, &mut rng_b);
        assert_eq!(a.collect_params(), b.collect_params(), "case={case} cfg={cfg:?}");

        let len = cfg.block.min(3);
        let x: Vec<usize> = (0..len).map(|i| (i + case) % cfg.vocab).collect();
        let y: Vec<usize> = (0..len).map(|i| (i + case + 1) % cfg.vocab).collect();
        assert_eq!(a.loss(&x, &y), b.loss(&x, &y), "case={case} cfg={cfg:?}");
    }
}

#[test]
fn invalid_config_families_fail_closed() {
    let invalid = [
        Config { vocab: 1, n_embd: 8, n_head: 2, n_layer: 1, block: 4, n_ff: 16 },
        Config { vocab: 5, n_embd: 0, n_head: 1, n_layer: 1, block: 4, n_ff: 16 },
        Config { vocab: 5, n_embd: 8, n_head: 0, n_layer: 1, block: 4, n_ff: 16 },
        Config { vocab: 5, n_embd: 8, n_head: 2, n_layer: 0, block: 4, n_ff: 16 },
        Config { vocab: 5, n_embd: 8, n_head: 2, n_layer: 1, block: 0, n_ff: 16 },
        Config { vocab: 5, n_embd: 8, n_head: 2, n_layer: 1, block: 4, n_ff: 0 },
        Config { vocab: 5, n_embd: 10, n_head: 4, n_layer: 1, block: 4, n_ff: 16 },
    ];

    for (case, cfg) in invalid.into_iter().enumerate() {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut rng = StdRng::seed_from_u64(case as u64);
            let _ = Gpt::new(cfg, &mut rng);
        }));
        assert!(result.is_err(), "invalid case={case} unexpectedly accepted: {cfg:?}");
    }
}

#[test]
fn invalid_token_indices_and_lengths_fail_closed() {
    let cfg = Config {
        vocab: 5,
        n_embd: 8,
        n_head: 2,
        n_layer: 1,
        block: 3,
        n_ff: 16,
    };
    let mut rng = StdRng::seed_from_u64(7);
    let model = Gpt::new(cfg, &mut rng);

    let out_of_vocab = catch_unwind(AssertUnwindSafe(|| model.loss(&[0, 5], &[1, 2])));
    assert!(out_of_vocab.is_err());

    let target_out_of_vocab = catch_unwind(AssertUnwindSafe(|| model.loss(&[0, 1], &[1, 5])));
    assert!(target_out_of_vocab.is_err());

    let too_long = catch_unwind(AssertUnwindSafe(|| model.loss(&[0, 1, 2, 3], &[1, 2, 3, 4])));
    assert!(too_long.is_err());

    let empty = catch_unwind(AssertUnwindSafe(|| model.loss(&[], &[])));
    assert!(empty.is_err());
}

#[test]
fn generation_empty_context_is_a_noop() {
    let cfg = Config {
        vocab: 5,
        n_embd: 8,
        n_head: 2,
        n_layer: 1,
        block: 3,
        n_ff: 16,
    };
    let mut rng = StdRng::seed_from_u64(9);
    let model = Gpt::new(cfg, &mut rng);
    let mut ids = Vec::new();
    model.generate(&mut ids, 10, 0.8, &mut rng);
    assert!(ids.is_empty());
}
