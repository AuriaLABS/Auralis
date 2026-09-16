//! Property tests for shapes, causal masks and run/model configs.
//!
//! Generators are seeded and bounded so CI stays cheap. Failures shrink toward
//! a smaller witness that can be copied into `tests/property_corpus.md`.

use auralis::kernels::{
    attention_forward_reference_into, matmul_reference_into,
};
use auralis::model::Config;
use auralis::run_config::RunConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SEED: u64 = 0xA11CE_0107;
const TRIALS: usize = 64;

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        assert!(n > 0);
        (self.next_u32() as usize) % n
    }

    fn inclusive(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
}

fn gen_valid_config(rng: &mut Lcg) -> Config {
    let n_head = rng.inclusive(1, 4);
    let hd = rng.inclusive(1, 4);
    Config {
        vocab: rng.inclusive(2, 12),
        n_embd: n_head * hd,
        n_head,
        n_layer: rng.inclusive(1, 3),
        block: rng.inclusive(1, 8),
        n_ff: rng.inclusive(1, 16),
    }
}

fn gen_config(rng: &mut Lcg) -> Config {
    if rng.below(2) == 0 {
        gen_valid_config(rng)
    } else {
        Config {
            vocab: rng.below(6),
            n_embd: rng.below(9),
            n_head: rng.below(6),
            n_layer: rng.below(4),
            block: rng.below(6),
            n_ff: rng.below(8),
        }
    }
}

fn shrink_config(cfg: Config) -> Vec<Config> {
    let mut out = Vec::new();
    let mut push = |c: Config| {
        if c != cfg {
            out.push(c);
        }
    };
    push(Config { vocab: cfg.vocab.saturating_sub(1), ..cfg });
    push(Config { n_embd: cfg.n_embd.saturating_sub(1), ..cfg });
    push(Config { n_head: cfg.n_head.saturating_sub(1), ..cfg });
    push(Config { n_layer: cfg.n_layer.saturating_sub(1), ..cfg });
    push(Config { block: cfg.block.saturating_sub(1), ..cfg });
    push(Config { n_ff: cfg.n_ff.saturating_sub(1), ..cfg });
    if cfg.n_head > 0 {
        push(Config {
            n_embd: cfg.n_head * (cfg.n_embd / cfg.n_head).max(1),
            ..cfg
        });
    }
    out
}

fn shrink_until_stable(mut cfg: Config, pred: impl Fn(&Config) -> bool) -> Config {
    loop {
        let mut progressed = false;
        for candidate in shrink_config(cfg) {
            if pred(&candidate) {
                cfg = candidate;
                progressed = true;
                break;
            }
        }
        if !progressed {
            return cfg;
        }
    }
}

fn config_constructs(cfg: &Config) -> bool {
    let mut rng = StdRng::seed_from_u64(1);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = auralis::model::Gpt::new(*cfg, &mut rng);
    }))
    .is_ok()
}

fn oracle_config_ok(cfg: &Config) -> bool {
    cfg.vocab > 1
        && cfg.n_embd > 0
        && cfg.n_head > 0
        && cfg.n_layer > 0
        && cfg.block > 0
        && cfg.n_ff > 0
        && cfg.n_embd % cfg.n_head == 0
}

/// Lower-triangular causal mask in row-major `[t, t]`. `true` = allowed.
fn causal_mask(t: usize) -> Vec<bool> {
    let mut mask = vec![false; t * t];
    for i in 0..t {
        for j in 0..=i {
            mask[i * t + j] = true;
        }
    }
    mask
}

fn mask_is_causal(t: usize, mask: &[bool]) -> bool {
    if mask.len() != t * t {
        return false;
    }
    for i in 0..t {
        for j in 0..t {
            let allowed = mask[i * t + j];
            if j <= i && !allowed {
                return false;
            }
            if j > i && allowed {
                return false;
            }
        }
    }
    true
}

fn corrupt_mask(mask: &mut [bool], t: usize, rng: &mut Lcg) {
    if t < 2 {
        return;
    }
    let i = rng.below(t - 1);
    let j = rng.inclusive(i + 1, t - 1);
    mask[i * t + j] = true;
}

#[test]
fn config_validator_matches_oracle_and_shrinks() {
    let mut rng = Lcg::new(SEED);
    let mut seen_invalid = false;
    let mut seen_valid = false;
    for _ in 0..TRIALS {
        let cfg = gen_config(&mut rng);
        let expected = oracle_config_ok(&cfg);
        let actual = config_constructs(&cfg);
        if actual != expected {
            let witness = shrink_until_stable(cfg, |c| config_constructs(c) != oracle_config_ok(c));
            panic!("config validator disagreed after shrink: {witness:?}");
        }
        seen_valid |= expected;
        seen_invalid |= !expected;
    }
    assert!(seen_valid && seen_invalid, "generator must emit both classes");
}

#[test]
fn valid_tiny_configs_construct_a_model() {
    let mut rng = Lcg::new(SEED ^ 0x11);
    let mut rust_rng = StdRng::seed_from_u64(SEED);
    for _ in 0..8 {
        let mut cfg = gen_valid_config(&mut rng);
        cfg.n_layer = 1;
        cfg.n_embd = cfg.n_embd.min(8);
        cfg.n_head = cfg.n_head.min(cfg.n_embd);
        while cfg.n_embd % cfg.n_head != 0 {
            cfg.n_head -= 1;
        }
        let _ = auralis::model::Gpt::new(cfg, &mut rust_rng);
    }
}

#[test]
fn causal_masks_are_lower_triangular_and_corruption_is_detected() {
    let mut rng = Lcg::new(SEED ^ 0x22);
    for _ in 0..TRIALS {
        let t = rng.inclusive(1, 8);
        let mask = causal_mask(t);
        assert!(mask_is_causal(t, &mask), "valid causal mask rejected t={t}");
        if t >= 2 {
            let mut bad = mask;
            corrupt_mask(&mut bad, t, &mut rng);
            assert!(!mask_is_causal(t, &bad), "future-unmasked row accepted t={t}");
        }
    }
}

#[test]
fn attention_probs_respect_causal_mask() {
    let mut rng = Lcg::new(SEED ^ 0x33);
    for _ in 0..16 {
        let n_head = rng.inclusive(1, 4);
        let hd = rng.inclusive(1, 3);
        let d = n_head * hd;
        let t = rng.inclusive(1, 6);
        let mut q = vec![0.0; t * d];
        let mut k = vec![0.0; t * d];
        let mut v = vec![0.0; t * d];
        for x in q.iter_mut().chain(k.iter_mut()).chain(v.iter_mut()) {
            *x = (rng.below(21) as f32 - 10.0) / 7.0;
        }
        let mut out = vec![0.0; t * d];
        let mut probs = vec![0.0; n_head * t * t];
        attention_forward_reference_into(&q, &k, &v, t, d, n_head, &mut out, &mut probs);

        for h in 0..n_head {
            for i in 0..t {
                let mut sum = 0.0f32;
                for j in 0..t {
                    let p = probs[(h * t + i) * t + j];
                    if j > i {
                        assert_eq!(p, 0.0, "future mass at h={h} i={i} j={j}");
                    } else {
                        assert!(p.is_finite());
                        sum += p;
                    }
                }
                assert!((sum - 1.0).abs() < 1e-5, "causal row sum {sum} h={h} i={i}");
            }
        }
    }
}

#[test]
fn matmul_accepts_matching_shapes_and_rejects_mismatches() {
    let mut rng = Lcg::new(SEED ^ 0x44);
    for _ in 0..TRIALS {
        let rows = rng.inclusive(1, 5);
        let inner = rng.inclusive(1, 5);
        let cols = rng.inclusive(1, 5);
        let a = vec![1.0; rows * inner];
        let b = vec![1.0; inner * cols];
        let mut out = vec![0.0; rows * cols];
        matmul_reference_into(&a, rows, inner, &b, cols, &mut out);

        let boom = std::panic::catch_unwind(|| {
            let mut bad = vec![0.0; rows * cols];
            matmul_reference_into(&a, rows, inner + 1, &b, cols, &mut bad);
        });
        assert!(boom.is_err(), "shape mismatch should panic");
    }
}

#[test]
fn run_config_properties_cover_zero_and_fraction_edges() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut seen_ok = false;
    let mut seen_err = false;
    for _ in 0..TRIALS {
        let mut cfg = RunConfig::default();
        match rng.gen_range(0..6) {
            0 => cfg.batch_size = 0,
            1 => cfg.gradient_accumulation_steps = 0,
            2 => cfg.learning_rate = if rng.gen_bool(0.5) { 0.0 } else { f32::NAN },
            3 => cfg.grad_clip_norm = -1.0,
            4 => {
                cfg.train_fraction = 0.7;
                cfg.validation_fraction = 0.3;
            }
            _ => {
                cfg.batch_size = rng.gen_range(1..8);
                cfg.gradient_accumulation_steps = rng.gen_range(1..4);
            }
        }
        let ok = cfg.validate().is_ok();
        seen_ok |= ok;
        seen_err |= !ok;
        if ok {
            assert!(cfg.effective_batch_size().unwrap() > 0);
        }
    }
    assert!(seen_ok && seen_err);
}

#[test]
fn corpus_edge_cases_stay_pinned() {
    let zeros = Config {
        vocab: 0,
        n_embd: 0,
        n_head: 0,
        n_layer: 0,
        block: 0,
        n_ff: 0,
    };
    assert!(!config_constructs(&zeros));

    let one_token = Config {
        vocab: 1,
        n_embd: 4,
        n_head: 2,
        n_layer: 1,
        block: 1,
        n_ff: 1,
    };
    assert!(!config_constructs(&one_token));

    let indivisible = Config {
        vocab: 2,
        n_embd: 6,
        n_head: 4,
        n_layer: 1,
        block: 1,
        n_ff: 1,
    };
    assert!(!config_constructs(&indivisible));

    let min_ok = Config {
        vocab: 2,
        n_embd: 1,
        n_head: 1,
        n_layer: 1,
        block: 1,
        n_ff: 1,
    };
    assert!(config_constructs(&min_ok));

    assert!(mask_is_causal(1, &[true]));
    assert!(!mask_is_causal(2, &[true, true, true, true]));
}
