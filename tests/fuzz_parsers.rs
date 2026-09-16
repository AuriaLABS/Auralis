//! Bounded mutation tests for manifest and checkpoint parsers (#108).
//!
//! Not libFuzzer: a seeded mutator with a size cap so CI cannot OOM. Crashes
//! must become fixtures in `tests/fuzz_corpus.md`.

use auralis::checkpoint;
use auralis::manifest::ExperimentManifest;
use auralis::model::{Config, Gpt};
use auralis::tokenizer::{AnyTok, CharTokenizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

const SEED: u64 = 0xF022_0108;
const TRIALS: usize = 48;
const MAX_MUTANT: usize = 4_096;

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("auralis-fuzz-{name}-{}", std::process::id()))
}

fn seed_manifest() -> String {
    ExperimentManifest {
        version: 3,
        code_revision: "unknown".into(),
        seed: 1,
        dataset_fingerprint: 0xabc,
        batch_size: 2,
        gradient_accumulation_steps: 1,
        learning_rate: 3e-3,
        grad_clip_norm: 1.0,
        train_fraction: 0.9,
        validation_fraction: 0.05,
        bpe_merges: 8,
        global_step: 0,
        tokenizer_kind: "char".into(),
        vocab: 5,
        n_embd: 8,
        n_head: 2,
        n_layer: 1,
        block: 4,
        n_ff: 16,
        parameter_count: 32,
        parameter_fingerprint: 0x11,
    }
    .encode()
}

fn mutate(bytes: &[u8], rng: &mut StdRng) -> Vec<u8> {
    if bytes.is_empty() {
        return vec![rng.gen()];
    }
    let mut out = bytes.to_vec();
    match rng.gen_range(0..5) {
        0 => {
            let i = rng.gen_range(0..out.len());
            out[i] ^= 1 << rng.gen_range(0..8);
        }
        1 => {
            let i = rng.gen_range(0..out.len());
            out.truncate(i.max(1));
        }
        2 => {
            let i = rng.gen_range(0..=out.len());
            out.insert(i, rng.gen());
        }
        3 => out.extend_from_slice(b"\nunknown_field=1\n"),
        _ => {
            if out.len() >= 4 {
                out[..4].copy_from_slice(&rng.gen::<u32>().to_le_bytes());
            }
        }
    }
    if out.len() > MAX_MUTANT {
        out.truncate(MAX_MUTANT);
    }
    out
}

#[test]
fn seed_manifest_roundtrips() {
    let text = seed_manifest();
    let decoded = ExperimentManifest::decode(&text).unwrap();
    assert_eq!(decoded.encode(), text);
}

#[test]
fn mutated_manifests_fail_closed_or_stay_valid() {
    let seed = seed_manifest().into_bytes();
    let mut rng = StdRng::seed_from_u64(SEED);
    for case in 0..TRIALS {
        let mutant = mutate(&seed, &mut rng);
        let text = String::from_utf8_lossy(&mutant);
        let result = catch_unwind(AssertUnwindSafe(|| ExperimentManifest::decode(&text)));
        assert!(
            result.is_ok(),
            "manifest decoder panicked case={case} bytes={}",
            mutant.len()
        );
    }
}

#[test]
fn truncated_and_unknown_manifests_are_errors() {
    assert!(ExperimentManifest::decode("").is_err());
    assert!(ExperimentManifest::decode("auralis_manifest=3\n").is_err());
    let mut extra = seed_manifest();
    extra.push_str("not_a_field=1\n");
    let _ = ExperimentManifest::decode(&extra);
}

fn seed_checkpoint(path: &std::path::Path) {
    let tok = AnyTok::Char(CharTokenizer::fit("abc\n "));
    let cfg = Config {
        vocab: tok.vocab_size(),
        n_embd: 8,
        n_head: 2,
        n_layer: 1,
        block: 4,
        n_ff: 16,
    };
    let mut rng = StdRng::seed_from_u64(3);
    let gpt = Gpt::new(cfg, &mut rng);
    checkpoint::save(path, &gpt, &tok).unwrap();
}

#[test]
fn mutated_checkpoints_do_not_panic() {
    let path = tmp("seed.bin");
    seed_checkpoint(&path);
    let seed = fs::read(&path).unwrap();
    let _ = fs::remove_file(&path);
    assert!(!seed.is_empty());

    let mut rng = StdRng::seed_from_u64(SEED ^ 1);
    for case in 0..TRIALS {
        let mutant = mutate(&seed, &mut rng);
        let mutant_path = tmp(&format!("m{case}.bin"));
        fs::write(&mutant_path, &mutant).unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| checkpoint::load_full(&mutant_path)));
        let _ = fs::remove_file(&mutant_path);
        assert!(
            result.is_ok(),
            "checkpoint loader panicked case={case} size={}",
            mutant.len()
        );
    }
}

#[test]
fn oversized_string_length_is_rejected() {
    let mut bytes = Vec::from(&b"AURLIS02"[..]);
    for n in [5u32, 8, 2, 1, 4, 16] {
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    let path = tmp("huge-str.bin");
    fs::write(&path, bytes).unwrap();
    let err = match checkpoint::load_full(&path) {
        Ok(_) => panic!("oversized string was accepted"),
        Err(err) => err,
    };
    let _ = fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("exceeds"));
}

#[test]
fn garbage_magic_is_invalid_data() {
    let path = tmp("garbage.bin");
    fs::write(&path, b"NOTAURALIS").unwrap();
    let err = match checkpoint::load_full(&path) {
        Ok(_) => panic!("garbage magic was accepted"),
        Err(err) => err,
    };
    let _ = fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}
