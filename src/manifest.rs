//! Strict, human-readable experiment manifests.
//!
//! Model weights alone are not enough to reproduce training. Foundation stores
//! the dataset identity, full run configuration, architecture, exact parameter
//! fingerprint and optimizer step next to each checkpoint so accidental drift
//! fails closed instead of silently becoming a different experiment.

use crate::model::Gpt;
use crate::run_config::RunConfig;
use crate::tokenizer::AnyTok;
use std::fs;
use std::path::{Path, PathBuf};

pub const MANIFEST_VERSION: u32 = 3;

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentManifest {
    pub version: u32,
    pub code_revision: String,
    pub seed: u64,
    pub dataset_fingerprint: u64,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f32,
    pub grad_clip_norm: f32,
    pub train_fraction: f32,
    pub validation_fraction: f32,
    pub bpe_merges: usize,
    pub global_step: u64,
    pub tokenizer_kind: String,
    pub vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block: usize,
    pub n_ff: usize,
    pub parameter_count: usize,
    pub parameter_fingerprint: u64,
}

impl ExperimentManifest {
    pub fn capture(
        gpt: &Gpt,
        tok: &AnyTok,
        run: &RunConfig,
        dataset_fingerprint: u64,
        global_step: u64,
    ) -> Self {
        let params = gpt.collect_params();
        Self {
            version: MANIFEST_VERSION,
            code_revision: build_revision().to_string(),
            seed: run.seed,
            dataset_fingerprint,
            batch_size: run.batch_size,
            gradient_accumulation_steps: run.gradient_accumulation_steps,
            learning_rate: run.learning_rate,
            grad_clip_norm: run.grad_clip_norm,
            train_fraction: run.train_fraction,
            validation_fraction: run.validation_fraction,
            bpe_merges: run.bpe_merges,
            global_step,
            tokenizer_kind: tok.kind().to_string(),
            vocab: gpt.cfg.vocab,
            n_embd: gpt.cfg.n_embd,
            n_head: gpt.cfg.n_head,
            n_layer: gpt.cfg.n_layer,
            block: gpt.cfg.block,
            n_ff: gpt.cfg.n_ff,
            parameter_count: params.len(),
            parameter_fingerprint: fingerprint_params(&params),
        }
    }

    pub fn validate_resume(
        &self,
        gpt: &Gpt,
        tok: &AnyTok,
        run: &RunConfig,
        dataset_fingerprint: u64,
    ) -> Result<(), String> {
        if self.version != MANIFEST_VERSION {
            return Err(format!(
                "manifest version {} is unsupported (expected {})",
                self.version, MANIFEST_VERSION
            ));
        }
        check("code_revision", self.code_revision.as_str(), build_revision())?;
        check("seed", self.seed, run.seed)?;
        check(
            "dataset_fingerprint",
            self.dataset_fingerprint,
            dataset_fingerprint,
        )?;
        check("batch_size", self.batch_size, run.batch_size)?;
        check(
            "gradient_accumulation_steps",
            self.gradient_accumulation_steps,
            run.gradient_accumulation_steps,
        )?;
        check_f32("learning_rate", self.learning_rate, run.learning_rate)?;
        check_f32("grad_clip_norm", self.grad_clip_norm, run.grad_clip_norm)?;
        check_f32("train_fraction", self.train_fraction, run.train_fraction)?;
        check_f32(
            "validation_fraction",
            self.validation_fraction,
            run.validation_fraction,
        )?;
        check("bpe_merges", self.bpe_merges, run.bpe_merges)?;
        check("tokenizer_kind", self.tokenizer_kind.as_str(), tok.kind())?;
        check("vocab", self.vocab, gpt.cfg.vocab)?;
        check("n_embd", self.n_embd, gpt.cfg.n_embd)?;
        check("n_head", self.n_head, gpt.cfg.n_head)?;
        check("n_layer", self.n_layer, gpt.cfg.n_layer)?;
        check("block", self.block, gpt.cfg.block)?;
        check("n_ff", self.n_ff, gpt.cfg.n_ff)?;

        let params = gpt.collect_params();
        check("parameter_count", self.parameter_count, params.len())?;
        check(
            "parameter_fingerprint",
            self.parameter_fingerprint,
            fingerprint_params(&params),
        )?;
        Ok(())
    }

    pub fn encode(&self) -> String {
        format!(
            concat!(
                "auralis_manifest={}\n",
                "code_revision={}\n",
                "seed={}\n",
                "dataset_fingerprint={:016x}\n",
                "batch_size={}\n",
                "gradient_accumulation_steps={}\n",
                "learning_rate={}\n",
                "grad_clip_norm={}\n",
                "train_fraction={}\n",
                "validation_fraction={}\n",
                "bpe_merges={}\n",
                "global_step={}\n",
                "tokenizer_kind={}\n",
                "vocab={}\n",
                "n_embd={}\n",
                "n_head={}\n",
                "n_layer={}\n",
                "block={}\n",
                "n_ff={}\n",
                "parameter_count={}\n",
                "parameter_fingerprint={:016x}\n"
            ),
            self.version,
            self.code_revision,
            self.seed,
            self.dataset_fingerprint,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.learning_rate,
            self.grad_clip_norm,
            self.train_fraction,
            self.validation_fraction,
            self.bpe_merges,
            self.global_step,
            self.tokenizer_kind,
            self.vocab,
            self.n_embd,
            self.n_head,
            self.n_layer,
            self.block,
            self.n_ff,
            self.parameter_count,
            self.parameter_fingerprint,
        )
    }

    pub fn decode(text: &str) -> Result<Self, String> {
        fn value<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
            text.lines()
                .find_map(|line| line.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
                .ok_or_else(|| format!("manifest missing {key}"))
        }
        fn number<T: std::str::FromStr>(text: &str, key: &str) -> Result<T, String> {
            value(text, key)?
                .parse::<T>()
                .map_err(|_| format!("invalid manifest value for {key}"))
        }
        fn hex_u64(text: &str, key: &str) -> Result<u64, String> {
            u64::from_str_radix(value(text, key)?, 16)
                .map_err(|_| format!("invalid manifest value for {key}"))
        }

        Ok(Self {
            version: number(text, "auralis_manifest")?,
            code_revision: value(text, "code_revision")?.to_string(),
            seed: number(text, "seed")?,
            dataset_fingerprint: hex_u64(text, "dataset_fingerprint")?,
            batch_size: number(text, "batch_size")?,
            gradient_accumulation_steps: number(text, "gradient_accumulation_steps")?,
            learning_rate: number(text, "learning_rate")?,
            grad_clip_norm: number(text, "grad_clip_norm")?,
            train_fraction: number(text, "train_fraction")?,
            validation_fraction: number(text, "validation_fraction")?,
            bpe_merges: number(text, "bpe_merges")?,
            global_step: number(text, "global_step")?,
            tokenizer_kind: value(text, "tokenizer_kind")?.to_string(),
            vocab: number(text, "vocab")?,
            n_embd: number(text, "n_embd")?,
            n_head: number(text, "n_head")?,
            n_layer: number(text, "n_layer")?,
            block: number(text, "block")?,
            n_ff: number(text, "n_ff")?,
            parameter_count: number(text, "parameter_count")?,
            parameter_fingerprint: hex_u64(text, "parameter_fingerprint")?,
        })
    }
}

pub fn build_revision() -> &'static str {
    option_env!("AURALIS_COMMIT_SHA").unwrap_or("unknown")
}

pub fn fingerprint_params(params: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &value in params {
        for byte in value.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

pub fn manifest_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    let mut os = checkpoint.as_ref().as_os_str().to_os_string();
    os.push(".manifest");
    PathBuf::from(os)
}

pub fn save_manifest(
    checkpoint: impl AsRef<Path>,
    manifest: &ExperimentManifest,
) -> std::io::Result<PathBuf> {
    let path = manifest_path(checkpoint);
    fs::write(&path, manifest.encode())?;
    Ok(path)
}

pub fn load_manifest(checkpoint: impl AsRef<Path>) -> Result<ExperimentManifest, String> {
    let path = manifest_path(checkpoint);
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    ExperimentManifest::decode(&text)
}

fn check<T>(name: &str, expected: T, actual: T) -> Result<(), String>
where
    T: PartialEq + std::fmt::Display,
{
    if expected == actual {
        Ok(())
    } else {
        Err(format!(
            "resume mismatch for {name}: checkpoint={expected} requested={actual}"
        ))
    }
}

fn check_f32(name: &str, expected: f32, actual: f32) -> Result<(), String> {
    if expected.to_bits() == actual.to_bits() {
        Ok(())
    } else {
        Err(format!(
            "resume mismatch for {name}: checkpoint={expected} requested={actual}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::BpeTokenizer;
    use crate::model::Config;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn fixture(seed: u64) -> (Gpt, AnyTok) {
        let tok = AnyTok::Bpe(BpeTokenizer::fit("auralis auralis", 4));
        let cfg = Config {
            vocab: tok.vocab_size(),
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(seed);
        (Gpt::new(cfg, &mut rng), tok)
    }

    #[test]
    fn manifest_roundtrip_is_exact() {
        let (gpt, tok) = fixture(1);
        let run = RunConfig::default();
        let a = ExperimentManifest::capture(&gpt, &tok, &run, 0xdeadbeef, 13);
        let b = ExperimentManifest::decode(&a.encode()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn resume_validation_fails_when_experiment_changes() {
        let (gpt, tok) = fixture(2);
        let run = RunConfig::default();
        let manifest = ExperimentManifest::capture(&gpt, &tok, &run, 99, 0);

        let mut different_seed = run;
        different_seed.seed += 1;
        assert!(manifest
            .validate_resume(&gpt, &tok, &different_seed, 99)
            .is_err());

        let mut different_lr = run;
        different_lr.learning_rate *= 2.0;
        assert!(manifest
            .validate_resume(&gpt, &tok, &different_lr, 99)
            .is_err());

        let mut different_accum = run;
        different_accum.gradient_accumulation_steps = 2;
        assert!(manifest
            .validate_resume(&gpt, &tok, &different_accum, 99)
            .is_err());

        assert!(manifest.validate_resume(&gpt, &tok, &run, 100).is_err());
        assert!(manifest.validate_resume(&gpt, &tok, &run, 99).is_ok());
    }

    #[test]
    fn manifest_detects_weight_mismatch() {
        let (gpt, tok) = fixture(3);
        let run = RunConfig::default();
        let manifest = ExperimentManifest::capture(&gpt, &tok, &run, 77, 0);
        let mut changed = gpt.clone();
        let mut params = changed.collect_params();
        params[0] += 1e-3;
        changed.write_params(&params);
        assert!(manifest.validate_resume(&changed, &tok, &run, 77).is_err());
    }
}
