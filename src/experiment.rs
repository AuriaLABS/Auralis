//! Reproducible experiment primitives for Auralis Foundation.
//!
//! Dataset partitioning happens before tokenizer fitting so validation/test do
//! not influence the learned BPE merges. This module deliberately has no
//! dependency on the model implementation.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitConfig {
    pub train: f32,
    pub validation: f32,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train: 0.90,
            validation: 0.05,
        }
    }
}

fn validate_split(cfg: SplitConfig) -> Result<(), &'static str> {
    if !cfg.train.is_finite()
        || !cfg.validation.is_finite()
        || cfg.train <= 0.0
        || cfg.validation <= 0.0
        || cfg.train + cfg.validation >= 1.0
    {
        return Err("invalid split ratios");
    }
    Ok(())
}

fn split_points(n: usize, cfg: SplitConfig) -> Result<(usize, usize), &'static str> {
    validate_split(cfg)?;
    if n < 3 {
        return Err("dataset needs at least three units");
    }

    let mut train_end = ((n as f64) * cfg.train as f64).floor() as usize;
    let mut val_end = train_end + ((n as f64) * cfg.validation as f64).floor() as usize;
    train_end = train_end.clamp(1, n - 2);
    val_end = val_end.clamp(train_end + 1, n - 1);
    Ok((train_end, val_end))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextSplit {
    pub train: String,
    pub validation: String,
    pub test: String,
}

impl TextSplit {
    pub fn total_chars(&self) -> usize {
        self.train.chars().count() + self.validation.chars().count() + self.test.chars().count()
    }
}

/// Deterministic contiguous split by Unicode scalar values, never by raw byte
/// offsets. The original text can be reconstructed exactly by concatenation.
pub fn split_text(text: &str, cfg: SplitConfig) -> Result<TextSplit, &'static str> {
    let n = text.chars().count();
    let (train_chars, val_chars) = split_points(n, cfg)?;

    let train_byte = byte_index_at_char(text, train_chars);
    let val_byte = byte_index_at_char(text, val_chars);
    Ok(TextSplit {
        train: text[..train_byte].to_string(),
        validation: text[train_byte..val_byte].to_string(),
        test: text[val_byte..].to_string(),
    })
}

fn byte_index_at_char(text: &str, char_index: usize) -> usize {
    if char_index == text.chars().count() {
        return text.len();
    }
    text.char_indices()
        .nth(char_index)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenSplit {
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
    pub test: Vec<usize>,
}

impl TokenSplit {
    pub fn total_len(&self) -> usize {
        self.train.len() + self.validation.len() + self.test.len()
    }
}

pub fn split_tokens(tokens: &[usize], cfg: SplitConfig) -> Result<TokenSplit, &'static str> {
    let (train_end, val_end) = split_points(tokens.len(), cfg)?;
    Ok(TokenSplit {
        train: tokens[..train_end].to_vec(),
        validation: tokens[train_end..val_end].to_vec(),
        test: tokens[val_end..].to_vec(),
    })
}

/// Stable non-cryptographic fingerprint (FNV-1a 64-bit) for experiment
/// manifests. It detects accidental dataset/config drift; it is not intended
/// as a security primitive.
pub fn fingerprint_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn fingerprint_tokens(tokens: &[usize]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &token in tokens {
        for b in (token as u64).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// SplitMix64 finalizer used as a stateless pseudo-random mapping.
///
/// Because the result depends only on `(seed, step, stream)`, checkpoint resume
/// can reproduce the same training-window sequence without saving RNG internals.
pub fn deterministic_u64(seed: u64, step: u64, stream: u64) -> u64 {
    let mut z = seed
        .wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(step.wrapping_add(1)))
        .wrapping_add(0xd1b54a32d192ed03u64.wrapping_mul(stream.wrapping_add(1)));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub fn deterministic_index(seed: u64, step: u64, stream: u64, upper: usize) -> usize {
    assert!(upper > 0, "upper bound must be positive");
    let r = deterministic_u64(seed, step, stream) as u128;
    ((r * upper as u128) >> 64) as usize
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExperimentIdentity {
    pub seed: u64,
    pub dataset_fingerprint: u64,
    pub train_tokens: usize,
    pub validation_tokens: usize,
    pub test_tokens: usize,
}

impl ExperimentIdentity {
    pub fn from_split(seed: u64, full_tokens: &[usize], split: &TokenSplit) -> Self {
        Self {
            seed,
            dataset_fingerprint: fingerprint_tokens(full_tokens),
            train_tokens: split.train.len(),
            validation_tokens: split.validation.len(),
            test_tokens: split.test.len(),
        }
    }

    pub fn from_raw(seed: u64, raw_text: &str, split: &TokenSplit) -> Self {
        Self {
            seed,
            dataset_fingerprint: fingerprint_bytes(raw_text.as_bytes()),
            train_tokens: split.train.len(),
            validation_tokens: split.validation.len(),
            test_tokens: split.test.len(),
        }
    }

    pub fn line(&self) -> String {
        format!(
            "seed={} dataset={:016x} train={} validation={} test={}",
            self.seed,
            self.dataset_fingerprint,
            self.train_tokens,
            self.validation_tokens,
            self.test_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_split_preserves_unicode_and_reconstructs_exactly() {
        let text = "Auralis αβγ 🤖 aprende\nsegunda línea\ntercera línea";
        let split = split_text(text, SplitConfig { train: 0.7, validation: 0.15 }).unwrap();
        assert_eq!(split.total_chars(), text.chars().count());
        assert_eq!(format!("{}{}{}", split.train, split.validation, split.test), text);
        assert!(!split.train.is_empty());
        assert!(!split.validation.is_empty());
        assert!(!split.test.is_empty());
    }

    #[test]
    fn token_split_is_complete_disjoint_and_deterministic() {
        let tokens: Vec<usize> = (0..100).collect();
        let a = split_tokens(&tokens, SplitConfig::default()).unwrap();
        let b = split_tokens(&tokens, SplitConfig::default()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.total_len(), tokens.len());
        let mut rebuilt = Vec::new();
        rebuilt.extend_from_slice(&a.train);
        rebuilt.extend_from_slice(&a.validation);
        rebuilt.extend_from_slice(&a.test);
        assert_eq!(rebuilt, tokens);
    }

    #[test]
    fn deterministic_index_is_reproducible_and_bounded() {
        let a: Vec<_> = (0..64).map(|step| deterministic_index(42, step, 0, 17)).collect();
        let b: Vec<_> = (0..64).map(|step| deterministic_index(42, step, 0, 17)).collect();
        assert_eq!(a, b);
        assert!(a.iter().all(|&x| x < 17));
        assert!(a.windows(2).any(|w| w[0] != w[1]));
    }

    #[test]
    fn fingerprints_change_when_data_changes() {
        let a = fingerprint_tokens(&[1, 2, 3, 4]);
        let b = fingerprint_tokens(&[1, 2, 3, 5]);
        assert_ne!(a, b);
        assert_eq!(a, fingerprint_tokens(&[1, 2, 3, 4]));
        assert_ne!(fingerprint_bytes(b"abc"), fingerprint_bytes(b"abd"));
    }

    #[test]
    fn identity_is_stable() {
        let tokens: Vec<usize> = (0..20).collect();
        let split = split_tokens(&tokens, SplitConfig { train: 0.8, validation: 0.1 }).unwrap();
        let id = ExperimentIdentity::from_split(7, &tokens, &split);
        assert_eq!(id.train_tokens + id.validation_tokens + id.test_tokens, 20);
        assert!(id.line().contains("seed=7"));
    }
}
