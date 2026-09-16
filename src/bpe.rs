//! BPE mínimo sobre caracteres Unicode, con entrenamiento determinista.
//!
//! Dos ejecuciones sobre el mismo texto deben producir exactamente los mismos
//! merges y el mismo vocabulario. Los caracteres no vistos durante `fit` no se
//! descartan: se representan mediante `<|unk|>`.

use std::collections::HashMap;

const UNK: &str = "<|unk|>";

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    pub itos: Vec<String>,
    pub stoi: HashMap<String, usize>,
    pub merges: Vec<(String, String)>,
}

impl BpeTokenizer {
    pub fn fit(text: &str, num_merges: usize) -> Self {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let mut merges = Vec::new();
        let mut vocab: Vec<String> = {
            let mut v: Vec<String> = tokens.clone();
            v.sort();
            v.dedup();
            v
        };

        for _ in 0..num_merges {
            let mut counts: HashMap<(String, String), usize> = HashMap::new();
            for w in tokens.windows(2) {
                *counts.entry((w[0].clone(), w[1].clone())).or_insert(0) += 1;
            }
            let best = counts.into_iter().max_by(|(pair_a, count_a), (pair_b, count_b)| {
                count_a.cmp(count_b).then_with(|| pair_b.cmp(pair_a))
            });
            let Some(((a, b), count)) = best else { break; };
            if count < 2 { break; }

            let merged = format!("{a}{b}");
            tokens = apply_merge(tokens, &a, &b, &merged);
            if !vocab.contains(&merged) { vocab.push(merged.clone()); }
            merges.push((a, b));
        }

        for extra in ["\n", " ", UNK] {
            if !vocab.iter().any(|s| s == extra) { vocab.push(extra.to_string()); }
        }
        vocab.sort();
        vocab.dedup();
        let stoi = vocab.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
        Self { itos: vocab, stoi, merges }
    }

    pub fn vocab_size(&self) -> usize { self.itos.len() }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        for (a, b) in &self.merges {
            let merged = format!("{a}{b}");
            tokens = apply_merge(tokens, a, b, &merged);
        }
        let unk = self.stoi.get(UNK).copied();
        tokens.into_iter().filter_map(|token| self.stoi.get(&token).copied().or(unk)).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().filter_map(|&i| self.itos.get(i).cloned()).collect()
    }
}

fn apply_merge(tokens: Vec<String>, a: &str, b: &str, merged: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
            out.push(merged.to_string());
            i += 2;
        } else {
            out.push(tokens[i].clone());
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{BpeTokenizer, UNK};

    #[test]
    fn bpe_roundtrip_and_merges() {
        let text = "abababab rust rust rust";
        let t = BpeTokenizer::fit(text, 8);
        assert!(!t.merges.is_empty());
        assert_eq!(t.decode(&t.encode(text)), text);
        assert!(t.vocab_size() >= 8);
    }

    #[test]
    fn fit_is_exactly_deterministic_even_with_frequency_ties() {
        let text = "abacabadabacaba";
        let first = BpeTokenizer::fit(text, 12);
        for _ in 0..32 {
            let next = BpeTokenizer::fit(text, 12);
            assert_eq!(next.itos, first.itos);
            assert_eq!(next.merges, first.merges);
            assert_eq!(next.encode(text), first.encode(text));
        }
    }

    #[test]
    fn unseen_characters_are_not_silently_dropped() {
        let t = BpeTokenizer::fit("abc abc", 4);
        let ids = t.encode("aΩc");
        assert_eq!(ids.len(), 3);
        let unk = t.stoi[UNK];
        assert_eq!(ids[1], unk);
        assert!(t.decode(&ids).contains(UNK));
    }

    #[test]
    fn legacy_vocab_without_unk_remains_readable() {
        let mut t = BpeTokenizer::fit("abc", 0);
        if let Some(unk) = t.stoi.remove(UNK) { t.itos[unk] = "<legacy-unused>".to_string(); }
        let ids = t.encode("aΩc");
        assert_eq!(ids.len(), 2);
    }
}
