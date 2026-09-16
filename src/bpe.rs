//! BPE mínimo sobre caracteres Unicode.
use std::collections::HashMap;

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
            let Some(((a, b), _)) = counts.into_iter().max_by_key(|(_, c)| *c) else {
                break;
            };
            if counts_pair(&tokens, &a, &b) < 2 {
                break;
            }
            let merged = format!("{a}{b}");
            tokens = apply_merge(tokens, &a, &b, &merged);
            if !vocab.contains(&merged) {
                vocab.push(merged.clone());
            }
            merges.push((a, b));
        }
        for extra in ["\n", " "] {
            if !vocab.iter().any(|s| s == extra) {
                vocab.push(extra.to_string());
            }
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
        tokens.into_iter().filter_map(|t| self.stoi.get(&t).copied()).collect()
    }
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().filter_map(|&i| self.itos.get(i).cloned()).collect()
    }
}

fn counts_pair(tokens: &[String], a: &str, b: &str) -> usize {
    tokens.windows(2).filter(|w| w[0] == a && w[1] == b).count()
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
    use super::BpeTokenizer;
    #[test]
    fn bpe_roundtrip_and_merges() {
        let text = "abababab rust rust rust";
        let t = BpeTokenizer::fit(text, 8);
        assert!(t.merges.len() >= 1);
        assert_eq!(t.decode(&t.encode(text)), text);
        assert!(t.vocab_size() >= 8);
    }
}
