//! Tokenizadores: caracteres y fachada común (char | BPE).

use crate::bpe::BpeTokenizer;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CharTokenizer {
    pub stoi: HashMap<char, usize>,
    pub itos: Vec<char>,
}

impl CharTokenizer {
    pub fn fit(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        for extra in ['\n', ' '] {
            if !chars.contains(&extra) {
                chars.push(extra);
            }
        }
        let itos = chars;
        let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self { stoi, itos }
    }
    pub fn vocab_size(&self) -> usize { self.itos.len() }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().filter_map(|c| self.stoi.get(&c).copied()).collect()
    }
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().filter_map(|&i| self.itos.get(i).copied()).collect()
    }
}

#[derive(Clone)]
pub enum AnyTok {
    Char(CharTokenizer),
    Bpe(BpeTokenizer),
}

impl AnyTok {
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            AnyTok::Char(t) => t.encode(text),
            AnyTok::Bpe(t) => t.encode(text),
        }
    }
    pub fn decode(&self, ids: &[usize]) -> String {
        match self {
            AnyTok::Char(t) => t.decode(ids),
            AnyTok::Bpe(t) => t.decode(ids),
        }
    }
    pub fn vocab_size(&self) -> usize {
        match self {
            AnyTok::Char(t) => t.vocab_size(),
            AnyTok::Bpe(t) => t.vocab_size(),
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            AnyTok::Char(_) => "char",
            AnyTok::Bpe(_) => "bpe",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CharTokenizer;
    #[test]
    fn roundtrip() {
        let t = CharTokenizer::fit("hola rust");
        assert_eq!(t.decode(&t.encode("hola")), "hola");
    }
}
