//! Bucle cognitivo mínimo: contexto persistente + generación.

use crate::model::Gpt;
use crate::tokenizer::AnyTok;
use rand::rngs::ThreadRng;

pub struct Agent {
    pub memory: String,
    pub max_memory_chars: usize,
}

impl Agent {
    pub fn new() -> Self {
        Self {
            memory: String::new(),
            max_memory_chars: 400,
        }
    }

    pub fn remember(&mut self, turn: &str) {
        self.memory.push_str(turn);
        self.memory.push('\n');
        if self.memory.chars().count() > self.max_memory_chars {
            let drop_n = self.memory.chars().count() - self.max_memory_chars;
            self.memory = self.memory.chars().skip(drop_n).collect();
        }
    }

    pub fn reply(
        &mut self,
        user: &str,
        gpt: &Gpt,
        tok: &AnyTok,
        rng: &mut ThreadRng,
        n_tokens: usize,
    ) -> String {
        self.remember(&format!("Usuario: {user}"));
        let prompt = format!("{}Auralis: ", self.memory);
        let mut ids = tok.encode(&prompt);
        if ids.is_empty() {
            ids = tok.encode("Auralis");
        }
        let start = ids.len();
        gpt.generate(&mut ids, n_tokens, 0.85, rng);
        let raw = tok.decode(&ids[start..]);
        let cut = raw
            .split('\n')
            .next()
            .unwrap_or(&raw)
            .trim()
            .chars()
            .take(160)
            .collect::<String>();
        self.remember(&format!("Auralis: {cut}"));
        cut
    }
}

impl Default for Agent {
    fn default() -> Self {
        Self::new()
    }
}
