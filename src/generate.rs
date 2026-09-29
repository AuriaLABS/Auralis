//! #300 greedy generation contract.

pub const GENERATE_SCHEMA_VERSION: u32 = 1;
pub const MAX_CONTEXT: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GenerateError {
    Empty,
    Overflow,
}

pub fn greedy(prompt: &[u32], steps: usize) -> Result<Vec<u32>, GenerateError> {
    if prompt.is_empty() {
        return Err(GenerateError::Empty);
    }
    if prompt.len() + steps > MAX_CONTEXT {
        return Err(GenerateError::Overflow);
    }
    let mut out = prompt.to_vec();
    for _ in 0..steps {
        let next = out.iter().fold(1u32, |acc, t| acc.wrapping_mul(33).wrapping_add(*t)) % 256;
        out.push(next);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_prompt_same_tokens_and_overflow_fails() {
        let a = greedy(&[1, 2], 3).unwrap();
        let b = greedy(&[1, 2], 3).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
        assert_eq!(greedy(&[], 1).unwrap_err(), GenerateError::Empty);
        assert_eq!(
            greedy(&[1; MAX_CONTEXT], 1).unwrap_err(),
            GenerateError::Overflow
        );
    }
}
