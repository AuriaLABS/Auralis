//! #102 deterministic multimodal collator.

use crate::modality::{CommonSequence, ModalityId};

pub const MM_COLLATOR_SCHEMA_VERSION: u32 = 1;
pub const MAX_BATCH_TOKENS: u32 = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CollatorError {
    Empty,
    Oom,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollatedBatch {
    pub tokens: Vec<Vec<u32>>,
    pub mask: Vec<Vec<bool>>,
    pub owners: Vec<Vec<ModalityId>>,
    pub pad_waste: u32,
}

impl CollatedBatch {
    pub fn token_count(&self) -> u32 {
        self.tokens.iter().map(|row| row.len() as u32).sum()
    }
}

pub fn collate(items: &[CommonSequence]) -> Result<CollatedBatch, CollatorError> {
    if items.is_empty() {
        return Err(CollatorError::Empty);
    }
    let width = items.iter().map(|item| item.tokens.len()).max().unwrap_or(0);
    let estimate = (items.len() * width) as u32;
    if estimate > MAX_BATCH_TOKENS {
        return Err(CollatorError::Oom);
    }
    let mut batch = CollatedBatch {
        tokens: Vec::new(),
        mask: Vec::new(),
        owners: Vec::new(),
        pad_waste: 0,
    };
    for item in items {
        item.validate().map_err(|_| CollatorError::Empty)?;
        let mut tokens = item.tokens.clone();
        let mut mask = item.mask.clone();
        let mut owners = vec![item.modality; item.tokens.len()];
        while tokens.len() < width {
            tokens.push(0);
            mask.push(false);
            owners.push(item.modality);
            batch.pad_waste += 1;
        }
        batch.tokens.push(tokens);
        batch.mask.push(mask);
        batch.owners.push(owners);
    }
    Ok(batch)
}

pub fn text_only(items: &[CommonSequence]) -> Result<CollatedBatch, CollatorError> {
    if items.iter().any(|item| item.modality != ModalityId::Text) {
        return Err(CollatorError::Empty);
    }
    collate(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioEncoder, Waveform};
    use crate::modality::{ModalityEncoder, TextModalityEncoder};
    use crate::vision::{Image, VisionEncoder};

    fn text(tokens: &[u32]) -> CommonSequence {
        TextModalityEncoder.encode(tokens).unwrap()
    }

    #[test]
    fn same_examples_same_batch_and_text_only_keeps_tokens() {
        let items = [text(&[1, 2]), text(&[3])];
        let left = text_only(&items).unwrap();
        let right = text_only(&items).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.tokens[0], vec![1, 2]);
        assert_eq!(left.tokens[1], vec![3, 0]);
        assert_eq!(left.mask[1], vec![true, false]);
        assert_eq!(left.pad_waste, 1);
    }

    #[test]
    fn mixed_modalities_have_combined_masks() {
        let vision = VisionEncoder::new(2)
            .unwrap()
            .encode(&Image::solid(2, 2, 4).unwrap())
            .unwrap();
        let audio = AudioEncoder::new(2)
            .unwrap()
            .encode(&Waveform::tone(16_000, vec![1, 2, 3]).unwrap())
            .unwrap();
        let batch = collate(&[text(&[9]), vision, audio]).unwrap();
        assert_eq!(batch.tokens.len(), 3);
        assert!(batch.mask.iter().any(|row| row.contains(&false)));
        assert!(batch.owners.iter().flatten().any(|id| *id == ModalityId::Image));
        assert!(batch.owners.iter().flatten().any(|id| *id == ModalityId::Audio));
    }

    #[test]
    fn oom_preflight_and_empty_fail() {
        assert_eq!(collate(&[]).unwrap_err(), CollatorError::Empty);
        let long = text(&[1; 33]);
        assert_eq!(collate(&[long.clone(), long]).unwrap_err(), CollatorError::Oom);
    }
}
