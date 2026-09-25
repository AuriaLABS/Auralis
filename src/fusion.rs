//! #64 reference multimodal fusion.
//!
//! Concatenate common sequences. Text-only is identity. A missing modality
//! is explicit. Cross-modal attention is gated by a mask.

use crate::modality::{CommonSequence, ModalityError, ModalityId};

pub const FUSION_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FusionError {
    MissingModality(ModalityId),
    InvalidShape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionBudget {
    pub units: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusedSequence {
    pub tokens: Vec<u32>,
    pub mask: Vec<bool>,
    pub positions: Vec<u32>,
    pub owners: Vec<ModalityId>,
    pub budget: FusionBudget,
}

impl FusedSequence {
    pub fn can_attend(&self, query: usize, key: usize) -> bool {
        self.mask[query] && self.mask[key]
    }
}

pub fn fuse(
    parts: &[Option<&CommonSequence>],
    required: &[ModalityId],
) -> Result<FusedSequence, FusionError> {
    for id in required {
        let present = parts.iter().flatten().any(|part| part.modality == *id);
        if !present {
            return Err(FusionError::MissingModality(*id));
        }
    }
    let mut fused = FusedSequence {
        tokens: Vec::new(),
        mask: Vec::new(),
        positions: Vec::new(),
        owners: Vec::new(),
        budget: FusionBudget { units: 0 },
    };
    for part in parts.iter().flatten() {
        part.validate().map_err(|_| FusionError::InvalidShape)?;
        fused.budget.units += part.tokens.len() as u32;
        fused.tokens.extend_from_slice(&part.tokens);
        fused.mask.extend_from_slice(&part.mask);
        fused.positions.extend_from_slice(&part.positions);
        fused.owners.extend(std::iter::repeat(part.modality).take(part.tokens.len()));
    }
    Ok(fused)
}

pub fn text_only(text: &CommonSequence) -> Result<FusedSequence, FusionError> {
    if text.modality != ModalityId::Text {
        return Err(FusionError::MissingModality(ModalityId::Text));
    }
    fuse(&[Some(text)], &[ModalityId::Text])
}

pub fn encode_error(err: ModalityError) -> FusionError {
    match err {
        ModalityError::LengthMismatch => FusionError::InvalidShape,
        ModalityError::UnknownModality => FusionError::InvalidShape,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modality::{ModalityEncoder, TextModalityEncoder};
    use crate::vision::{Image, VisionEncoder};

    fn text() -> CommonSequence {
        TextModalityEncoder.encode(&[1, 2, 3]).unwrap()
    }

    fn image() -> CommonSequence {
        VisionEncoder::new(2)
            .unwrap()
            .encode(&Image::solid(2, 2, 7).unwrap())
            .unwrap()
    }

    #[test]
    fn text_only_is_identity() {
        let src = text();
        let fused = text_only(&src).unwrap();
        assert_eq!(fused.tokens, src.tokens);
        assert_eq!(fused.owners, vec![ModalityId::Text; 3]);
        assert_eq!(fused.budget.units, 3);
    }

    #[test]
    fn missing_modality_is_explicit() {
        let err = fuse(&[Some(&text())], &[ModalityId::Text, ModalityId::Image]).unwrap_err();
        assert_eq!(err, FusionError::MissingModality(ModalityId::Image));
    }

    #[test]
    fn cross_modal_mask_positive_and_negative() {
        let fused = fuse(&[Some(&text()), Some(&image())], &[ModalityId::Text, ModalityId::Image])
            .unwrap();
        assert!(fused.can_attend(0, 3));
        let mut blocked = fused.clone();
        blocked.mask[3] = false;
        assert!(!blocked.can_attend(0, 3));
        assert_eq!(fused.budget.units, 4);
    }
}
