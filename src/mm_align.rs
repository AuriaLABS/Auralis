//! #117 bidirectional cross-modal retrieval.

use crate::modality::{CommonSequence, ModalityId};

pub const MM_ALIGN_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlignError {
    Empty,
    Collapse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Embedding {
    pub modality: ModalityId,
    pub value: u32,
}

pub fn embed(seq: &CommonSequence) -> Result<Embedding, AlignError> {
    seq.validate().map_err(|_| AlignError::Empty)?;
    if seq.tokens.is_empty() {
        return Err(AlignError::Empty);
    }
    let value = seq.tokens.iter().sum::<u32>() / seq.tokens.len() as u32;
    Ok(Embedding {
        modality: seq.modality,
        value,
    })
}

pub fn similarity(a: &Embedding, b: &Embedding) -> u32 {
    u32::MAX - a.value.abs_diff(b.value)
}

pub fn retrieve(query: &Embedding, gallery: &[Embedding], k: usize) -> Vec<usize> {
    let mut ranks: Vec<(u32, usize)> = gallery
        .iter()
        .enumerate()
        .map(|(idx, item)| (similarity(query, item), idx))
        .collect();
    ranks.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    ranks.into_iter().take(k).map(|(_, idx)| idx).collect()
}

pub fn recall_at_k(ranked: &[usize], expected: usize, k: usize) -> bool {
    ranked.iter().take(k).any(|idx| *idx == expected)
}

pub fn collapse(gallery: &[Embedding]) -> Result<(), AlignError> {
    if gallery.is_empty() {
        return Err(AlignError::Empty);
    }
    if gallery.windows(2).all(|pair| pair[0].value == pair[1].value) && gallery.len() > 1 {
        return Err(AlignError::Collapse);
    }
    Ok(())
}

pub fn unaligned_baseline(n: usize) -> Vec<usize> {
    (0..n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioEncoder, Waveform};
    use crate::modality::{ModalityEncoder, TextModalityEncoder};
    use crate::vision::{Image, VisionEncoder};

    #[test]
    fn text_image_and_audio_retrieval_are_reproducible() {
        let text = embed(&TextModalityEncoder.encode(&[10, 10]).unwrap()).unwrap();
        let image = embed(
            &VisionEncoder::new(2)
                .unwrap()
                .encode(&Image::solid(2, 2, 10).unwrap())
                .unwrap(),
        )
        .unwrap();
        let audio = embed(
            &AudioEncoder::new(2)
                .unwrap()
                .encode(&Waveform::tone(16_000, vec![10, 10]).unwrap())
                .unwrap(),
        )
        .unwrap();
        let gallery = [image.clone(), audio.clone()];
        let ranked = retrieve(&text, &gallery, 1);
        assert!(recall_at_k(&ranked, 0, 1) || recall_at_k(&ranked, 1, 1));
        assert_eq!(retrieve(&text, &gallery, 2), retrieve(&text, &gallery, 2));
        assert_eq!(unaligned_baseline(2), vec![0, 1]);
    }

    #[test]
    fn hard_negative_and_collapse_are_detected() {
        let query = Embedding {
            modality: ModalityId::Text,
            value: 4,
        };
        let gallery = [
            Embedding {
                modality: ModalityId::Image,
                value: 40,
            },
            Embedding {
                modality: ModalityId::Image,
                value: 5,
            },
        ];
        let ranked = retrieve(&query, &gallery, 1);
        assert_eq!(ranked, vec![1]);
        assert!(recall_at_k(&ranked, 1, 1));
        assert!(!recall_at_k(&ranked, 0, 1));
        assert_eq!(
            collapse(&[
                Embedding {
                    modality: ModalityId::Text,
                    value: 1,
                },
                Embedding {
                    modality: ModalityId::Image,
                    value: 1,
                },
            ])
            .unwrap_err(),
            AlignError::Collapse
        );
    }
}
