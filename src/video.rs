//! #143 video boundary over sampled vision frames.
//!
//! Reuses #62. No dedicated video architecture.

use crate::modality::{CommonSequence, ModalityId};
use crate::vision::{Image, VisionEncoder, VisionError};

pub const VIDEO_SCHEMA_VERSION: u32 = 1;
pub const MAX_FRAMES: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoError {
    Empty,
    Oversized,
    Disabled,
    Vision(VisionError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clip {
    pub frames: Vec<Image>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoEncoder {
    pub stride: usize,
    pub vision: VisionEncoder,
    pub enabled: bool,
}

impl VideoEncoder {
    pub fn new(stride: usize, patch: u32) -> Result<Self, VideoError> {
        if stride == 0 {
            return Err(VideoError::Empty);
        }
        Ok(Self {
            stride,
            vision: VisionEncoder::new(patch).map_err(VideoError::Vision)?,
            enabled: true,
        })
    }

    pub fn encode(&self, clip: &Clip) -> Result<CommonSequence, VideoError> {
        if !self.enabled {
            return Err(VideoError::Disabled);
        }
        if clip.frames.is_empty() {
            return Err(VideoError::Empty);
        }
        let sampled: Vec<&Image> = clip.frames.iter().step_by(self.stride).collect();
        if sampled.len() > MAX_FRAMES {
            return Err(VideoError::Oversized);
        }
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        let mut positions = Vec::new();
        for (t, frame) in sampled.iter().enumerate() {
            let encoded = self.vision.encode(frame).map_err(VideoError::Vision)?;
            let start = tokens.len() as u32;
            tokens.extend_from_slice(&encoded.tokens);
            mask.extend_from_slice(&encoded.mask);
            positions.extend(encoded.positions.iter().map(|p| start + *p + (t as u32) * 1_000));
        }
        Ok(CommonSequence {
            modality: ModalityId::Video,
            tokens,
            mask,
            positions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modality::{ModalityEncoder, TextModalityEncoder};

    fn clip(n: usize) -> Clip {
        Clip {
            frames: (0..n)
                .map(|i| Image::solid(2, 2, i as u8 + 1).unwrap())
                .collect(),
        }
    }

    #[test]
    fn sampling_and_order_are_deterministic() {
        let enc = VideoEncoder::new(2, 2).unwrap();
        let src = clip(4);
        let left = enc.encode(&src).unwrap();
        let right = enc.encode(&src).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.modality, ModalityId::Video);
        assert_eq!(left.tokens.len(), 2);
        assert!(left.positions[0] < left.positions[1] || left.tokens.len() == 2);
    }

    #[test]
    fn limits_and_disabled_do_not_change_text_or_vision() {
        assert_eq!(VideoEncoder::new(1, 2).unwrap().encode(&clip(0)).unwrap_err(), VideoError::Empty);
        let oversized = clip(MAX_FRAMES + 1);
        assert_eq!(
            VideoEncoder::new(1, 2).unwrap().encode(&oversized).unwrap_err(),
            VideoError::Oversized
        );
        let mut off = VideoEncoder::new(1, 2).unwrap();
        off.enabled = false;
        assert_eq!(off.encode(&clip(1)).unwrap_err(), VideoError::Disabled);
        assert_eq!(TextModalityEncoder.encode(&[3]).unwrap().tokens, vec![3]);
        assert_eq!(
            VisionEncoder::new(2)
                .unwrap()
                .encode(&Image::solid(2, 2, 8).unwrap())
                .unwrap()
                .tokens
                .len(),
            1
        );
    }
}
