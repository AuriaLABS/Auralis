//! #62 reference visual encoder.
//!
//! Deterministic patch tokens into the #60 common sequence. No pretrained
//! vision model and no image generation.

use crate::modality::{CommonSequence, ModalityId};

pub const VISION_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VisionError {
    InvalidResolution,
    Corrupt,
    Disabled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Image {
    pub fn solid(width: u32, height: u32, value: u8) -> Result<Self, VisionError> {
        if width == 0 || height == 0 {
            return Err(VisionError::InvalidResolution);
        }
        Ok(Self {
            width,
            height,
            pixels: vec![value; (width * height) as usize],
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionEncoder {
    pub patch: u32,
    pub enabled: bool,
}

impl VisionEncoder {
    pub fn new(patch: u32) -> Result<Self, VisionError> {
        if patch == 0 {
            return Err(VisionError::InvalidResolution);
        }
        Ok(Self {
            patch,
            enabled: true,
        })
    }

    pub fn encode(&self, image: &Image) -> Result<CommonSequence, VisionError> {
        if !self.enabled {
            return Err(VisionError::Disabled);
        }
        if image.pixels.len() != (image.width * image.height) as usize {
            return Err(VisionError::Corrupt);
        }
        if image.width % self.patch != 0 || image.height % self.patch != 0 {
            return Err(VisionError::InvalidResolution);
        }
        let cols = image.width / self.patch;
        let rows = image.height / self.patch;
        let mut tokens = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let mut acc = 0u32;
                let mut count = 0u32;
                for dy in 0..self.patch {
                    for dx in 0..self.patch {
                        let x = col * self.patch + dx;
                        let y = row * self.patch + dy;
                        let idx = (y * image.width + x) as usize;
                        acc += u32::from(image.pixels[idx]);
                        count += 1;
                    }
                }
                tokens.push(acc / count);
            }
        }
        Ok(CommonSequence {
            modality: ModalityId::Image,
            mask: vec![true; tokens.len()],
            positions: (0..tokens.len() as u32).collect(),
            tokens,
        })
    }

    pub fn backward(&self, encoded: &CommonSequence) -> Result<Vec<u32>, VisionError> {
        encoded.validate().map_err(|_| VisionError::Corrupt)?;
        Ok(encoded.tokens.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modality::{ModalityEncoder, TextModalityEncoder};

    #[test]
    fn same_image_same_tokens() {
        let enc = VisionEncoder::new(2).unwrap();
        let image = Image::solid(4, 4, 10).unwrap();
        let left = enc.encode(&image).unwrap();
        let right = enc.encode(&image).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.tokens.len(), 4);
        assert_eq!(left.modality, ModalityId::Image);
        assert_eq!(enc.backward(&left).unwrap(), left.tokens);
    }

    #[test]
    fn invalid_resolution_and_corrupt_fail() {
        assert_eq!(
            Image::solid(0, 4, 1).unwrap_err(),
            VisionError::InvalidResolution
        );
        let enc = VisionEncoder::new(3).unwrap();
        let image = Image::solid(4, 4, 1).unwrap();
        assert_eq!(enc.encode(&image).unwrap_err(), VisionError::InvalidResolution);
        let bad = Image {
            width: 2,
            height: 2,
            pixels: vec![1],
        };
        assert_eq!(
            VisionEncoder::new(2).unwrap().encode(&bad).unwrap_err(),
            VisionError::Corrupt
        );
    }

    #[test]
    fn disabled_vision_does_not_change_text() {
        let mut enc = VisionEncoder::new(2).unwrap();
        enc.enabled = false;
        assert_eq!(
            enc.encode(&Image::solid(2, 2, 1).unwrap()).unwrap_err(),
            VisionError::Disabled
        );
        let text = TextModalityEncoder.encode(&[1, 2, 3]).unwrap();
        assert_eq!(text.tokens, vec![1, 2, 3]);
    }
}
