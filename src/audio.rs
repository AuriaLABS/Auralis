//! #63 reference audio encoder.
//!
//! Deterministic framed tokens into the #60 common sequence. No pretrained
//! ASR/TTS and no realtime streaming.

use crate::modality::{CommonSequence, ModalityId};

pub const AUDIO_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioError {
    InvalidRate,
    InvalidFrame,
    Empty,
    Disabled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waveform {
    pub sample_rate: u32,
    pub samples: Vec<i16>,
}

impl Waveform {
    pub fn tone(sample_rate: u32, samples: Vec<i16>) -> Result<Self, AudioError> {
        if sample_rate == 0 {
            return Err(AudioError::InvalidRate);
        }
        if samples.is_empty() {
            return Err(AudioError::Empty);
        }
        Ok(Self {
            sample_rate,
            samples,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioEncoder {
    pub frame: usize,
    pub enabled: bool,
}

impl AudioEncoder {
    pub fn new(frame: usize) -> Result<Self, AudioError> {
        if frame == 0 {
            return Err(AudioError::InvalidFrame);
        }
        Ok(Self {
            frame,
            enabled: true,
        })
    }

    pub fn encode(&self, wave: &Waveform) -> Result<CommonSequence, AudioError> {
        if !self.enabled {
            return Err(AudioError::Disabled);
        }
        if wave.sample_rate == 0 {
            return Err(AudioError::InvalidRate);
        }
        if wave.samples.is_empty() {
            return Err(AudioError::Empty);
        }
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        for chunk in wave.samples.chunks(self.frame) {
            let acc: i32 = chunk.iter().map(|s| i32::from(*s)).sum();
            tokens.push((acc / chunk.len() as i32) as u32);
            mask.push(chunk.len() == self.frame);
        }
        let n = tokens.len() as u32;
        Ok(CommonSequence {
            modality: ModalityId::Audio,
            tokens,
            mask,
            positions: (0..n).collect(),
        })
    }

    pub fn backward(&self, encoded: &CommonSequence) -> Result<Vec<u32>, AudioError> {
        encoded.validate().map_err(|_| AudioError::Empty)?;
        Ok(encoded.tokens.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modality::{ModalityEncoder, TextModalityEncoder};
    use crate::vision::{Image, VisionEncoder};

    #[test]
    fn same_audio_same_tokens_and_partial_frame_is_masked() {
        let enc = AudioEncoder::new(2).unwrap();
        let wave = Waveform::tone(16_000, vec![2, 4, 6]).unwrap();
        let left = enc.encode(&wave).unwrap();
        let right = enc.encode(&wave).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.tokens, vec![3, 6]);
        assert_eq!(left.mask, vec![true, false]);
        assert_eq!(left.modality, ModalityId::Audio);
        assert_eq!(enc.backward(&left).unwrap(), left.tokens);
    }

    #[test]
    fn invalid_rate_and_empty_fail() {
        assert_eq!(
            Waveform::tone(0, vec![1]).unwrap_err(),
            AudioError::InvalidRate
        );
        assert_eq!(
            Waveform::tone(16_000, vec![]).unwrap_err(),
            AudioError::Empty
        );
        assert_eq!(AudioEncoder::new(0).unwrap_err(), AudioError::InvalidFrame);
    }

    #[test]
    fn disabled_audio_does_not_change_text_or_vision() {
        let mut enc = AudioEncoder::new(2).unwrap();
        enc.enabled = false;
        assert_eq!(
            enc.encode(&Waveform::tone(8_000, vec![1, 2]).unwrap())
                .unwrap_err(),
            AudioError::Disabled
        );
        let text = TextModalityEncoder.encode(&[9]).unwrap();
        assert_eq!(text.tokens, vec![9]);
        let vision = VisionEncoder::new(2)
            .unwrap()
            .encode(&Image::solid(2, 2, 3).unwrap())
            .unwrap();
        assert_eq!(vision.tokens.len(), 1);
    }
}
