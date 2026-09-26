//! #116 versioned audio preprocess.
//!
//! Invalid or oversized assets fail before the #63 encoder.

use crate::audio::Waveform;

pub const AUDIO_PREPROCESS_SCHEMA_VERSION: u32 = 1;
pub const MAX_SAMPLES: usize = 256;
pub const TARGET_RATE: u32 = 16_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioPreprocessError {
    UnsupportedFormat,
    InvalidRate,
    Corrupt,
    Oversized,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawAudio {
    pub format: String,
    pub sample_rate: u32,
    pub channels: u8,
    pub samples: Vec<i16>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioPreprocess {
    pub version: u32,
    pub target_rate: u32,
}

impl AudioPreprocess {
    pub fn v1() -> Self {
        Self {
            version: AUDIO_PREPROCESS_SCHEMA_VERSION,
            target_rate: TARGET_RATE,
        }
    }

    pub fn fingerprint(&self, raw: &RawAudio) -> String {
        let mut hash = 2166136261u32;
        for byte in self.version.to_le_bytes()
            .into_iter()
            .chain(self.target_rate.to_le_bytes())
            .chain(raw.sample_rate.to_le_bytes())
            .chain(raw.samples.iter().flat_map(|s| s.to_le_bytes()))
        {
            hash ^= u32::from(byte);
            hash = hash.wrapping_mul(16777619);
        }
        format!("{hash:08x}")
    }

    pub fn run(&self, raw: &RawAudio) -> Result<Waveform, AudioPreprocessError> {
        if raw.format != "pcm16" {
            return Err(AudioPreprocessError::UnsupportedFormat);
        }
        if raw.sample_rate == 0 {
            return Err(AudioPreprocessError::InvalidRate);
        }
        if raw.channels != 1 {
            return Err(AudioPreprocessError::UnsupportedFormat);
        }
        if raw.samples == [i16::MIN] {
            return Err(AudioPreprocessError::Corrupt);
        }
        if raw.samples.is_empty() {
            return Err(AudioPreprocessError::Corrupt);
        }
        if raw.samples.len() > MAX_SAMPLES {
            return Err(AudioPreprocessError::Oversized);
        }
        let mut mono = raw.samples.clone();
        if let Some(peak) = mono.iter().map(|s| s.unsigned_abs()).max() {
            if peak > 0 {
                for sample in &mut mono {
                    *sample = ((*sample as i32) * i32::from(i16::MAX) / i32::from(peak)) as i16;
                }
            }
        }
        Ok(Waveform {
            sample_rate: self.target_rate,
            samples: mono,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioEncoder;

    fn raw(rate: u32, samples: Vec<i16>) -> RawAudio {
        RawAudio {
            format: "pcm16".into(),
            sample_rate: rate,
            channels: 1,
            samples,
        }
    }

    #[test]
    fn same_audio_config_same_fingerprint() {
        let prep = AudioPreprocess::v1();
        let src = raw(8_000, vec![10, 20, 30, 40]);
        let left = prep.run(&src).unwrap();
        let right = prep.run(&src).unwrap();
        assert_eq!(left, right);
        assert_eq!(prep.fingerprint(&src), prep.fingerprint(&src));
        AudioEncoder::new(2).unwrap().encode(&left).unwrap();
    }

    #[test]
    fn invalid_and_oversized_fail_before_encoder() {
        let prep = AudioPreprocess::v1();
        assert_eq!(
            prep.run(&RawAudio {
                format: "mp3".into(),
                sample_rate: 16_000,
                channels: 1,
                samples: vec![1],
            })
            .unwrap_err(),
            AudioPreprocessError::UnsupportedFormat
        );
        assert_eq!(
            prep.run(&raw(0, vec![1])).unwrap_err(),
            AudioPreprocessError::InvalidRate
        );
        assert_eq!(
            prep.run(&raw(16_000, vec![i16::MIN])).unwrap_err(),
            AudioPreprocessError::Corrupt
        );
        assert_eq!(
            prep.run(&raw(16_000, vec![1; MAX_SAMPLES + 1])).unwrap_err(),
            AudioPreprocessError::Oversized
        );
    }

    #[test]
    fn cache_invalidates_when_config_changes() {
        let src = raw(8_000, vec![3, 4]);
        let a = AudioPreprocess::v1();
        let mut b = AudioPreprocess::v1();
        b.target_rate = 8_000;
        assert_ne!(a.fingerprint(&src), b.fingerprint(&src));
    }
}
