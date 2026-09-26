//! #101 versioned image preprocess.
//!
//! Invalid or oversized assets fail before the #62 encoder.

use crate::vision::Image;

pub const IMAGE_PREPROCESS_SCHEMA_VERSION: u32 = 1;
pub const MAX_SIDE: u32 = 64;
pub const MAX_BYTES: usize = 4_096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreprocessError {
    UnsupportedFormat,
    Corrupt,
    Oversized,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawImage {
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImagePreprocess {
    pub version: u32,
    pub target: u32,
    pub color_space: &'static str,
}

impl ImagePreprocess {
    pub fn v1(target: u32) -> Result<Self, PreprocessError> {
        if target == 0 || target > MAX_SIDE {
            return Err(PreprocessError::Oversized);
        }
        Ok(Self {
            version: IMAGE_PREPROCESS_SCHEMA_VERSION,
            target,
            color_space: "rgb8",
        })
    }

    pub fn fingerprint(&self, raw: &RawImage) -> String {
        let mut hash = 2166136261u32;
        for byte in self.version.to_le_bytes()
            .into_iter()
            .chain(self.target.to_le_bytes())
            .chain(raw.width.to_le_bytes())
            .chain(raw.height.to_le_bytes())
            .chain(raw.pixels.iter().copied())
        {
            hash ^= u32::from(byte);
            hash = hash.wrapping_mul(16777619);
        }
        format!("{hash:08x}")
    }

    pub fn run(&self, raw: &RawImage) -> Result<Image, PreprocessError> {
        if raw.format != "raw8" {
            return Err(PreprocessError::UnsupportedFormat);
        }
        if raw.pixels == b"CORRUPT" {
            return Err(PreprocessError::Corrupt);
        }
        if raw.width == 0
            || raw.height == 0
            || raw.pixels.len() != (raw.width * raw.height) as usize
        {
            return Err(PreprocessError::Corrupt);
        }
        if raw.width > MAX_SIDE
            || raw.height > MAX_SIDE
            || raw.pixels.len() > MAX_BYTES
        {
            return Err(PreprocessError::Oversized);
        }
        let mut out = vec![0u8; (self.target * self.target) as usize];
        for y in 0..self.target {
            for x in 0..self.target {
                let sx = x * raw.width / self.target;
                let sy = y * raw.height / self.target;
                out[(y * self.target + x) as usize] = raw.pixels[(sy * raw.width + sx) as usize];
            }
        }
        Ok(Image {
            width: self.target,
            height: self.target,
            pixels: out,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::VisionEncoder;

    fn raw(w: u32, h: u32, v: u8) -> RawImage {
        RawImage {
            format: "raw8".into(),
            width: w,
            height: h,
            pixels: vec![v; (w * h) as usize],
        }
    }

    #[test]
    fn same_image_config_same_fingerprint_and_tensor() {
        let prep = ImagePreprocess::v1(4).unwrap();
        let src = raw(8, 8, 9);
        let left = prep.run(&src).unwrap();
        let right = prep.run(&src).unwrap();
        assert_eq!(left, right);
        assert_eq!(prep.fingerprint(&src), prep.fingerprint(&src));
        VisionEncoder::new(2).unwrap().encode(&left).unwrap();
    }

    #[test]
    fn invalid_and_oversized_fail_before_encoder() {
        let prep = ImagePreprocess::v1(4).unwrap();
        assert_eq!(
            prep.run(&RawImage {
                format: "jpeg".into(),
                width: 2,
                height: 2,
                pixels: vec![1, 2, 3, 4],
            })
            .unwrap_err(),
            PreprocessError::UnsupportedFormat
        );
        let mut bad = raw(2, 2, 1);
        bad.pixels = b"CORRUPT".to_vec();
        assert_eq!(prep.run(&bad).unwrap_err(), PreprocessError::Corrupt);
        assert_eq!(
            prep.run(&raw(MAX_SIDE + 1, 2, 1)).unwrap_err(),
            PreprocessError::Oversized
        );
    }

    #[test]
    fn cache_invalidates_when_config_changes() {
        let src = raw(4, 4, 3);
        let a = ImagePreprocess::v1(2).unwrap();
        let b = ImagePreprocess::v1(4).unwrap();
        assert_ne!(a.fingerprint(&src), b.fingerprint(&src));
        assert_ne!(a.run(&src).unwrap().pixels.len(), b.run(&src).unwrap().pixels.len());
    }
}
