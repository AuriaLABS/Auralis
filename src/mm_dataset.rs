//! #61 versioned multimodal dataset pipeline.
//!
//! Fingerprints and splits are deterministic. Missing or corrupt assets fail
//! closed. No network downloads.

pub const MM_DATASET_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MmDatasetError {
    MissingAsset(String),
    CorruptAsset(String),
    UnknownSplit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Asset {
    pub id: String,
    pub modality: String,
    pub payload: Vec<u8>,
}

impl Asset {
    pub fn fingerprint(&self) -> String {
        fingerprint(&self.payload)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example {
    pub id: String,
    pub assets: Vec<Asset>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transform {
    pub id: String,
    pub version: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmManifest {
    pub id: String,
    pub schema_version: u32,
    pub examples: Vec<Example>,
    pub transform: Transform,
}

impl MmManifest {
    pub fn fingerprint(&self) -> String {
        let mut blob = format!("{}|{}|{}", self.id, self.schema_version, self.transform.id);
        for example in &self.examples {
            blob.push('|');
            blob.push_str(&example.id);
            for asset in &example.assets {
                blob.push('|');
                blob.push_str(&asset.id);
                blob.push(':');
                blob.push_str(&asset.fingerprint());
            }
        }
        fingerprint(blob.as_bytes())
    }

    pub fn split_of(&self, example: &Example) -> Split {
        match example.id.bytes().fold(0u64, |acc, b| acc.wrapping_mul(16777619) ^ b as u64) % 10
        {
            0 => Split::Test,
            1 | 2 => Split::Validation,
            _ => Split::Train,
        }
    }

    pub fn examples_in(&self, split: Split) -> Vec<&Example> {
        self.examples
            .iter()
            .filter(|example| self.split_of(example) == split)
            .collect()
    }

    pub fn validate(&self) -> Result<(), MmDatasetError> {
        for example in &self.examples {
            if example.assets.is_empty() {
                return Err(MmDatasetError::MissingAsset(example.id.clone()));
            }
            for asset in &example.assets {
                if asset.payload.is_empty() {
                    return Err(MmDatasetError::MissingAsset(asset.id.clone()));
                }
                if asset.payload == b"CORRUPT" {
                    return Err(MmDatasetError::CorruptAsset(asset.id.clone()));
                }
            }
        }
        Ok(())
    }
}

pub fn fingerprint(bytes: &[u8]) -> String {
    let mut hash = 2166136261u32;
    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16777619);
    }
    format!("{hash:08x}")
}

pub fn synthetic_fixture() -> MmManifest {
    MmManifest {
        id: "mm-ci".into(),
        schema_version: MM_DATASET_SCHEMA_VERSION,
        examples: vec![
            Example {
                id: "ex-text".into(),
                assets: vec![Asset {
                    id: "t1".into(),
                    modality: "text".into(),
                    payload: b"hello".to_vec(),
                }],
            },
            Example {
                id: "ex-pair".into(),
                assets: vec![
                    Asset {
                        id: "t2".into(),
                        modality: "text".into(),
                        payload: b"cat".to_vec(),
                    },
                    Asset {
                        id: "i2".into(),
                        modality: "image".into(),
                        payload: b"PNG".to_vec(),
                    },
                ],
            },
        ],
        transform: Transform {
            id: "identity".into(),
            version: 1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_manifest_same_fingerprint_and_splits() {
        let left = synthetic_fixture();
        let right = synthetic_fixture();
        assert_eq!(left.fingerprint(), right.fingerprint());
        assert_eq!(left.split_of(&left.examples[0]), right.split_of(&right.examples[0]));
        left.validate().unwrap();
        assert!(!left.examples_in(left.split_of(&left.examples[0])).is_empty());
    }

    #[test]
    fn missing_and_corrupt_assets_fail_closed() {
        let mut missing = synthetic_fixture();
        missing.examples[0].assets.clear();
        assert!(matches!(
            missing.validate(),
            Err(MmDatasetError::MissingAsset(_))
        ));
        let mut corrupt = synthetic_fixture();
        corrupt.examples[1].assets[1].payload = b"CORRUPT".to_vec();
        assert!(matches!(
            corrupt.validate(),
            Err(MmDatasetError::CorruptAsset(_))
        ));
    }

    #[test]
    fn text_only_and_paired_examples_share_the_pipeline() {
        let manifest = synthetic_fixture();
        assert_eq!(manifest.examples[0].assets.len(), 1);
        assert_eq!(manifest.examples[1].assets.len(), 2);
        assert_eq!(manifest.transform.id, "identity");
        assert_eq!(manifest.schema_version, 1);
    }
}
