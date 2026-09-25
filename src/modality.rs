//! #60 common modality contract.
//!
//! Text is the reference backend and must stay an identity. Visual/audio
//! encoders are out of scope.

pub const MODALITY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModalityId {
    Off,
    Text,
}

impl ModalityId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Text => "text",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModalityError {
    LengthMismatch,
    UnknownModality,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommonSequence {
    pub modality: ModalityId,
    pub tokens: Vec<u32>,
    pub mask: Vec<bool>,
    pub positions: Vec<u32>,
}

impl CommonSequence {
    pub fn validate(&self) -> Result<(), ModalityError> {
        if self.tokens.len() != self.mask.len() || self.tokens.len() != self.positions.len() {
            return Err(ModalityError::LengthMismatch);
        }
        Ok(())
    }
}

pub trait ModalityEncoder {
    fn id(&self) -> ModalityId;
    fn encode(&self, tokens: &[u32]) -> Result<CommonSequence, ModalityError>;
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextModalityEncoder;

impl ModalityEncoder for TextModalityEncoder {
    fn id(&self) -> ModalityId {
        ModalityId::Text
    }

    fn encode(&self, tokens: &[u32]) -> Result<CommonSequence, ModalityError> {
        Ok(CommonSequence {
            modality: ModalityId::Text,
            tokens: tokens.to_vec(),
            mask: vec![true; tokens.len()],
            positions: (0..tokens.len() as u32).collect(),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OffModalityEncoder;

impl ModalityEncoder for OffModalityEncoder {
    fn id(&self) -> ModalityId {
        ModalityId::Off
    }

    fn encode(&self, _tokens: &[u32]) -> Result<CommonSequence, ModalityError> {
        Ok(CommonSequence {
            modality: ModalityId::Off,
            tokens: Vec::new(),
            mask: Vec::new(),
            positions: Vec::new(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModalityManifest {
    pub schema_version: u32,
    pub adapters: Vec<&'static str>,
}

impl ModalityManifest {
    pub fn active(encoders: &[&dyn ModalityEncoder]) -> Self {
        Self {
            schema_version: MODALITY_SCHEMA_VERSION,
            adapters: encoders.iter().map(|enc| enc.id().as_str()).collect(),
        }
    }

    pub fn canonical(&self) -> String {
        format!(
            "schema={} adapters={}",
            self.schema_version,
            self.adapters.join(",")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_identity_and_off_is_empty() {
        let text = TextModalityEncoder.encode(&[7, 8, 9]).unwrap();
        assert_eq!(text.tokens, vec![7, 8, 9]);
        assert_eq!(text.positions, vec![0, 1, 2]);
        assert!(text.mask.iter().all(|bit| *bit));
        let off = OffModalityEncoder.encode(&[7, 8, 9]).unwrap();
        assert!(off.tokens.is_empty());
        assert_eq!(off.modality, ModalityId::Off);
    }

    #[test]
    fn invalid_shapes_fail_before_core() {
        let bad = CommonSequence {
            modality: ModalityId::Text,
            tokens: vec![1],
            mask: vec![true, false],
            positions: vec![0],
        };
        assert_eq!(bad.validate(), Err(ModalityError::LengthMismatch));
    }

    #[test]
    fn manifesto_records_active_adapters() {
        let text = TextModalityEncoder;
        let off = OffModalityEncoder;
        let manifest = ModalityManifest::active(&[&text, &off]);
        assert_eq!(manifest.adapters, vec!["text", "off"]);
        assert!(manifest.canonical().contains("schema=1"));
    }
}
