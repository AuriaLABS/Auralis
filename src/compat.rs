//! #80 v1 format freeze and fail-closed compatibility.

pub const COMPAT_SCHEMA_VERSION: u32 = 1;
pub const V1_CHECKPOINT: u32 = 1;
pub const V1_MANIFEST: u32 = 1;
pub const V1_SESSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Checkpoint,
    Manifest,
    Session,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompatError {
    UnknownFormat,
    UnsupportedVersion,
    Corrupt,
}

pub fn current(format: Format) -> u32 {
    match format {
        Format::Checkpoint => V1_CHECKPOINT,
        Format::Manifest => V1_MANIFEST,
        Format::Session => V1_SESSION,
    }
}

pub fn accept(format: Format, version: u32) -> Result<(), CompatError> {
    if version == 0 {
        return Err(CompatError::Corrupt);
    }
    if version == current(format) {
        return Ok(());
    }
    Err(CompatError::UnsupportedVersion)
}

pub fn migrate(format: Format, version: u32) -> Result<u32, CompatError> {
    accept(format, version)?;
    Ok(version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_loads_and_unknown_fails_closed() {
        assert!(accept(Format::Checkpoint, 1).is_ok());
        assert_eq!(
            accept(Format::Checkpoint, 2).unwrap_err(),
            CompatError::UnsupportedVersion
        );
        assert_eq!(
            accept(Format::Manifest, 0).unwrap_err(),
            CompatError::Corrupt
        );
        assert_eq!(migrate(Format::Session, 1).unwrap(), 1);
        assert!(migrate(Format::Session, 9).is_err());
    }
}
