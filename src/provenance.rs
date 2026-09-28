//! #81 release artifact provenance.

pub const PROVENANCE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvenanceError {
    Empty,
    Secret,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Artifact {
    pub commit: String,
    pub target: String,
    pub features: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub schema_version: u32,
    pub commit: String,
    pub target: String,
    pub features: String,
    pub checksum: u64,
    pub size: usize,
}

pub fn checksum(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .enumerate()
        .fold(bytes.len() as u64, |acc, (i, b)| {
            acc.wrapping_mul(131).wrapping_add((*b as u64) + i as u64)
        })
}

pub fn record(art: &Artifact) -> Result<Provenance, ProvenanceError> {
    if art.commit.is_empty() || art.target.is_empty() || art.bytes.is_empty() {
        return Err(ProvenanceError::Empty);
    }
    let blob = format!("{} {} {}", art.commit, art.target, art.features);
    if blob.to_ascii_lowercase().contains("secret") || blob.contains("TOKEN=") {
        return Err(ProvenanceError::Secret);
    }
    Ok(Provenance {
        schema_version: PROVENANCE_SCHEMA_VERSION,
        commit: art.commit.clone(),
        target: art.target.clone(),
        features: art.features.clone(),
        checksum: checksum(&art.bytes),
        size: art.bytes.len(),
    })
}

pub fn compare(a: &Provenance, b: &Provenance) -> bool {
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn art(bytes: &[u8]) -> Artifact {
        Artifact {
            commit: "f4cbfdc7".into(),
            target: "x86_64-unknown-linux-gnu".into(),
            features: "default".into(),
            bytes: bytes.to_vec(),
        }
    }

    #[test]
    fn equivalent_builds_match_and_secrets_fail() {
        let left = record(&art(b"bin")).unwrap();
        let right = record(&art(b"bin")).unwrap();
        assert!(compare(&left, &right));
        assert_eq!(left.checksum, checksum(b"bin"));
        assert_eq!(
            record(&Artifact {
                commit: "abc".into(),
                target: "cpu".into(),
                features: "TOKEN=1".into(),
                bytes: vec![1],
            })
            .unwrap_err(),
            ProvenanceError::Secret
        );
        assert_eq!(record(&art(b"")).unwrap_err(), ProvenanceError::Empty);
    }
}
