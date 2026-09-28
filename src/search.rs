//! #78 budgeted architectural search.
//!
//! Deterministic grid only. No auto-merge and no unbounded NAS.

use crate::pareto::ResultVector;

pub const SEARCH_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchSpec {
    pub layers: Vec<u32>,
    pub heads: Vec<u32>,
    pub budget: usize,
    pub max_params: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub layers: u32,
    pub heads: u32,
}

impl Candidate {
    pub fn params(&self) -> u32 {
        self.layers * self.heads * 8
    }

    pub fn id(&self) -> String {
        format!("L{}H{}", self.layers, self.heads)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchError {
    EmptySpace,
    Invalid,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchRecord {
    pub tried: Vec<Candidate>,
    pub discarded: Vec<Candidate>,
}

pub fn enumerate(spec: &SearchSpec) -> Result<SearchRecord, SearchError> {
    if spec.layers.is_empty() || spec.heads.is_empty() || spec.budget == 0 {
        return Err(SearchError::EmptySpace);
    }
    let mut record = SearchRecord {
        tried: Vec::new(),
        discarded: Vec::new(),
    };
    for layers in &spec.layers {
        for heads in &spec.heads {
            if *layers == 0 || *heads == 0 {
                return Err(SearchError::Invalid);
            }
            let cand = Candidate {
                layers: *layers,
                heads: *heads,
            };
            if cand.params() > spec.max_params {
                record.discarded.push(cand);
                continue;
            }
            if record.tried.len() >= spec.budget {
                record.discarded.push(cand);
                continue;
            }
            if record.tried.iter().any(|seen| seen == &cand) {
                continue;
            }
            record.tried.push(cand);
        }
    }
    Ok(record)
}

pub fn as_vectors(record: &SearchRecord) -> Vec<ResultVector> {
    record
        .tried
        .iter()
        .map(|cand| ResultVector {
            id: cand.id(),
            quality: Some(cand.layers + cand.heads),
            cost: Some(cand.params()),
            memory: Some(cand.params()),
            latency: Some(cand.layers),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_spec_same_sequence_and_budget_is_respected() {
        let spec = SearchSpec {
            layers: vec![1, 2],
            heads: vec![2, 4],
            budget: 3,
            max_params: 64,
        };
        let left = enumerate(&spec).unwrap();
        let right = enumerate(&spec).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.tried.len(), 3);
        assert!(!left.discarded.is_empty());
        assert_eq!(as_vectors(&left).len(), 3);
    }

    #[test]
    fn invalid_and_over_budget_are_rejected_before_train() {
        assert_eq!(
            enumerate(&SearchSpec {
                layers: vec![0],
                heads: vec![1],
                budget: 4,
                max_params: 64,
            })
            .unwrap_err(),
            SearchError::Invalid
        );
        let over = enumerate(&SearchSpec {
            layers: vec![8],
            heads: vec![8],
            budget: 4,
            max_params: 16,
        })
        .unwrap();
        assert!(over.tried.is_empty());
        assert_eq!(over.discarded.len(), 1);
    }
}
