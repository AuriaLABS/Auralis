//! Optional model integration for the #37 external-memory contract.
//!
//! Retrieval and fusion are deliberately separate. The model never writes to
//! external memory. Memory-off is defined as the unmodified model baseline.

use crate::memory::{ExternalMemory, MemoryQuery};
use std::time::Instant;

pub const MEMORY_INTEGRATION_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemoryFusion {
    LastHiddenMeanAdd { scale: f32 },
}

impl MemoryFusion {
    pub fn validate(self) -> Result<Self, String> {
        match self {
            Self::LastHiddenMeanAdd { scale } => {
                if !scale.is_finite() || scale <= 0.0 {
                    return Err("memory fusion scale must be finite and positive".into());
                }
            }
        }
        Ok(self)
    }

    pub fn scale(self) -> f32 {
        match self {
            Self::LastHiddenMeanAdd { scale } => scale,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::LastHiddenMeanAdd { .. } => "last_hidden_mean_add",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MemoryInferenceMode {
    Off,
    On {
        query: MemoryQuery,
        fusion: MemoryFusion,
    },
}

impl MemoryInferenceMode {
    pub fn off() -> Self {
        Self::Off
    }

    pub fn on(query: MemoryQuery, scale: f32) -> Self {
        Self::On {
            query,
            fusion: MemoryFusion::LastHiddenMeanAdd { scale },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryTrace {
    pub schema_version: u32,
    pub enabled: bool,
    pub memory_present: bool,
    pub memory_version: Option<u32>,
    pub memory_len: usize,
    pub memory_capacity: usize,
    pub query_ns: u64,
    pub hits: usize,
    pub record_ids: Vec<u64>,
    pub fused_width: usize,
    pub fusion: Option<&'static str>,
    pub fusion_scale: Option<f32>,
}

impl MemoryTrace {
    fn off() -> Self {
        Self {
            schema_version: MEMORY_INTEGRATION_SCHEMA_VERSION,
            enabled: false,
            memory_present: false,
            memory_version: None,
            memory_len: 0,
            memory_capacity: 0,
            query_ns: 0,
            hits: 0,
            record_ids: Vec::new(),
            fused_width: 0,
            fusion: None,
            fusion_scale: None,
        }
    }

    pub fn human(&self) -> String {
        let ids = self
            .record_ids
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "memory_integration | schema={} enabled={} present={} memory_version={} len={} capacity={} query_ns={} hits={} record_ids={} fused_width={} fusion={} scale={}",
            self.schema_version,
            self.enabled,
            self.memory_present,
            self.memory_version
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".into()),
            self.memory_len,
            self.memory_capacity,
            self.query_ns,
            self.hits,
            if ids.is_empty() { "none" } else { &ids },
            self.fused_width,
            self.fusion.unwrap_or("none"),
            self.fusion_scale
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".into()),
        )
    }

    pub fn json(&self) -> String {
        let ids = self
            .record_ids
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"schema_version\":{},\"enabled\":{},\"memory_present\":{},",
                "\"memory_version\":{},\"memory_len\":{},\"memory_capacity\":{},",
                "\"query_ns\":{},\"hits\":{},\"record_ids\":[{}],",
                "\"fused_width\":{},\"fusion\":{},\"fusion_scale\":{}}}"
            ),
            self.schema_version,
            self.enabled,
            self.memory_present,
            self.memory_version
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".into()),
            self.memory_len,
            self.memory_capacity,
            self.query_ns,
            self.hits,
            ids,
            self.fused_width,
            self.fusion
                .map(|v| format!("\"{v}\""))
                .unwrap_or_else(|| "null".into()),
            self.fusion_scale
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".into()),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryRetrieval {
    pub residual: Option<Vec<f32>>,
    pub trace: MemoryTrace,
}

pub fn retrieve_hidden_residual(
    memory: Option<&dyn ExternalMemory>,
    mode: &MemoryInferenceMode,
    width: usize,
) -> Result<MemoryRetrieval, String> {
    if width == 0 {
        return Err("memory integration width must be positive".into());
    }

    let MemoryInferenceMode::On { query, fusion } = mode else {
        return Ok(MemoryRetrieval {
            residual: None,
            trace: MemoryTrace::off(),
        });
    };
    let fusion = fusion.validate()?;

    let Some(memory) = memory else {
        return Ok(MemoryRetrieval {
            residual: None,
            trace: MemoryTrace {
                schema_version: MEMORY_INTEGRATION_SCHEMA_VERSION,
                enabled: true,
                memory_present: false,
                memory_version: None,
                memory_len: 0,
                memory_capacity: 0,
                query_ns: 0,
                hits: 0,
                record_ids: Vec::new(),
                fused_width: 0,
                fusion: Some(fusion.as_str()),
                fusion_scale: Some(fusion.scale()),
            },
        });
    };

    let started = Instant::now();
    let records = memory
        .query(query)
        .map_err(|e| format!("external memory query failed: {e}"))?;
    let query_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;

    let mut trace = MemoryTrace {
        schema_version: MEMORY_INTEGRATION_SCHEMA_VERSION,
        enabled: true,
        memory_present: true,
        memory_version: Some(memory.version()),
        memory_len: memory.len(),
        memory_capacity: memory.capacity(),
        query_ns,
        hits: records.len(),
        record_ids: records.iter().map(|record| record.id).collect(),
        fused_width: 0,
        fusion: Some(fusion.as_str()),
        fusion_scale: Some(fusion.scale()),
    };

    if records.is_empty() {
        return Ok(MemoryRetrieval {
            residual: None,
            trace,
        });
    }

    let mut residual = vec![0.0f32; width];
    for record in &records {
        if record.value.len() != width {
            return Err(format!(
                "memory record {} width mismatch: expected {}, got {}",
                record.id,
                width,
                record.value.len()
            ));
        }
        for (dst, &value) in residual.iter_mut().zip(&record.value) {
            *dst += value;
        }
    }
    let inv = 1.0 / records.len() as f32;
    for value in &mut residual {
        *value *= inv;
    }
    trace.fused_width = width;

    Ok(MemoryRetrieval {
        residual: Some(residual),
        trace,
    })
}

pub fn fuse_last_hidden(
    hidden: &mut [f32],
    positions: usize,
    width: usize,
    residual: &[f32],
    fusion: MemoryFusion,
) -> Result<(), String> {
    let fusion = fusion.validate()?;
    let expected = positions
        .checked_mul(width)
        .ok_or_else(|| "memory hidden shape overflow".to_string())?;
    if positions == 0 || width == 0 || hidden.len() != expected {
        return Err(format!(
            "memory hidden shape mismatch: positions={positions} width={width} len={}",
            hidden.len()
        ));
    }
    if residual.len() != width {
        return Err(format!(
            "memory residual width mismatch: expected {width}, got {}",
            residual.len()
        ));
    }

    let start = (positions - 1) * width;
    let scale = fusion.scale();
    for (dst, &value) in hidden[start..start + width].iter_mut().zip(residual) {
        *dst += scale * value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{
        InMemoryExternalMemory, MemoryClass, MemoryWrite,
    };

    #[test]
    fn off_does_not_require_memory_or_create_residual() {
        let result = retrieve_hidden_residual(None, &MemoryInferenceMode::off(), 4).unwrap();
        assert!(result.residual.is_none());
        assert!(!result.trace.enabled);
        assert_eq!(result.trace.query_ns, 0);
        assert_eq!(result.trace.hits, 0);
    }

    #[test]
    fn empty_or_missing_memory_is_defined() {
        let mode = MemoryInferenceMode::on(MemoryQuery::exact("facts", "key"), 1.0);
        let missing = retrieve_hidden_residual(None, &mode, 4).unwrap();
        assert!(missing.residual.is_none());
        assert!(missing.trace.enabled);
        assert!(!missing.trace.memory_present);

        let memory = InMemoryExternalMemory::new(4).unwrap();
        let empty = retrieve_hidden_residual(Some(&memory), &mode, 4).unwrap();
        assert!(empty.residual.is_none());
        assert_eq!(empty.trace.hits, 0);
        assert!(empty.trace.memory_present);
    }

    #[test]
    fn retrieval_is_deterministic_mean_and_traceable() {
        let mut memory = InMemoryExternalMemory::new(4).unwrap();
        memory
            .write(MemoryWrite::new(
                "facts",
                "same",
                vec![1.0, 2.0, 3.0, 4.0],
                MemoryClass::Persistent,
            ))
            .unwrap();
        memory
            .write(MemoryWrite::new(
                "facts",
                "same",
                vec![3.0, 4.0, 5.0, 6.0],
                MemoryClass::Persistent,
            ))
            .unwrap();

        let query = MemoryQuery {
            namespace: Some("facts".into()),
            key: Some("same".into()),
            class: None,
            limit: 2,
        };
        let mode = MemoryInferenceMode::on(query, 0.5);
        let a = retrieve_hidden_residual(Some(&memory), &mode, 4).unwrap();
        let b = retrieve_hidden_residual(Some(&memory), &mode, 4).unwrap();
        assert_eq!(a.residual, Some(vec![2.0, 3.0, 4.0, 5.0]));
        assert_eq!(a.trace.record_ids, vec![0, 1]);
        assert_eq!(a.trace.hits, 2);
        assert_eq!(a.residual, b.residual);
        assert_eq!(a.trace.record_ids, b.trace.record_ids);
    }

    #[test]
    fn bad_width_fails_before_fusion() {
        let mut memory = InMemoryExternalMemory::new(2).unwrap();
        memory
            .write(MemoryWrite::new(
                "facts",
                "bad",
                vec![1.0, 2.0],
                MemoryClass::Persistent,
            ))
            .unwrap();
        let mode = MemoryInferenceMode::on(MemoryQuery::exact("facts", "bad"), 1.0);
        let err = retrieve_hidden_residual(Some(&memory), &mode, 4).unwrap_err();
        assert!(err.contains("width mismatch"));
    }

    #[test]
    fn fusion_touches_only_last_hidden_row() {
        let mut hidden = vec![1.0f32; 8];
        fuse_last_hidden(
            &mut hidden,
            2,
            4,
            &[1.0, -1.0, 2.0, -2.0],
            MemoryFusion::LastHiddenMeanAdd { scale: 0.5 },
        )
        .unwrap();
        assert_eq!(&hidden[..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&hidden[4..], &[1.5, 0.5, 2.0, 0.0]);
    }
}
