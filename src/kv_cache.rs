//! Explicit per-session KV cache for autoregressive inference.
//!
//! The cache stores only derived key/value activations. It is not model state,
//! is never part of checkpoints, and can be reset or cloned independently.

use crate::position::PositionKind;

pub const KV_CACHE_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug)]
struct LayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct KvCache {
    version: u32,
    layers: usize,
    width: usize,
    query_heads: usize,
    kv_heads: usize,
    capacity: usize,
    position: PositionKind,
    len: usize,
    data: Vec<LayerKvCache>,
}

impl KvCache {
    pub fn new(
        layers: usize,
        width: usize,
        capacity: usize,
        position: PositionKind,
    ) -> Result<Self, String> {
        Self::new_with_heads(layers, width, 1, 1, capacity, position)
    }

    pub fn new_with_heads(
        layers: usize,
        width: usize,
        query_heads: usize,
        kv_heads: usize,
        capacity: usize,
        position: PositionKind,
    ) -> Result<Self, String> {
        if layers == 0 || width == 0 || capacity == 0 || query_heads == 0 || kv_heads == 0 {
            return Err(format!(
                "kv cache requires positive shape: layers={layers} width={width} query_heads={query_heads} kv_heads={kv_heads} capacity={capacity}"
            ));
        }
        if kv_heads > query_heads || query_heads % kv_heads != 0 || width % kv_heads != 0 {
            return Err(format!(
                "kv cache requires compatible attention heads: width={width} query_heads={query_heads} kv_heads={kv_heads}"
            ));
        }
        let _ = layers
            .checked_mul(width)
            .and_then(|n| n.checked_mul(capacity))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| "kv cache shape size overflow".to_string())?;
        Ok(Self {
            version: KV_CACHE_SCHEMA_VERSION,
            layers,
            width,
            query_heads,
            kv_heads,
            capacity,
            position,
            len: 0,
            data: (0..layers)
                .map(|_| LayerKvCache {
                    keys: Vec::new(),
                    values: Vec::new(),
                })
                .collect(),
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn query_heads(&self) -> usize {
        self.query_heads
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn position_kind(&self) -> PositionKind {
        self.position
    }

    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.len)
    }

    pub fn logical_bytes(&self) -> usize {
        self.len
            .saturating_mul(self.layers)
            .saturating_mul(self.width)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub fn allocated_bytes(&self) -> usize {
        self.data
            .iter()
            .map(|layer| {
                layer
                    .keys
                    .capacity()
                    .saturating_add(layer.values.capacity())
                    .saturating_mul(std::mem::size_of::<f32>())
            })
            .sum()
    }

    pub fn reset(&mut self) {
        for layer in &mut self.data {
            layer.keys.clear();
            layer.values.clear();
        }
        self.len = 0;
    }

    pub fn validate_for(
        &self,
        layers: usize,
        width: usize,
        capacity: usize,
        position: PositionKind,
    ) -> Result<(), String> {
        if self.version != KV_CACHE_SCHEMA_VERSION {
            return Err(format!(
                "kv cache schema {} is unsupported (expected {})",
                self.version, KV_CACHE_SCHEMA_VERSION
            ));
        }
        if self.layers != layers
            || self.width != width
            || self.capacity != capacity
            || self.position != position
        {
            return Err(format!(
                "kv cache/model mismatch: cache=layers:{} width:{} capacity:{} position:{} model=layers:{} width:{} capacity:{} position:{}",
                self.layers,
                self.width,
                self.capacity,
                self.position.as_str(),
                layers,
                width,
                capacity,
                position.as_str(),
            ));
        }
        if self.data.len() != self.layers {
            return Err("kv cache layer storage mismatch".into());
        }
        if self.len > self.capacity {
            return Err("kv cache length exceeds capacity".into());
        }
        let expected = self
            .len
            .checked_mul(self.width)
            .ok_or_else(|| "kv cache active length overflow".to_string())?;
        for layer in &self.data {
            if layer.keys.len() != expected || layer.values.len() != expected {
                return Err("kv cache layer activation length mismatch".into());
            }
        }
        Ok(())
    }

    pub fn validate_for_heads(
        &self,
        layers: usize,
        width: usize,
        query_heads: usize,
        kv_heads: usize,
        capacity: usize,
        position: PositionKind,
    ) -> Result<(), String> {
        self.validate_for(layers, width, capacity, position)?;
        if self.query_heads != query_heads || self.kv_heads != kv_heads {
            return Err(format!(
                "kv cache attention-layout mismatch: cache=query_heads:{} kv_heads:{} model=query_heads:{} kv_heads:{}",
                self.query_heads, self.kv_heads, query_heads, kv_heads
            ));
        }
        Ok(())
    }

    pub(crate) fn history(&self, layer: usize) -> Result<(&[f32], &[f32]), String> {
        let layer = self
            .data
            .get(layer)
            .ok_or_else(|| "kv cache layer index outside range".to_string())?;
        Ok((&layer.keys, &layer.values))
    }

    pub(crate) fn commit(&mut self, staged: &[(Vec<f32>, Vec<f32>)]) -> Result<(), String> {
        if self.len >= self.capacity {
            return Err(format!(
                "kv cache is full: len={} capacity={}",
                self.len, self.capacity
            ));
        }
        if staged.len() != self.layers {
            return Err(format!(
                "kv cache staged layer count mismatch: expected {}, got {}",
                self.layers,
                staged.len()
            ));
        }
        for (keys, values) in staged {
            if keys.len() != self.width || values.len() != self.width {
                return Err(format!(
                    "kv cache staged width mismatch: expected {}, got k={} v={}",
                    self.width,
                    keys.len(),
                    values.len()
                ));
            }
        }
        for (layer, (keys, values)) in self.data.iter_mut().zip(staged) {
            layer.keys.extend_from_slice(keys);
            layer.values.extend_from_slice(values);
        }
        self.len += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_is_explicit_and_clone_is_independent() {
        let mut cache = KvCache::new(2, 4, 8, PositionKind::LearnedAbsolute).unwrap();
        assert_eq!(cache.version(), KV_CACHE_SCHEMA_VERSION);
        assert!(cache.is_empty());
        assert_eq!(cache.logical_bytes(), 0);

        cache
            .commit(&[
                (vec![1.0; 4], vec![2.0; 4]),
                (vec![3.0; 4], vec![4.0; 4]),
            ])
            .unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.logical_bytes(), 2 * 4 * 2 * 4);

        let mut cloned = cache.clone();
        cloned.reset();
        assert_eq!(cloned.len(), 0);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.history(1).unwrap().0, &[3.0; 4]);

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.logical_bytes(), 0);
    }

    #[test]
    fn shape_mismatch_and_overflow_fail_closed() {
        let cache = KvCache::new(2, 4, 2, PositionKind::Rope).unwrap();
        assert!(cache
            .validate_for(2, 4, 2, PositionKind::LearnedAbsolute)
            .is_err());
        assert!(KvCache::new(0, 4, 2, PositionKind::Rope).is_err());

        let mut cache = cache;
        let staged = [
            (vec![0.0; 4], vec![0.0; 4]),
            (vec![0.0; 4], vec![0.0; 4]),
        ];
        cache.commit(&staged).unwrap();
        cache.commit(&staged).unwrap();
        assert!(cache.commit(&staged).is_err());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn attention_layout_identity_fails_closed() {
        let cache = KvCache::new_with_heads(
            2,
            4,
            4,
            2,
            8,
            PositionKind::LearnedAbsolute,
        )
        .unwrap();
        cache
            .validate_for_heads(2, 4, 4, 2, 8, PositionKind::LearnedAbsolute)
            .unwrap();
        assert!(cache
            .validate_for_heads(2, 4, 2, 1, 8, PositionKind::LearnedAbsolute)
            .is_err());
        assert!(KvCache::new_with_heads(
            2,
            4,
            4,
            3,
            8,
            PositionKind::LearnedAbsolute,
        )
        .is_err());
    }

    #[test]
    fn failed_commit_does_not_partially_mutate_cache() {
        let mut cache = KvCache::new(2, 4, 4, PositionKind::Alibi).unwrap();
        let bad = [
            (vec![1.0; 4], vec![2.0; 4]),
            (vec![3.0; 3], vec![4.0; 4]),
        ];
        assert!(cache.commit(&bad).is_err());
        assert_eq!(cache.len(), 0);
        for layer in 0..2 {
            let (k, v) = cache.history(layer).unwrap();
            assert!(k.is_empty() && v.is_empty());
        }
    }
}
