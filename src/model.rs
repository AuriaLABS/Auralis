//! Decoder-only Transformer from first principles.
//!
//! No deep-learning framework is used here: forward pass, causal multi-head
//! attention, layer normalization, GELU, cross-entropy and backward pass are
//! implemented explicitly over `Vec<f32>`.

use rand::Rng;
