//! LoRA injection points
//!
//! Manages LoRA weights and their application to the render pipeline.

use std::collections::HashMap;

/// A loaded, ready-to-use LoRA
#[derive(Debug, Clone)]
pub struct LoadedLora {
    /// LoRA name/identifier
    pub name: String,
    /// Content hash
    pub hash: [u8; 32],
    /// Category (aesthetic, entity, effects, etc.)
    pub category: LoraCategory,
    /// Loaded weights
    pub weights: LoraWeights,
}

/// LoRA category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoraCategory {
    /// Overall visual style
    Aesthetic,
    /// Specific entity rendering
    Entity,
    /// Visual effects
    Effects,
    /// Environment rendering
    Environment,
    /// Color grading
    Color,
}

/// LoRA weights in memory
#[derive(Debug, Clone)]
pub struct LoraWeights {
    /// Rank/dimension of the LoRA
    pub rank: u32,
    /// Layer name → (A matrix, B matrix)
    pub layers: HashMap<String, LoraLayer>,
}

/// A single LoRA layer
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// A matrix (low-rank down projection)
    pub a: Vec<f32>,
    /// B matrix (low-rank up projection)
    pub b: Vec<f32>,
    /// Input dimension
    pub in_dim: usize,
    /// Output dimension
    pub out_dim: usize,
}

impl LoraLayer {
    /// Create a new LoRA layer
    pub fn new(in_dim: usize, out_dim: usize, rank: usize) -> Self {
        Self {
            a: vec![0.0; in_dim * rank],
            b: vec![0.0; rank * out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Get the rank of this layer
    pub fn rank(&self) -> usize {
        if self.in_dim == 0 {
            0
        } else {
            self.a.len() / self.in_dim
        }
    }

    /// Apply LoRA delta to base weights
    /// W' = W + scale * (A @ B)
    pub fn apply_to_weights(&self, base_weights: &mut [f32], scale: f32) {
        let rank = self.rank();
        if rank == 0 || base_weights.len() != self.in_dim * self.out_dim {
            return;
        }

        // Compute A @ B and add scaled result to base weights
        // This is a simplified version - production would use BLAS
        for out_idx in 0..self.out_dim {
            for in_idx in 0..self.in_dim {
                let mut delta = 0.0;
                for r in 0..rank {
                    // A[in_idx, r] * B[r, out_idx]
                    let a_val = self.a[in_idx * rank + r];
                    let b_val = self.b[r * self.out_dim + out_idx];
                    delta += a_val * b_val;
                }
                base_weights[in_idx * self.out_dim + out_idx] += scale * delta;
            }
        }
    }
}

impl LoraWeights {
    /// Create empty weights
    pub fn empty(rank: u32) -> Self {
        Self {
            rank,
            layers: HashMap::new(),
        }
    }

    /// Add a layer
    pub fn add_layer(&mut self, name: impl Into<String>, layer: LoraLayer) {
        self.layers.insert(name.into(), layer);
    }

    /// Get a layer by name
    pub fn get_layer(&self, name: &str) -> Option<&LoraLayer> {
        self.layers.get(name)
    }

    /// Estimate memory usage
    pub fn memory_size(&self) -> usize {
        self.layers
            .values()
            .map(|l| (l.a.len() + l.b.len()) * std::mem::size_of::<f32>())
            .sum()
    }
}

/// Active LoRA stack with weights
pub struct LoraStack {
    /// Ordered list of LoRAs to apply with their weights
    loras: Vec<(LoadedLora, f32)>,
}

impl LoraStack {
    /// Create an empty LoRA stack
    pub fn empty() -> Self {
        Self { loras: Vec::new() }
    }

    /// Add a LoRA to the stack
    pub fn push(&mut self, lora: LoadedLora, weight: f32) {
        self.loras.push((lora, weight.clamp(0.0, 1.0)));
    }

    /// Remove a LoRA by name
    pub fn remove(&mut self, name: &str) -> Option<LoadedLora> {
        if let Some(idx) = self.loras.iter().position(|(l, _)| l.name == name) {
            Some(self.loras.remove(idx).0)
        } else {
            None
        }
    }

    /// Clear all LoRAs
    pub fn clear(&mut self) {
        self.loras.clear();
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.loras.is_empty()
    }

    /// Get number of LoRAs
    pub fn len(&self) -> usize {
        self.loras.len()
    }

    /// Get LoRAs by category
    pub fn by_category(&self, category: LoraCategory) -> Vec<(&LoadedLora, f32)> {
        self.loras
            .iter()
            .filter(|(l, _)| l.category == category)
            .map(|(l, w)| (l, *w))
            .collect()
    }

    /// Apply LoRA modifications to a layer's weights
    pub fn apply_to_layer(&self, layer_name: &str, base_weights: &mut [f32]) {
        for (lora, weight) in &self.loras {
            if let Some(layer) = lora.weights.get_layer(layer_name) {
                layer.apply_to_weights(base_weights, *weight);
            }
        }
    }

    /// Get total LoRA weight for a category
    pub fn category_weight(&self, category: LoraCategory) -> f32 {
        self.loras
            .iter()
            .filter(|(l, _)| l.category == category)
            .map(|(_, w)| w)
            .sum()
    }

    /// Iterate over all LoRAs
    pub fn iter(&self) -> impl Iterator<Item = &(LoadedLora, f32)> {
        self.loras.iter()
    }
}

impl Default for LoraStack {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_layer() {
        let mut layer = LoraLayer::new(4, 4, 2);

        // Set up simple weights
        // A: 4x2, B: 2x4
        layer.a = vec![1.0; 8]; // All ones
        layer.b = vec![0.5; 8]; // All 0.5

        let mut base = vec![0.0; 16];
        layer.apply_to_weights(&mut base, 1.0);

        // Each output should be 2 * 1.0 * 0.5 = 1.0
        assert!((base[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_lora_stack() {
        let mut stack = LoraStack::empty();
        assert!(stack.is_empty());

        let lora = LoadedLora {
            name: "test".to_string(),
            hash: [0; 32],
            category: LoraCategory::Aesthetic,
            weights: LoraWeights::empty(16),
        };

        stack.push(lora, 0.5);
        assert_eq!(stack.len(), 1);

        let aesthetic = stack.by_category(LoraCategory::Aesthetic);
        assert_eq!(aesthetic.len(), 1);
        assert_eq!(aesthetic[0].1, 0.5);

        stack.remove("test");
        assert!(stack.is_empty());
    }

    #[test]
    fn test_lora_category_weight() {
        let mut stack = LoraStack::empty();

        stack.push(
            LoadedLora {
                name: "a".to_string(),
                hash: [0; 32],
                category: LoraCategory::Aesthetic,
                weights: LoraWeights::empty(16),
            },
            0.3,
        );

        stack.push(
            LoadedLora {
                name: "b".to_string(),
                hash: [1; 32],
                category: LoraCategory::Aesthetic,
                weights: LoraWeights::empty(16),
            },
            0.5,
        );

        stack.push(
            LoadedLora {
                name: "c".to_string(),
                hash: [2; 32],
                category: LoraCategory::Effects,
                weights: LoraWeights::empty(16),
            },
            0.7,
        );

        assert!((stack.category_weight(LoraCategory::Aesthetic) - 0.8).abs() < 0.001);
        assert!((stack.category_weight(LoraCategory::Effects) - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_lora_weights_memory() {
        let mut weights = LoraWeights::empty(16);
        weights.add_layer("test", LoraLayer::new(64, 64, 16));

        // Should have memory for both A and B matrices
        let expected_floats = 64 * 16 + 16 * 64; // 2048 floats
        let expected_bytes = expected_floats * 4;
        assert_eq!(weights.memory_size(), expected_bytes);
    }
}
