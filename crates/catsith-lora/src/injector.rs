//! LoRA injection into models
//!
//! Handles applying LoRA weights to base model weights.

use std::collections::HashMap;

/// LoRA weights ready for injection
#[derive(Debug, Clone)]
pub struct LoraWeights {
    /// Rank of the LoRA
    pub rank: u32,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Layer weights: layer_name -> (A matrix, B matrix)
    pub layers: HashMap<String, LoraLayerWeights>,
}

/// Weights for a single layer
#[derive(Debug, Clone)]
pub struct LoraLayerWeights {
    /// A matrix (down projection)
    pub a: Vec<f32>,
    /// B matrix (up projection)
    pub b: Vec<f32>,
    /// Input dimension
    pub in_dim: usize,
    /// Output dimension
    pub out_dim: usize,
}

impl LoraLayerWeights {
    /// Create new layer weights
    pub fn new(in_dim: usize, out_dim: usize, rank: usize) -> Self {
        Self {
            a: vec![0.0; in_dim * rank],
            b: vec![0.0; rank * out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        if self.in_dim == 0 {
            return 0;
        }
        self.a.len() / self.in_dim
    }

    /// Compute the LoRA delta: A @ B
    pub fn compute_delta(&self) -> Vec<f32> {
        let rank = self.rank();
        let mut delta = vec![0.0; self.in_dim * self.out_dim];

        for i in 0..self.in_dim {
            for j in 0..self.out_dim {
                let mut sum = 0.0;
                for r in 0..rank {
                    sum += self.a[i * rank + r] * self.b[r * self.out_dim + j];
                }
                delta[i * self.out_dim + j] = sum;
            }
        }

        delta
    }
}

impl LoraWeights {
    /// Create empty LoRA weights
    pub fn empty(rank: u32, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            layers: HashMap::new(),
        }
    }

    /// Add a layer
    pub fn add_layer(&mut self, name: impl Into<String>, weights: LoraLayerWeights) {
        self.layers.insert(name.into(), weights);
    }

    /// Get a layer
    pub fn get_layer(&self, name: &str) -> Option<&LoraLayerWeights> {
        self.layers.get(name)
    }

    /// Parse from raw bytes (simple format)
    pub fn from_bytes(data: &[u8], rank: u32, alpha: f32) -> Option<Self> {
        // Simple format: just interpret as f32 array
        // Real implementation would parse a proper format (safetensors, etc.)

        if data.len() % 4 != 0 {
            return None;
        }

        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Create a dummy layer for testing
        let mut weights = Self::empty(rank, alpha);

        if floats.len() >= (64 * rank as usize + rank as usize * 64) {
            let mut layer = LoraLayerWeights::new(64, 64, rank as usize);
            layer.a = floats[..64 * rank as usize].to_vec();
            layer.b = floats[64 * rank as usize..64 * rank as usize + rank as usize * 64].to_vec();
            weights.add_layer("default", layer);
        }

        Some(weights)
    }

    /// Estimate memory usage
    pub fn memory_size(&self) -> usize {
        self.layers
            .values()
            .map(|l| (l.a.len() + l.b.len()) * std::mem::size_of::<f32>())
            .sum()
    }
}

/// LoRA injector
pub struct LoraInjector {
    /// Active LoRA stack
    loras: Vec<(LoraWeights, f32)>, // (weights, strength)
}

impl LoraInjector {
    /// Create a new injector
    pub fn new() -> Self {
        Self { loras: Vec::new() }
    }

    /// Add a LoRA to the stack
    pub fn push(&mut self, weights: LoraWeights, strength: f32) {
        self.loras.push((weights, strength.clamp(0.0, 1.0)));
    }

    /// Remove all LoRAs
    pub fn clear(&mut self) {
        self.loras.clear();
    }

    /// Get number of active LoRAs
    pub fn len(&self) -> usize {
        self.loras.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.loras.is_empty()
    }

    /// Apply LoRAs to a layer's weights
    /// W' = W + sum(strength * alpha/rank * (A @ B))
    pub fn apply(&self, layer_name: &str, base_weights: &mut [f32]) {
        for (lora, strength) in &self.loras {
            if let Some(layer) = lora.get_layer(layer_name) {
                let scale = *strength * lora.alpha / lora.rank as f32;
                let delta = layer.compute_delta();

                if delta.len() == base_weights.len() {
                    for (w, d) in base_weights.iter_mut().zip(delta.iter()) {
                        *w += scale * d;
                    }
                }
            }
        }
    }

    /// Compute merged weights for a layer (without modifying base)
    pub fn compute_merged(&self, layer_name: &str, base_weights: &[f32]) -> Vec<f32> {
        let mut merged = base_weights.to_vec();
        self.apply(layer_name, &mut merged);
        merged
    }
}

impl Default for LoraInjector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_weights() {
        let mut layer = LoraLayerWeights::new(4, 4, 2);

        // Simple initialization
        layer.a = vec![1.0; 8]; // 4x2
        layer.b = vec![0.5; 8]; // 2x4

        let delta = layer.compute_delta();

        // Each element should be 2 * 1.0 * 0.5 = 1.0
        assert_eq!(delta.len(), 16);
        assert!((delta[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_lora_injector() {
        let mut injector = LoraInjector::new();

        let mut weights = LoraWeights::empty(2, 1.0);
        let mut layer = LoraLayerWeights::new(2, 2, 2);
        layer.a = vec![1.0; 4];
        layer.b = vec![0.1; 4];
        weights.add_layer("test", layer);

        injector.push(weights, 1.0);

        let mut base = vec![0.0; 4];
        injector.apply("test", &mut base);

        // Should have applied the delta
        assert!(base[0] > 0.0);
    }

    #[test]
    fn test_from_bytes() {
        let data: Vec<u8> = (0..1024)
            .flat_map(|i| (i as f32 * 0.01).to_le_bytes())
            .collect();

        let weights = LoraWeights::from_bytes(&data, 16, 1.0);
        assert!(weights.is_some());
    }
}
