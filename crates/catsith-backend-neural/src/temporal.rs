//! Temporal coherence
//!
//! Maintains visual consistency between frames to reduce flickering.

use catsith_core::EntityId;
use std::collections::HashMap;

/// Temporal coherence tracker
pub struct TemporalCoherence {
    /// Previous frame embeddings by entity
    previous_embeddings: HashMap<EntityId, Vec<f32>>,
    /// Blending factor (0 = no blending, 1 = full previous frame)
    blend_factor: f32,
    /// Frame counter
    frame_count: u64,
}

impl TemporalCoherence {
    /// Create a new temporal coherence tracker
    pub fn new(blend_factor: f32) -> Self {
        Self {
            previous_embeddings: HashMap::new(),
            blend_factor: blend_factor.clamp(0.0, 0.9),
            frame_count: 0,
        }
    }

    /// Get blend factor
    pub fn blend_factor(&self) -> f32 {
        self.blend_factor
    }

    /// Set blend factor
    pub fn set_blend_factor(&mut self, factor: f32) {
        self.blend_factor = factor.clamp(0.0, 0.9);
    }

    /// Start a new frame
    pub fn begin_frame(&mut self) {
        self.frame_count += 1;
    }

    /// Get temporally smoothed embedding
    pub fn smooth_embedding(&mut self, entity_id: EntityId, current: &[f32]) -> Vec<f32> {
        match self.previous_embeddings.get(&entity_id) {
            Some(previous) if previous.len() == current.len() => {
                // Blend with previous frame
                let smoothed: Vec<f32> = current
                    .iter()
                    .zip(previous.iter())
                    .map(|(c, p)| c * (1.0 - self.blend_factor) + p * self.blend_factor)
                    .collect();

                // Store for next frame
                self.previous_embeddings.insert(entity_id, smoothed.clone());
                smoothed
            }
            _ => {
                // First time seeing this entity
                self.previous_embeddings.insert(entity_id, current.to_vec());
                current.to_vec()
            }
        }
    }

    /// Clear tracking for an entity
    pub fn remove_entity(&mut self, entity_id: &EntityId) {
        self.previous_embeddings.remove(entity_id);
    }

    /// Clear all tracked entities
    pub fn clear(&mut self) {
        self.previous_embeddings.clear();
    }

    /// Get number of tracked entities
    pub fn tracked_count(&self) -> usize {
        self.previous_embeddings.len()
    }

    /// Get current frame number
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Prune old entities (not seen in recent frames)
    /// In a real implementation, this would track last-seen frame
    pub fn prune_stale(&mut self, max_entities: usize) {
        while self.previous_embeddings.len() > max_entities {
            // Remove arbitrary entry (should be LRU in production)
            if let Some(key) = self.previous_embeddings.keys().next().copied() {
                self.previous_embeddings.remove(&key);
            }
        }
    }
}

impl Default for TemporalCoherence {
    fn default() -> Self {
        Self::new(0.3)
    }
}

/// Motion interpolation for smooth entity movement
pub struct MotionInterpolator {
    /// Previous positions
    previous_positions: HashMap<EntityId, [f64; 2]>,
    /// Previous velocities (for prediction)
    previous_velocities: HashMap<EntityId, [f64; 2]>,
}

impl MotionInterpolator {
    /// Create a new motion interpolator
    pub fn new() -> Self {
        Self {
            previous_positions: HashMap::new(),
            previous_velocities: HashMap::new(),
        }
    }

    /// Update entity position
    pub fn update(&mut self, entity_id: EntityId, position: [f64; 2], velocity: [f64; 2]) {
        self.previous_positions.insert(entity_id, position);
        self.previous_velocities.insert(entity_id, velocity);
    }

    /// Interpolate position between frames
    pub fn interpolate(&self, entity_id: &EntityId, t: f64) -> Option<[f64; 2]> {
        let pos = self.previous_positions.get(entity_id)?;
        let vel = self
            .previous_velocities
            .get(entity_id)
            .unwrap_or(&[0.0, 0.0]);

        Some([pos[0] + vel[0] * t, pos[1] + vel[1] * t])
    }

    /// Clear tracking
    pub fn clear(&mut self) {
        self.previous_positions.clear();
        self.previous_velocities.clear();
    }
}

impl Default for MotionInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_smoothing() {
        let mut tc = TemporalCoherence::new(0.5);
        let entity = EntityId::new();

        // First frame
        let e1 = tc.smooth_embedding(entity, &[1.0, 0.0]);
        assert_eq!(e1, vec![1.0, 0.0]);

        // Second frame - should blend
        let e2 = tc.smooth_embedding(entity, &[0.0, 1.0]);
        assert_eq!(e2, vec![0.5, 0.5]); // 0.5 blend
    }

    #[test]
    fn test_motion_interpolation() {
        let mut mi = MotionInterpolator::new();
        let entity = EntityId::new();

        mi.update(entity, [0.0, 0.0], [10.0, 5.0]);

        let pos = mi.interpolate(&entity, 0.5).unwrap();
        assert_eq!(pos, [5.0, 2.5]);
    }

    #[test]
    fn test_prune_stale() {
        let mut tc = TemporalCoherence::new(0.3);

        for _ in 0..100 {
            tc.smooth_embedding(EntityId::new(), &[0.0; 10]);
        }

        assert_eq!(tc.tracked_count(), 100);

        tc.prune_stale(50);
        assert_eq!(tc.tracked_count(), 50);
    }
}
