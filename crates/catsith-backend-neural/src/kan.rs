//! KAN-based sprite generation integration
//!
//! This module provides integration between the neural backend and the
//! KAN (Kolmogorov-Arnold Network) sprite generator.
//!
//! # Architecture
//!
//! The KAN sprite generator maps semantic descriptions to sprites:
//!
//! ```text
//! Entity Description -> Embedding -> Latent Code -> SpriteKAN -> RGBA Pixels
//! ```
//!
//! KANs are particularly well-suited for sprite generation because:
//! - B-spline basis functions capture smooth color gradients naturally
//! - Latent space interpolation produces smooth transitions between states
//! - The learned functions can be visualized for interpretability

use std::collections::HashMap;
use std::path::Path;

use catsith_kan::{GpuContext, KanError, SpriteKAN, SpriteLatent};

/// A sprite generator using Kolmogorov-Arnold Networks
///
/// Wraps a [`SpriteKAN`] and provides sprite caching and entity-to-latent mapping.
pub struct KanSpriteGenerator {
    /// The underlying KAN model
    kan: SpriteKAN,
    /// Cached sprites by entity identifier
    cache: HashMap<String, Vec<u8>>,
    /// Latent codes for known entity types
    latent_library: HashMap<String, SpriteLatent>,
    /// Default latent dimension
    latent_dim: usize,
}

impl KanSpriteGenerator {
    /// Create a new KAN sprite generator
    ///
    /// # Arguments
    /// * `latent_dim` - Dimension of latent codes (typically 8-32)
    /// * `width` - Sprite width in pixels
    /// * `height` - Sprite height in pixels
    pub async fn new(latent_dim: usize, width: usize, height: usize) -> Result<Self, KanError> {
        let gpu = GpuContext::new().await?;

        // Create a SpriteKAN with reasonable defaults
        let kan = SpriteKAN::new(
            gpu,
            latent_dim,
            &[32, 32],   // Two hidden layers
            &[8, 8, 8],  // 8 functions per layer
            width,
            height,
        )?;

        Ok(Self {
            kan,
            cache: HashMap::new(),
            latent_library: HashMap::new(),
            latent_dim,
        })
    }

    /// Register a latent code for an entity type
    ///
    /// This allows you to define what latent code produces a particular
    /// type of sprite (e.g., "fighter", "asteroid", "explosion").
    pub fn register_entity(&mut self, entity_kind: impl Into<String>, latent: SpriteLatent) {
        self.latent_library.insert(entity_kind.into(), latent);
    }

    /// Generate a sprite for an entity type
    ///
    /// If the entity type is registered, uses its latent code.
    /// Otherwise, generates a random sprite.
    pub fn generate(&mut self, entity_kind: &str) -> Result<Vec<u8>, KanError> {
        // Check cache first
        if let Some(pixels) = self.cache.get(entity_kind) {
            return Ok(pixels.clone());
        }

        // Get or create latent code
        let latent = self
            .latent_library
            .get(entity_kind)
            .cloned()
            .unwrap_or_else(|| SpriteLatent::random(self.latent_dim));

        // Generate sprite
        let pixels = self.kan.generate(&latent)?;

        // Cache for future use
        self.cache.insert(entity_kind.to_string(), pixels.clone());

        Ok(pixels)
    }

    /// Generate a sprite with a specific latent code (no caching)
    pub fn generate_from_latent(&self, latent: &SpriteLatent) -> Result<Vec<u8>, KanError> {
        self.kan.generate(latent)
    }

    /// Generate an interpolated sprite between two entity types
    ///
    /// Useful for smooth transitions between states.
    pub fn generate_interpolated(
        &self,
        from_kind: &str,
        to_kind: &str,
        t: f32,
    ) -> Result<Vec<u8>, KanError> {
        let from_latent = self
            .latent_library
            .get(from_kind)
            .cloned()
            .unwrap_or_else(|| SpriteLatent::random(self.latent_dim));

        let to_latent = self
            .latent_library
            .get(to_kind)
            .cloned()
            .unwrap_or_else(|| SpriteLatent::random(self.latent_dim));

        // Spherical interpolation for better quality
        let interpolated = from_latent.slerp(&to_latent, t);

        self.kan.generate(&interpolated)
    }

    /// Train the generator on example sprites
    ///
    /// # Arguments
    /// * `entity_kind` - Entity type identifier
    /// * `pixels` - Target RGBA pixels
    /// * `epochs` - Number of training epochs
    /// * `learning_rate` - Learning rate for optimization
    pub fn train(
        &mut self,
        entity_kind: &str,
        pixels: &[u8],
        epochs: usize,
        learning_rate: f32,
    ) -> Result<f32, KanError> {
        // Get or create latent for this entity
        let latent = self
            .latent_library
            .entry(entity_kind.to_string())
            .or_insert_with(|| SpriteLatent::random(self.latent_dim))
            .clone();

        let mut final_loss = 0.0;

        for _epoch in 0..epochs {
            final_loss = self.kan.train(&latent, pixels, learning_rate)?;
        }

        // Invalidate cache
        self.cache.remove(entity_kind);

        Ok(final_loss)
    }

    /// Clear the sprite cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get the dimensions of generated sprites
    pub fn sprite_dimensions(&self) -> (usize, usize) {
        (self.kan.width, self.kan.height)
    }

    /// Get the total parameter count
    pub fn parameter_count(&self) -> usize {
        self.kan.parameter_count()
    }

    /// Save learned latent codes to a file
    pub fn save_latent_library(&self, _path: &Path) -> Result<(), KanError> {
        // TODO: Implement serialization
        Ok(())
    }

    /// Load latent codes from a file
    pub fn load_latent_library(&mut self, _path: &Path) -> Result<(), KanError> {
        // TODO: Implement deserialization
        Ok(())
    }
}

/// Convert a semantic entity description to a latent code
///
/// This is a placeholder for more sophisticated embedding.
/// In production, you'd use a trained encoder or semantic hashing.
pub fn description_to_latent(description: &str, dim: usize) -> SpriteLatent {
    // Simple hash-based approach for now
    let mut code = vec![0.0; dim];

    for (i, byte) in description.bytes().enumerate() {
        let idx = i % dim;
        code[idx] += (byte as f32 - 128.0) / 128.0;
    }

    // Normalize
    let norm: f32 = code.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for c in &mut code {
            *c /= norm;
        }
    }

    SpriteLatent::new(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kan_generator_create() {
        let result = KanSpriteGenerator::new(8, 16, 16).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_kan_generator_generate() {
        let mut generator = KanSpriteGenerator::new(8, 16, 16).await.unwrap();

        // Register an entity
        generator.register_entity("fighter", SpriteLatent::random(8));

        // Generate sprite
        let pixels = generator.generate("fighter").unwrap();
        assert_eq!(pixels.len(), 16 * 16 * 4); // RGBA
    }

    #[test]
    fn test_description_to_latent() {
        let latent = description_to_latent("fighter ship", 8);
        assert_eq!(latent.dim(), 8);

        // Should produce different latents for different descriptions
        let latent2 = description_to_latent("asteroid rock", 8);
        assert_ne!(latent.code, latent2.code);
    }
}
