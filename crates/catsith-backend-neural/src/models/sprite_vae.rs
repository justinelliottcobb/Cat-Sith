//! Sprite VAE model
//!
//! Small variational autoencoder for generating sprites from embeddings.

use crate::inference::{InferenceEngine, InferenceError, Tensor};

/// Sprite VAE configuration
#[derive(Debug, Clone)]
pub struct SpriteVAEConfig {
    /// Latent dimension
    pub latent_dim: usize,
    /// Output sprite size
    pub sprite_size: u32,
    /// Number of channels
    pub channels: usize,
}

impl Default for SpriteVAEConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            sprite_size: 32,
            channels: 4, // RGBA
        }
    }
}

/// Sprite VAE model wrapper
pub struct SpriteVAE {
    config: SpriteVAEConfig,
    loaded: bool,
}

impl SpriteVAE {
    /// Create a new Sprite VAE
    pub fn new(config: SpriteVAEConfig) -> Self {
        Self {
            config,
            loaded: false,
        }
    }

    /// Load the model
    pub fn load(&mut self, engine: &mut InferenceEngine) -> Result<(), InferenceError> {
        engine.load_model("sprite_vae_encoder")?;
        engine.load_model("sprite_vae_decoder")?;
        self.loaded = true;
        Ok(())
    }

    /// Check if loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Get configuration
    pub fn config(&self) -> &SpriteVAEConfig {
        &self.config
    }

    /// Encode an image to latent space
    pub fn encode(
        &self,
        engine: &InferenceEngine,
        image: &Tensor,
    ) -> Result<Tensor, InferenceError> {
        if !self.loaded {
            return Err(InferenceError::ModelNotFound(
                "sprite_vae_encoder".to_string(),
            ));
        }

        let outputs = engine.run("sprite_vae_encoder", vec![image.clone()])?;
        Ok(outputs.into_iter().next().unwrap())
    }

    /// Decode latent to image
    pub fn decode(
        &self,
        engine: &InferenceEngine,
        latent: &Tensor,
    ) -> Result<Tensor, InferenceError> {
        if !self.loaded {
            return Err(InferenceError::ModelNotFound(
                "sprite_vae_decoder".to_string(),
            ));
        }

        let outputs = engine.run("sprite_vae_decoder", vec![latent.clone()])?;
        Ok(outputs.into_iter().next().unwrap())
    }

    /// Generate sprite from embedding
    pub fn generate(
        &self,
        engine: &InferenceEngine,
        embedding: &[f32],
    ) -> Result<SpriteOutput, InferenceError> {
        // Convert embedding to latent tensor
        let latent = Tensor::new(vec![1, self.config.latent_dim], embedding.to_vec());

        // Decode to image
        let image_tensor = self.decode(engine, &latent)?;

        // Convert to sprite output
        Ok(SpriteOutput::from_tensor(
            &image_tensor,
            self.config.sprite_size,
        ))
    }

    /// Generate sprite without model (for testing)
    pub fn generate_test(&self, embedding: &[f32]) -> SpriteOutput {
        let size = self.config.sprite_size as usize;
        let mut pixels = vec![0u8; size * size * 4];

        // Generate a simple pattern based on embedding
        for y in 0..size {
            for x in 0..size {
                let idx = (y * size + x) * 4;
                let emb_idx = (x + y) % embedding.len().max(1);

                let val = embedding.get(emb_idx).copied().unwrap_or(0.0);
                let brightness = ((val + 1.0) * 127.5) as u8;

                pixels[idx] = brightness; // R
                pixels[idx + 1] = brightness / 2; // G
                pixels[idx + 2] = brightness / 3; // B
                pixels[idx + 3] = 255; // A
            }
        }

        SpriteOutput {
            width: self.config.sprite_size,
            height: self.config.sprite_size,
            pixels,
        }
    }
}

impl Default for SpriteVAE {
    fn default() -> Self {
        Self::new(SpriteVAEConfig::default())
    }
}

/// Generated sprite output
#[derive(Debug, Clone)]
pub struct SpriteOutput {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// RGBA pixel data
    pub pixels: Vec<u8>,
}

impl SpriteOutput {
    /// Create from tensor (CHW format, values 0-1)
    pub fn from_tensor(tensor: &Tensor, size: u32) -> Self {
        let size_usize = size as usize;
        let mut pixels = vec![0u8; size_usize * size_usize * 4];

        // Assuming tensor is [1, C, H, W] with C=3 or C=4
        for y in 0..size_usize {
            for x in 0..size_usize {
                let pixel_idx = (y * size_usize + x) * 4;

                // Get RGB values
                for c in 0..3 {
                    let tensor_idx = c * size_usize * size_usize + y * size_usize + x;
                    let val = tensor.data.get(tensor_idx).copied().unwrap_or(0.0);
                    pixels[pixel_idx + c] = (val.clamp(0.0, 1.0) * 255.0) as u8;
                }

                // Alpha (if present, otherwise 255)
                let alpha_idx = 3 * size_usize * size_usize + y * size_usize + x;
                let alpha = tensor
                    .data
                    .get(alpha_idx)
                    .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                    .unwrap_or(255);
                pixels[pixel_idx + 3] = alpha;
            }
        }

        Self {
            width: size,
            height: size,
            pixels,
        }
    }

    /// Get pixel at position
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y * self.width + x) * 4) as usize;
        Some([
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
            self.pixels[idx + 3],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprite_vae_config() {
        let config = SpriteVAEConfig::default();
        assert_eq!(config.sprite_size, 32);
        assert_eq!(config.channels, 4);
    }

    #[test]
    fn test_generate_test() {
        let vae = SpriteVAE::default();
        let embedding = vec![0.5; 128];

        let output = vae.generate_test(&embedding);
        assert_eq!(output.width, 32);
        assert_eq!(output.height, 32);
        assert_eq!(output.pixels.len(), 32 * 32 * 4);
    }

    #[test]
    fn test_sprite_output_from_tensor() {
        let tensor = Tensor::random(vec![1, 4, 16, 16]);
        let output = SpriteOutput::from_tensor(&tensor, 16);

        assert_eq!(output.width, 16);
        assert!(output.get_pixel(0, 0).is_some());
        assert!(output.get_pixel(16, 16).is_none());
    }
}
