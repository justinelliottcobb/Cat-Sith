//! Sprite-specific KAN architecture
//!
//! This module provides a KAN architecture tailored for sprite generation.
//! The network learns to map from a latent code + pixel coordinates to RGBA colors.
//!
//! # Architecture
//!
//! ```text
//! Input: [latent_code (N dims), x, y] -> normalized coordinates
//! Output: [r, g, b, a] -> pixel color
//! ```
//!
//! This allows generating sprites of arbitrary resolution from compact latent codes.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::*;

use crate::kan::KAN;
use crate::shaders::SPRITE_GENERATE;
use crate::{GpuContext, KanError, Result};

/// Latent code for a sprite
///
/// This compact representation encodes the visual properties of a sprite.
/// The latent space can be interpolated for smooth transitions between sprites.
#[derive(Debug, Clone)]
pub struct SpriteLatent {
    /// Latent vector (typically 8-32 dimensions)
    pub code: Vec<f32>,
}

impl SpriteLatent {
    /// Create a new latent code
    pub fn new(code: Vec<f32>) -> Self {
        Self { code }
    }

    /// Create a random latent code
    pub fn random(dim: usize) -> Self {
        let code: Vec<f32> = (0..dim).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        Self { code }
    }

    /// Create a zero latent code
    pub fn zeros(dim: usize) -> Self {
        Self {
            code: vec![0.0; dim],
        }
    }

    /// Linear interpolation between two latents
    pub fn lerp(&self, other: &SpriteLatent, t: f32) -> Self {
        assert_eq!(self.code.len(), other.code.len());
        let code = self
            .code
            .iter()
            .zip(other.code.iter())
            .map(|(a, b)| a * (1.0 - t) + b * t)
            .collect();
        Self { code }
    }

    /// Spherical interpolation (better for normalized latents)
    pub fn slerp(&self, other: &SpriteLatent, t: f32) -> Self {
        assert_eq!(self.code.len(), other.code.len());

        // Compute angle between vectors
        let dot: f32 = self
            .code
            .iter()
            .zip(other.code.iter())
            .map(|(a, b)| a * b)
            .sum();
        let dot = dot.clamp(-1.0, 1.0);
        let theta = dot.acos();

        if theta.abs() < 1e-6 {
            return self.lerp(other, t);
        }

        let sin_theta = theta.sin();
        let s0 = ((1.0 - t) * theta).sin() / sin_theta;
        let s1 = (t * theta).sin() / sin_theta;

        let code = self
            .code
            .iter()
            .zip(other.code.iter())
            .map(|(a, b)| a * s0 + b * s1)
            .collect();
        Self { code }
    }

    /// Dimension of the latent code
    pub fn dim(&self) -> usize {
        self.code.len()
    }
}

/// GPU parameters for sprite generation
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SpriteGenParams {
    width: u32,
    height: u32,
    latent_dim: u32,
    _padding: u32,
}

/// KAN-based sprite generator
///
/// Generates pixel art sprites from compact latent codes.
/// The architecture is designed for smooth, continuous color gradients
/// that are characteristic of stylized pixel art.
pub struct SpriteKAN {
    /// Underlying KAN network
    kan: KAN,
    /// Latent code dimension
    pub latent_dim: usize,
    /// Default sprite width
    pub width: usize,
    /// Default sprite height
    pub height: usize,
    // GPU resources for batch generation (prefixed with _ to silence warnings for now)
    #[allow(dead_code)]
    gpu: GpuContext,
    #[allow(dead_code)]
    sprite_pipeline: ComputePipeline,
    #[allow(dead_code)]
    sprite_bind_group_layout: BindGroupLayout,
}

impl SpriteKAN {
    /// Create a new sprite KAN
    ///
    /// # Arguments
    /// * `gpu` - Shared GPU context
    /// * `latent_dim` - Dimension of latent codes (typically 8-32)
    /// * `hidden_dims` - Hidden layer dimensions
    /// * `functions_per_layer` - Univariate functions per layer
    /// * `width` - Default sprite width
    /// * `height` - Default sprite height
    ///
    /// # Example
    /// ```ignore
    /// let sprite_kan = SpriteKAN::new(
    ///     gpu,
    ///     16,           // 16-dim latent code
    ///     &[32, 32],    // two hidden layers of 32
    ///     &[8, 8, 8],   // 8 functions per layer
    ///     32, 32,       // 32x32 sprites
    /// )?;
    /// ```
    pub fn new(
        gpu: GpuContext,
        latent_dim: usize,
        hidden_dims: &[usize],
        functions_per_layer: &[usize],
        width: usize,
        height: usize,
    ) -> Result<Self> {
        // Build layer dimensions: [latent + 2 (x,y), ...hidden, 4 (rgba)]
        let input_dim = latent_dim + 2; // latent code + x + y
        let output_dim = 4; // RGBA

        let mut layer_dims = vec![input_dim];
        layer_dims.extend_from_slice(hidden_dims);
        layer_dims.push(output_dim);

        if layer_dims.len() - 1 != functions_per_layer.len() {
            return Err(KanError::InvalidConfig(format!(
                "functions_per_layer length ({}) must match number of layers ({})",
                functions_per_layer.len(),
                layer_dims.len() - 1
            )));
        }

        // Create KAN with normalized input range
        let kan = KAN::new(
            gpu.clone(),
            &layer_dims,
            functions_per_layer,
            -1.0..1.0, // Normalized coordinate range
            8,         // 8 knots for smooth but detailed curves
            3,         // Cubic B-splines
        )?;

        // Create GPU pipeline for batch sprite generation
        let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Sprite Generate Shader"),
            source: ShaderSource::Wgsl(SPRITE_GENERATE.into()),
        });

        let sprite_bind_group_layout =
            gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Sprite Gen Layout"),
                entries: &[
                    // Params
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Latent code
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output pixels (RGBA)
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Sprite Gen Pipeline Layout"),
            bind_group_layouts: &[&sprite_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sprite_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
            compilation_options: PipelineCompilationOptions {
                constants: &HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            label: Some("Sprite Gen Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            kan,
            latent_dim,
            width,
            height,
            gpu,
            sprite_pipeline,
            sprite_bind_group_layout,
        })
    }

    /// Generate a sprite from a latent code (CPU)
    ///
    /// Returns RGBA pixels as a flat array [r0,g0,b0,a0, r1,g1,b1,a1, ...]
    pub fn generate(&self, latent: &SpriteLatent) -> Result<Vec<u8>> {
        self.generate_sized(latent, self.width, self.height)
    }

    /// Generate a sprite at a specific size (CPU)
    pub fn generate_sized(
        &self,
        latent: &SpriteLatent,
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        if latent.dim() != self.latent_dim {
            return Err(KanError::DimensionMismatch {
                expected: self.latent_dim,
                actual: latent.dim(),
            });
        }

        let mut pixels = Vec::with_capacity(width * height * 4);

        for y in 0..height {
            for x in 0..width {
                // Normalize coordinates to [-1, 1]
                let nx = (x as f32 / (width - 1) as f32) * 2.0 - 1.0;
                let ny = (y as f32 / (height - 1) as f32) * 2.0 - 1.0;

                // Build input: [latent..., x, y]
                let mut input = latent.code.clone();
                input.push(nx);
                input.push(ny);

                // Forward through KAN
                let output = self.kan.forward(&input)?;

                // Convert to u8 RGBA (sigmoid activation for [0,1] range)
                let r = (sigmoid(output[0]) * 255.0) as u8;
                let g = (sigmoid(output[1]) * 255.0) as u8;
                let b = (sigmoid(output[2]) * 255.0) as u8;
                let a = (sigmoid(output[3]) * 255.0) as u8;

                pixels.extend_from_slice(&[r, g, b, a]);
            }
        }

        Ok(pixels)
    }

    /// Generate a sprite (GPU) - placeholder for full GPU implementation
    ///
    /// TODO: Implement full GPU-accelerated sprite generation
    pub async fn generate_gpu(&self, latent: &SpriteLatent) -> Result<Vec<u8>> {
        // For now, fall back to CPU
        // Full GPU implementation would evaluate all pixels in parallel
        self.generate(latent)
    }

    /// Train on a sprite example
    ///
    /// # Arguments
    /// * `latent` - The latent code for this sprite
    /// * `pixels` - Target RGBA pixels (width * height * 4 bytes)
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// Returns average loss per pixel.
    pub fn train(
        &mut self,
        latent: &SpriteLatent,
        pixels: &[u8],
        learning_rate: f32,
    ) -> Result<f32> {
        self.train_sized(latent, pixels, self.width, self.height, learning_rate)
    }

    /// Train on a sprite at specific size
    pub fn train_sized(
        &mut self,
        latent: &SpriteLatent,
        pixels: &[u8],
        width: usize,
        height: usize,
        learning_rate: f32,
    ) -> Result<f32> {
        if latent.dim() != self.latent_dim {
            return Err(KanError::DimensionMismatch {
                expected: self.latent_dim,
                actual: latent.dim(),
            });
        }

        if pixels.len() != width * height * 4 {
            return Err(KanError::InvalidConfig(format!(
                "Expected {} pixels, got {}",
                width * height * 4,
                pixels.len()
            )));
        }

        let mut total_loss = 0.0;

        for y in 0..height {
            for x in 0..width {
                let px_idx = (y * width + x) * 4;

                // Normalize coordinates
                let nx = (x as f32 / (width - 1) as f32) * 2.0 - 1.0;
                let ny = (y as f32 / (height - 1) as f32) * 2.0 - 1.0;

                // Build input
                let mut input = latent.code.clone();
                input.push(nx);
                input.push(ny);

                // Target (inverse sigmoid to match our activation)
                let target = vec![
                    inverse_sigmoid(pixels[px_idx] as f32 / 255.0),
                    inverse_sigmoid(pixels[px_idx + 1] as f32 / 255.0),
                    inverse_sigmoid(pixels[px_idx + 2] as f32 / 255.0),
                    inverse_sigmoid(pixels[px_idx + 3] as f32 / 255.0),
                ];

                total_loss += self.kan.train(&input, &target, learning_rate)?;
            }
        }

        Ok(total_loss / (width * height) as f32)
    }

    /// Get the underlying KAN for inspection/modification
    pub fn kan(&self) -> &KAN {
        &self.kan
    }

    /// Get mutable access to the underlying KAN
    pub fn kan_mut(&mut self) -> &mut KAN {
        &mut self.kan
    }

    /// Total parameter count
    pub fn parameter_count(&self) -> usize {
        self.kan.parameter_count()
    }
}

/// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse sigmoid (logit)
fn inverse_sigmoid(y: f32) -> f32 {
    let y = y.clamp(0.001, 0.999); // Avoid infinity
    (y / (1.0 - y)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SpriteLatent tests ---

    #[test]
    fn test_sprite_latent_new() {
        let latent = SpriteLatent::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(latent.dim(), 3);
        assert_eq!(latent.code, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sprite_latent_random() {
        let latent = SpriteLatent::random(16);
        assert_eq!(latent.dim(), 16);

        // Random values should be in [-1, 1]
        for val in &latent.code {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_sprite_latent_zeros() {
        let latent = SpriteLatent::zeros(8);
        assert_eq!(latent.dim(), 8);

        for val in &latent.code {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn test_sprite_latent_lerp() {
        let a = SpriteLatent::new(vec![0.0, 0.0]);
        let b = SpriteLatent::new(vec![1.0, 1.0]);

        let mid = a.lerp(&b, 0.5);
        assert!((mid.code[0] - 0.5).abs() < 1e-6);
        assert!((mid.code[1] - 0.5).abs() < 1e-6);

        // Test endpoints
        let start = a.lerp(&b, 0.0);
        assert!((start.code[0] - 0.0).abs() < 1e-6);

        let end = a.lerp(&b, 1.0);
        assert!((end.code[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sprite_latent_slerp() {
        let a = SpriteLatent::new(vec![1.0, 0.0]);
        let b = SpriteLatent::new(vec![0.0, 1.0]);

        let mid = a.slerp(&b, 0.5);

        // Slerp should produce a point on the arc between a and b
        // The magnitude should be close to 1 (unit vectors)
        let mag: f32 = mid.code.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (mag - 1.0).abs() < 0.1,
            "Slerp should preserve approximate magnitude"
        );

        // The interpolated point should be between a and b
        assert!(mid.code[0] > 0.0 && mid.code[0] < 1.0);
        assert!(mid.code[1] > 0.0 && mid.code[1] < 1.0);
    }

    #[test]
    fn test_sprite_latent_slerp_same_vector() {
        let a = SpriteLatent::new(vec![1.0, 0.0]);
        let b = SpriteLatent::new(vec![1.0, 0.0]);

        // Slerp of same vector should return that vector
        let result = a.slerp(&b, 0.5);
        assert!((result.code[0] - 1.0).abs() < 1e-5);
        assert!((result.code[1] - 0.0).abs() < 1e-5);
    }

    // --- SpriteKAN tests ---

    #[tokio::test]
    async fn test_sprite_kan_creation() {
        let gpu = GpuContext::new().await.unwrap();
        let sprite_kan = SpriteKAN::new(gpu, 8, &[16], &[4, 4], 8, 8).unwrap();

        assert_eq!(sprite_kan.latent_dim, 8);
        assert_eq!(sprite_kan.width, 8);
        assert_eq!(sprite_kan.height, 8);
    }

    #[tokio::test]
    async fn test_sprite_kan_generate() {
        let gpu = GpuContext::new().await.unwrap();
        let sprite_kan = SpriteKAN::new(gpu, 8, &[16], &[4, 4], 8, 8).unwrap();

        let latent = SpriteLatent::random(8);
        let pixels = sprite_kan.generate(&latent).unwrap();

        assert_eq!(pixels.len(), 8 * 8 * 4); // 8x8 RGBA
    }

    #[tokio::test]
    async fn test_sprite_kan_generate_sized() {
        let gpu = GpuContext::new().await.unwrap();
        let sprite_kan = SpriteKAN::new(gpu, 8, &[16], &[4, 4], 8, 8).unwrap();

        let latent = SpriteLatent::random(8);

        // Generate at different sizes
        let pixels_16 = sprite_kan.generate_sized(&latent, 16, 16).unwrap();
        assert_eq!(pixels_16.len(), 16 * 16 * 4);

        let pixels_4 = sprite_kan.generate_sized(&latent, 4, 4).unwrap();
        assert_eq!(pixels_4.len(), 4 * 4 * 4);
    }

    #[tokio::test]
    async fn test_sprite_kan_wrong_latent_dim() {
        let gpu = GpuContext::new().await.unwrap();
        let sprite_kan = SpriteKAN::new(gpu, 8, &[16], &[4, 4], 8, 8).unwrap();

        let wrong_latent = SpriteLatent::random(16); // Wrong dimension
        let result = sprite_kan.generate(&wrong_latent);

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_sprite_kan_pixels_in_valid_range() {
        let gpu = GpuContext::new().await.unwrap();
        let sprite_kan = SpriteKAN::new(gpu, 8, &[16], &[4, 4], 8, 8).unwrap();

        let latent = SpriteLatent::random(8);
        let pixels = sprite_kan.generate(&latent).unwrap();

        // Pixels should have been generated (non-empty)
        assert!(!pixels.is_empty());
        // Since pixels are u8, they're inherently in valid range 0-255
    }

    #[tokio::test]
    async fn test_sprite_kan_deterministic() {
        let gpu = GpuContext::new().await.unwrap();
        let sprite_kan = SpriteKAN::new(gpu, 8, &[16], &[4, 4], 4, 4).unwrap();

        let latent = SpriteLatent::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

        let pixels1 = sprite_kan.generate(&latent).unwrap();
        let pixels2 = sprite_kan.generate(&latent).unwrap();

        assert_eq!(pixels1, pixels2, "Same latent should produce same pixels");
    }

    #[tokio::test]
    async fn test_sprite_kan_forward_pass_works() {
        // Test that the forward pass produces valid output for different latents
        // Note: An untrained network may produce similar outputs due to sigmoid
        // saturation around 0.5, but the forward pass should still complete
        let gpu = GpuContext::new().await.unwrap();
        let sprite_kan = SpriteKAN::new(gpu, 8, &[16], &[4, 4], 4, 4).unwrap();

        let latent1 = SpriteLatent::new(vec![0.1; 8]);
        let latent2 = SpriteLatent::new(vec![0.9; 8]);

        // Both should produce valid output
        let pixels1 = sprite_kan.generate(&latent1).unwrap();
        let pixels2 = sprite_kan.generate(&latent2).unwrap();

        // Same size
        assert_eq!(pixels1.len(), pixels2.len());
        // Expected size for 4x4 RGBA
        assert_eq!(pixels1.len(), 4 * 4 * 4);
    }

    #[tokio::test]
    async fn test_sprite_kan_parameter_count() {
        let gpu = GpuContext::new().await.unwrap();
        let sprite_kan = SpriteKAN::new(gpu, 8, &[16], &[4, 4], 8, 8).unwrap();

        let count = sprite_kan.parameter_count();
        assert!(count > 0, "Should have parameters");
    }

    #[tokio::test]
    async fn test_sprite_kan_train() {
        let gpu = GpuContext::new().await.unwrap();
        let mut sprite_kan = SpriteKAN::new(gpu, 4, &[8], &[4, 4], 4, 4).unwrap();

        let latent = SpriteLatent::random(4);

        // Create a simple target (all red)
        let target: Vec<u8> = (0..4 * 4)
            .flat_map(|_| vec![255, 0, 0, 255])
            .collect();

        // Training should run without panic
        let loss = sprite_kan.train(&latent, &target, 0.01).unwrap();
        assert!(loss.is_finite(), "Loss should be finite");
    }

    #[tokio::test]
    async fn test_sprite_kan_config_mismatch() {
        let gpu = GpuContext::new().await.unwrap();

        // functions_per_layer length doesn't match number of layers
        let result = SpriteKAN::new(
            gpu,
            8,
            &[16, 16],  // 2 hidden layers = 3 total layers
            &[4, 4],    // Only 2 function counts (should be 3)
            8,
            8,
        );

        assert!(result.is_err());
    }
}
