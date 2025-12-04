//! Candle-based Stable Diffusion inference
//!
//! This module provides GPU-accelerated image generation using Hugging Face's
//! Candle framework with Stable Diffusion models.
//!
//! # Overview
//!
//! The [`CandleDiffusionPipeline`] implements a complete text-to-image pipeline:
//!
//! 1. **Tokenization**: Text prompts are tokenized using CLIP's BPE tokenizer
//! 2. **Text Encoding**: Tokens are converted to embeddings via CLIP text encoder
//! 3. **Denoising**: UNet iteratively denoises random latents guided by text embeddings
//! 4. **Decoding**: VAE decoder converts latents to RGB images
//!
//! # Example
//!
//! ```rust,ignore
//! use catsith_backend_neural::{CandleDiffusionPipeline, DiffusionConfig};
//!
//! // Use pixel art preset (64x64, 15 steps)
//! let config = DiffusionConfig::pixel_art();
//! let mut pipeline = CandleDiffusionPipeline::new(config)?;
//! pipeline.load()?;
//!
//! let image = pipeline.generate(
//!     "pixel art wizard casting spell",
//!     Some("blurry, low quality"),  // negative prompt
//!     42,  // seed (currently unused, for API compatibility)
//! )?;
//! image.save("wizard.png")?;
//! ```
//!
//! # Configuration Presets
//!
//! - [`DiffusionConfig::default()`] - Standard 512x512, 20 steps
//! - [`DiffusionConfig::pixel_art()`] - Small 64x64 sprites, 15 steps
//! - [`DiffusionConfig::sprite_sheet()`] - 256x256 sprite sheets, 20 steps
//!
//! # Model Requirements
//!
//! The pipeline expects a standard Stable Diffusion model directory structure:
//!
//! ```text
//! model_path/
//! ├── tokenizer/
//! │   └── tokenizer.json (or vocab.json + merges.txt)
//! ├── text_encoder/
//! │   └── model.safetensors (or pytorch_model.bin)
//! ├── vae/
//! │   └── diffusion_pytorch_model.safetensors (or .bin)
//! └── unet/
//!     └── diffusion_pytorch_model.safetensors (or .bin)
//! ```
//!
//! If the tokenizer is not found locally, it will be downloaded from Hugging Face
//! (openai/clip-vit-base-patch32 for SD 1.5/2.1, openai/clip-vit-large-patch14 for SDXL).

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Module, Tensor};
#[cfg(feature = "candle")]
use candle_nn as nn;
#[cfg(feature = "candle")]
use candle_transformers::models::stable_diffusion::{
    self,
    schedulers::SchedulerConfig,
};

use std::path::{Path, PathBuf};

use crate::inference::InferenceError;

/// Supported Stable Diffusion versions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableDiffusionVersion {
    /// SD 1.5 - Most compatible, good for pixel art LoRAs
    V1_5,
    /// SD 2.1 - Improved quality
    V2_1,
    /// SDXL 1.0 - High resolution, best quality
    Xl,
    /// SDXL Turbo - Fast generation (fewer steps)
    Turbo,
}

impl StableDiffusionVersion {
    /// Get the default guidance scale for this version
    pub fn default_guidance_scale(&self) -> f64 {
        match self {
            Self::V1_5 | Self::V2_1 => 7.5,
            Self::Xl => 5.0,
            Self::Turbo => 0.0, // Turbo doesn't use CFG
        }
    }

    /// Get the default number of inference steps
    pub fn default_steps(&self) -> usize {
        match self {
            Self::V1_5 | Self::V2_1 => 20,
            Self::Xl => 30,
            Self::Turbo => 4, // Turbo is designed for few steps
        }
    }

    /// Get the latent channels for this version
    pub fn latent_channels(&self) -> usize {
        4 // All versions use 4-channel latents
    }
}

/// Configuration for the diffusion pipeline
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Model version to use
    pub version: StableDiffusionVersion,
    /// Path to model weights directory
    pub model_path: PathBuf,
    /// Number of denoising steps
    pub num_steps: usize,
    /// Classifier-free guidance scale
    pub guidance_scale: f64,
    /// Output image width (must be divisible by 8)
    pub width: usize,
    /// Output image height (must be divisible by 8)
    pub height: usize,
    /// Use FP16 precision (faster, less memory)
    pub use_f16: bool,
    /// Use flash attention if available
    pub use_flash_attn: bool,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            version: StableDiffusionVersion::V1_5,
            model_path: PathBuf::from("models/pixel-art-style"),
            num_steps: 20,
            guidance_scale: 7.5,
            width: 512,
            height: 512,
            use_f16: true,
            use_flash_attn: false,
        }
    }
}

impl DiffusionConfig {
    /// Create config for pixel art generation (smaller, faster)
    pub fn pixel_art() -> Self {
        Self {
            width: 64,
            height: 64,
            num_steps: 15,
            ..Default::default()
        }
    }

    /// Create config for sprite sheet generation
    pub fn sprite_sheet() -> Self {
        Self {
            width: 256,
            height: 256,
            num_steps: 20,
            ..Default::default()
        }
    }
}

/// Candle-based Stable Diffusion pipeline
#[cfg(feature = "candle")]
pub struct CandleDiffusionPipeline {
    config: DiffusionConfig,
    device: Device,
    dtype: DType,
    // Model components (loaded lazily)
    text_encoder: Option<stable_diffusion::clip::ClipTextTransformer>,
    tokenizer: Option<tokenizers::Tokenizer>,
    vae: Option<stable_diffusion::vae::AutoEncoderKL>,
    unet: Option<stable_diffusion::unet_2d::UNet2DConditionModel>,
    scheduler_config: stable_diffusion::ddim::DDIMSchedulerConfig,
}

#[cfg(feature = "candle")]
impl CandleDiffusionPipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: DiffusionConfig) -> Result<Self, InferenceError> {
        // Select device (CUDA if available, otherwise CPU)
        let device = Device::cuda_if_available(0)
            .map_err(|e| InferenceError::DeviceError(e.to_string()))?;

        let dtype = if config.use_f16 {
            DType::F16
        } else {
            DType::F32
        };

        // Create scheduler config (scheduler built fresh for each generation)
        let scheduler_config = stable_diffusion::ddim::DDIMSchedulerConfig::default();

        Ok(Self {
            config,
            device,
            dtype,
            text_encoder: None,
            tokenizer: None,
            vae: None,
            unet: None,
            scheduler_config,
        })
    }

    /// Load all model components
    pub fn load(&mut self) -> Result<(), InferenceError> {
        self.load_tokenizer()?;
        self.load_text_encoder()?;
        self.load_vae()?;
        self.load_unet()?;
        Ok(())
    }

    /// Load the tokenizer
    fn load_tokenizer(&mut self) -> Result<(), InferenceError> {
        let tokenizer_path = self.config.model_path.join("tokenizer");
        let tokenizer_json = tokenizer_path.join("tokenizer.json");

        // Try local tokenizer.json first
        if tokenizer_json.exists() {
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_json)
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            self.tokenizer = Some(tokenizer);
            return Ok(());
        }

        // Try to download standard CLIP tokenizer from Hugging Face
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        let repo = match self.config.version {
            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                api.model("openai/clip-vit-large-patch14".to_string())
            }
            _ => api.model("openai/clip-vit-base-patch32".to_string()),
        };

        let tokenizer_file = repo
            .get("tokenizer.json")
            .map_err(|e| InferenceError::ModelLoadError(
                format!("Failed to download CLIP tokenizer: {}", e)
            ))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Load the text encoder (CLIP)
    fn load_text_encoder(&mut self) -> Result<(), InferenceError> {
        let encoder_path = self.config.model_path.join("text_encoder");

        // Try safetensors first, then pytorch .bin
        let safetensors_file = encoder_path.join("model.safetensors");
        let pytorch_file = encoder_path.join("pytorch_model.bin");

        let vb = if safetensors_file.exists() {
            unsafe {
                nn::VarBuilder::from_mmaped_safetensors(
                    &[safetensors_file],
                    self.dtype,
                    &self.device,
                ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
            }
        } else if pytorch_file.exists() {
            nn::VarBuilder::from_pth(&pytorch_file, self.dtype, &self.device)
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
        } else {
            return Err(InferenceError::ModelNotFound(
                format!("No model file found in {}", encoder_path.display())
            ));
        };

        // Load CLIP config and weights
        let clip_config = match self.config.version {
            StableDiffusionVersion::V1_5 => stable_diffusion::clip::Config::v1_5(),
            StableDiffusionVersion::V2_1 => stable_diffusion::clip::Config::v2_1(),
            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                stable_diffusion::clip::Config::sdxl()
            }
        };

        let text_encoder = stable_diffusion::clip::ClipTextTransformer::new(
            vb,
            &clip_config,
        ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        self.text_encoder = Some(text_encoder);
        Ok(())
    }

    /// Load the VAE
    fn load_vae(&mut self) -> Result<(), InferenceError> {
        let vae_path = self.config.model_path.join("vae");

        // Try safetensors first, then pytorch .bin
        let safetensors_file = vae_path.join("diffusion_pytorch_model.safetensors");
        let pytorch_file = vae_path.join("diffusion_pytorch_model.bin");

        let vb = if safetensors_file.exists() {
            unsafe {
                nn::VarBuilder::from_mmaped_safetensors(
                    &[safetensors_file],
                    self.dtype,
                    &self.device,
                ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
            }
        } else if pytorch_file.exists() {
            nn::VarBuilder::from_pth(&pytorch_file, self.dtype, &self.device)
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
        } else {
            return Err(InferenceError::ModelNotFound(
                format!("No model file found in {}", vae_path.display())
            ));
        };

        // VAE config based on version (SDXL has different block channels)
        let vae_config = match self.config.version {
            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                stable_diffusion::vae::AutoEncoderKLConfig {
                    block_out_channels: vec![128, 256, 512, 512],
                    layers_per_block: 2,
                    latent_channels: 4,
                    norm_num_groups: 32,
                    use_quant_conv: true,
                    use_post_quant_conv: true,
                }
            }
            _ => stable_diffusion::vae::AutoEncoderKLConfig::default(),
        };

        let vae = stable_diffusion::vae::AutoEncoderKL::new(
            vb,
            3,  // in_channels
            3,  // out_channels
            vae_config,
        ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        self.vae = Some(vae);
        Ok(())
    }

    /// Load the UNet
    fn load_unet(&mut self) -> Result<(), InferenceError> {
        let unet_path = self.config.model_path.join("unet");

        // Try safetensors first, then pytorch .bin
        let safetensors_file = unet_path.join("diffusion_pytorch_model.safetensors");
        let pytorch_file = unet_path.join("diffusion_pytorch_model.bin");

        let vb = if safetensors_file.exists() {
            unsafe {
                nn::VarBuilder::from_mmaped_safetensors(
                    &[safetensors_file],
                    self.dtype,
                    &self.device,
                ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
            }
        } else if pytorch_file.exists() {
            nn::VarBuilder::from_pth(&pytorch_file, self.dtype, &self.device)
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
        } else {
            return Err(InferenceError::ModelNotFound(
                format!("No model file found in {}", unet_path.display())
            ));
        };

        // Use default UNet config (SD 1.5 style)
        let unet_config = stable_diffusion::unet_2d::UNet2DConditionModelConfig::default();

        let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
            vb,
            4,  // in_channels
            4,  // out_channels
            self.config.use_flash_attn,
            unet_config,
        ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        self.unet = Some(unet);
        Ok(())
    }

    /// Generate an image from a text prompt
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        _seed: u64,
    ) -> Result<GeneratedImage, InferenceError> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or(InferenceError::ModelNotLoaded("tokenizer".into()))?;
        let text_encoder = self.text_encoder.as_ref()
            .ok_or(InferenceError::ModelNotLoaded("text_encoder".into()))?;
        let unet = self.unet.as_ref()
            .ok_or(InferenceError::ModelNotLoaded("unet".into()))?;
        let vae = self.vae.as_ref()
            .ok_or(InferenceError::ModelNotLoaded("vae".into()))?;

        // Build scheduler for this generation
        let mut scheduler = self.scheduler_config.build(self.config.num_steps)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        // Step 1: Encode prompts
        let text_embeddings = self.encode_prompt(prompt, tokenizer, text_encoder)?;

        let text_embeddings = if self.config.guidance_scale > 1.0 {
            let neg_prompt = negative_prompt.unwrap_or("");
            let uncond_embeddings = self.encode_prompt(neg_prompt, tokenizer, text_encoder)?;
            Tensor::cat(&[uncond_embeddings, text_embeddings], 0)
                .map_err(|e| InferenceError::InferenceError(e.to_string()))?
        } else {
            text_embeddings
        };

        // Step 2: Initialize latents
        let latent_height = self.config.height / 8;
        let latent_width = self.config.width / 8;
        let mut latents = self.random_latent(latent_height, latent_width)?;

        // Step 3: Denoise
        let timesteps: Vec<usize> = scheduler.timesteps().to_vec();
        for t in timesteps {
            let latent_input = if self.config.guidance_scale > 1.0 {
                Tensor::cat(&[&latents, &latents], 0)
                    .map_err(|e| InferenceError::InferenceError(e.to_string()))?
            } else {
                latents.clone()
            };

            let noise_pred = unet.forward(&latent_input, t as f64, &text_embeddings)
                .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

            let noise_pred = if self.config.guidance_scale > 1.0 {
                let chunks = noise_pred.chunk(2, 0)
                    .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
                let noise_uncond = &chunks[0];
                let noise_cond = &chunks[1];

                let diff = (noise_cond - noise_uncond)
                    .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
                let scaled = (diff * self.config.guidance_scale)
                    .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
                (noise_uncond + scaled)
                    .map_err(|e| InferenceError::InferenceError(e.to_string()))?
            } else {
                noise_pred
            };

            latents = scheduler.step(&noise_pred, t, &latents)
                .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
        }

        // Step 4: Decode
        let scaled_latents = (&latents / 0.18215)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
        let image = vae.decode(&scaled_latents)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        // Convert to image
        self.tensor_to_image(&image)
    }

    /// Encode a text prompt to embeddings
    fn encode_prompt(
        &self,
        prompt: &str,
        tokenizer: &tokenizers::Tokenizer,
        text_encoder: &stable_diffusion::clip::ClipTextTransformer,
    ) -> Result<Tensor, InferenceError> {
        let tokens = tokenizer.encode(prompt, true)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

        let token_ids: Vec<i64> = tokens.get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();

        // Pad or truncate to 77 tokens
        let mut padded = vec![49407i64; 77]; // CLIP pad token
        let len = token_ids.len().min(77);
        padded[..len].copy_from_slice(&token_ids[..len]);

        let input_ids = Tensor::new(&padded[..], &self.device)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        text_encoder.forward(&input_ids)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))
    }

    /// Create random initial latent
    fn random_latent(
        &self,
        height: usize,
        width: usize,
    ) -> Result<Tensor, InferenceError> {
        let shape = (1, 4, height, width);
        Tensor::randn(0f32, 1f32, shape, &self.device)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))
    }

    /// Convert output tensor to image
    fn tensor_to_image(&self, tensor: &Tensor) -> Result<GeneratedImage, InferenceError> {
        let clamped = tensor.clamp(-1f32, 1f32)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
        let shifted = (&clamped + 1.0)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
        let scaled = (&shifted * 127.5)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
        let tensor = scaled
            .to_dtype(DType::U8)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?
            .squeeze(0)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        let (_channels, height, width) = tensor.dims3()
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        let permuted = tensor.permute((1, 2, 0))
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
        let flattened = permuted.flatten_all()
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
        let data = flattened.to_vec1::<u8>()
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        // Add alpha channel
        let mut rgba = Vec::with_capacity(width * height * 4);
        for chunk in data.chunks(3) {
            rgba.extend_from_slice(chunk);
            rgba.push(255);
        }

        Ok(GeneratedImage {
            width: width as u32,
            height: height as u32,
            data: rgba,
        })
    }
}

/// Stub implementation when candle feature is disabled
#[cfg(not(feature = "candle"))]
pub struct CandleDiffusionPipeline {
    config: DiffusionConfig,
}

#[cfg(not(feature = "candle"))]
impl CandleDiffusionPipeline {
    pub fn new(config: DiffusionConfig) -> Result<Self, InferenceError> {
        Ok(Self { config })
    }

    pub fn load(&mut self) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn generate(
        &self,
        _prompt: &str,
        _negative_prompt: Option<&str>,
        _seed: u64,
    ) -> Result<GeneratedImage, InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }
}

/// Generated image output
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGBA
}

impl GeneratedImage {
    /// Convert to CatSith ImageFrame
    pub fn to_image_frame(&self) -> catsith_core::ImageFrame {
        catsith_core::ImageFrame {
            width: self.width,
            height: self.height,
            format: catsith_core::ImageFormat::Rgba8,
            data: self.data.clone(),
        }
    }

    /// Save to file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), InferenceError> {
        image::save_buffer(
            path,
            &self.data,
            self.width,
            self.height,
            image::ColorType::Rgba8,
        ).map_err(|e| InferenceError::IoError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = DiffusionConfig::default();
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
        assert_eq!(config.num_steps, 20);
    }

    #[test]
    fn test_pixel_art_config() {
        let config = DiffusionConfig::pixel_art();
        assert_eq!(config.width, 64);
        assert_eq!(config.height, 64);
    }

    #[test]
    fn test_version_defaults() {
        assert_eq!(StableDiffusionVersion::V1_5.default_steps(), 20);
        assert_eq!(StableDiffusionVersion::Turbo.default_steps(), 4);
        assert_eq!(StableDiffusionVersion::Turbo.default_guidance_scale(), 0.0);
    }

    #[test]
    #[cfg(feature = "candle")]
    fn test_pipeline_creation() {
        // Test that we can create a pipeline with default config
        let config = DiffusionConfig::default();
        let pipeline = CandleDiffusionPipeline::new(config);
        assert!(pipeline.is_ok());
    }
}
