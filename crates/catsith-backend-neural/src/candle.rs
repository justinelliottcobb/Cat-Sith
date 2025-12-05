//! Candle-based Stable Diffusion inference
//!
//! This module provides GPU-accelerated image generation using Hugging Face's
//! Candle framework with Stable Diffusion models.
//!
//! # Overview
//!
//! The [`CandleDiffusionPipeline`] implements a modular text-to-image pipeline:
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
//! // Add a LoRA for style customization
//! pipeline.add_lora("models/pixel-lora.safetensors", 0.8)?;
//!
//! let image = pipeline.generate(
//!     "pixel art wizard casting spell",
//!     Some("blurry, low quality"),  // negative prompt
//!     42,
//! )?;
//! image.save("wizard.png")?;
//! ```
//!
//! # Modular API
//!
//! Components can be loaded and swapped individually:
//!
//! ```rust,ignore
//! // Load specific components
//! pipeline.load_unet_from("models/other-model/unet")?;
//! pipeline.load_vae_from("models/other-model/vae")?;
//!
//! // LoRA management
//! pipeline.add_lora("style.safetensors", 0.7)?;
//! pipeline.set_lora_weight("style.safetensors", 0.5)?;
//! pipeline.remove_lora("style.safetensors");
//! pipeline.clear_loras();
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

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Module, Tensor};
#[cfg(feature = "candle")]
use candle_nn as nn;
#[cfg(feature = "candle")]
use candle_transformers::models::stable_diffusion::{
    self,
    schedulers::SchedulerConfig,
};

use std::collections::HashMap;
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

    /// Set the model path
    pub fn with_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = path.into();
        self
    }

    /// Set the version
    pub fn with_version(mut self, version: StableDiffusionVersion) -> Self {
        self.version = version;
        self
    }
}

/// LoRA application mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoraMode {
    /// LoRAs are merged into model weights (fast inference, slow to change)
    #[default]
    Fused,
    /// LoRAs are applied dynamically during forward pass (slower inference, fast to change)
    /// Note: Not yet implemented - reserved for future use
    Dynamic,
}

/// A loaded LoRA with its weight/strength
#[derive(Debug, Clone)]
pub struct LoadedLora {
    /// Path to the LoRA file (used as identifier)
    pub path: PathBuf,
    /// Weight/strength of this LoRA (0.0 to 1.0)
    pub weight: f32,
    /// The LoRA tensors keyed by layer name
    #[cfg(feature = "candle")]
    pub tensors: HashMap<String, LoraLayer>,
    #[cfg(not(feature = "candle"))]
    pub tensors: HashMap<String, ()>,
}

/// LoRA layer weights (A and B matrices)
#[cfg(feature = "candle")]
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// Down projection matrix
    pub lora_down: Tensor,
    /// Up projection matrix
    pub lora_up: Tensor,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Rank of this LoRA layer
    pub rank: usize,
}

#[cfg(feature = "candle")]
impl LoraLayer {
    /// Compute the LoRA delta: scale * (down @ up)
    pub fn compute_delta(&self, weight: f32) -> Result<Tensor, candle_core::Error> {
        let scale = (self.alpha / self.rank as f32) * weight;
        let delta = self.lora_down.matmul(&self.lora_up)?;
        delta * scale as f64
    }
}

#[cfg(not(feature = "candle"))]
#[derive(Debug, Clone)]
pub struct LoraLayer;

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
    // LoRA management
    loras: Vec<LoadedLora>,
    lora_mode: LoraMode,
    loras_applied: bool,
    // Cached paths for model reconstruction
    unet_path: Option<PathBuf>,
    text_encoder_path: Option<PathBuf>,
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
            loras: Vec::new(),
            lora_mode: LoraMode::default(),
            loras_applied: false,
            unet_path: None,
            text_encoder_path: None,
        })
    }

    // ==================== Configuration Access ====================

    /// Get the current configuration
    pub fn config(&self) -> &DiffusionConfig {
        &self.config
    }

    /// Get mutable access to configuration
    pub fn config_mut(&mut self) -> &mut DiffusionConfig {
        &mut self.config
    }

    /// Get the device being used
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if all components are loaded
    pub fn is_loaded(&self) -> bool {
        self.tokenizer.is_some()
            && self.text_encoder.is_some()
            && self.vae.is_some()
            && self.unet.is_some()
    }

    // ==================== Full Model Loading ====================

    /// Load all model components from config.model_path
    pub fn load(&mut self) -> Result<(), InferenceError> {
        self.load_tokenizer()?;
        self.load_text_encoder()?;
        self.load_vae()?;
        self.load_unet()?;
        Ok(())
    }

    /// Load a complete checkpoint from a new path
    ///
    /// This reloads UNet, VAE, and text encoder from the new path.
    /// Tokenizer is kept if compatible, otherwise reloaded.
    pub fn load_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<(), InferenceError> {
        self.config.model_path = path.as_ref().to_path_buf();
        self.load_text_encoder()?;
        self.load_vae()?;
        self.load_unet()?;
        // Clear LoRAs when switching checkpoints (they may be incompatible)
        self.loras.clear();
        Ok(())
    }

    // ==================== Individual Component Loading ====================

    /// Load tokenizer from config.model_path or download from HuggingFace
    pub fn load_tokenizer(&mut self) -> Result<(), InferenceError> {
        let tokenizer_path = self.config.model_path.join("tokenizer");
        self.load_tokenizer_from(&tokenizer_path)
    }

    /// Load tokenizer from a specific path
    pub fn load_tokenizer_from(&mut self, path: impl AsRef<Path>) -> Result<(), InferenceError> {
        let path = path.as_ref();
        let tokenizer_json = path.join("tokenizer.json");
        let vocab_json = path.join("vocab.json");
        let merges_txt = path.join("merges.txt");

        // Try local tokenizer.json first
        if tokenizer_json.exists() {
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_json)
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            self.tokenizer = Some(tokenizer);
            return Ok(());
        }

        // Try vocab.json + merges.txt (older CLIP format)
        if vocab_json.exists() && merges_txt.exists() {
            let tokenizer = Self::load_clip_tokenizer(&vocab_json, &merges_txt)?;
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

    /// Load CLIP tokenizer from vocab.json and merges.txt
    fn load_clip_tokenizer(
        vocab_path: &Path,
        merges_path: &Path,
    ) -> Result<tokenizers::Tokenizer, InferenceError> {
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::processors::template::TemplateProcessing;

        // Load BPE model from vocab and merges files
        let bpe = BPE::from_file(
            vocab_path.to_str().ok_or_else(|| InferenceError::ModelLoadError("Invalid vocab path".into()))?,
            merges_path.to_str().ok_or_else(|| InferenceError::ModelLoadError("Invalid merges path".into()))?,
        )
        .unk_token("<|endoftext|>".to_string())
        .build()
        .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

        let mut tokenizer = tokenizers::Tokenizer::new(bpe);

        // Set up CLIP-style pre-tokenization
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));

        // Set up post-processing with start/end tokens
        // CLIP uses <|startoftext|> (49406) and <|endoftext|> (49407)
        let template = TemplateProcessing::builder()
            .try_single("<|startoftext|> $A <|endoftext|>")
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?
            .special_tokens(vec![
                ("<|startoftext|>", 49406),
                ("<|endoftext|>", 49407),
            ])
            .build()
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;
        tokenizer.with_post_processor(Some(template));

        Ok(tokenizer)
    }

    /// Load text encoder from config.model_path
    pub fn load_text_encoder(&mut self) -> Result<(), InferenceError> {
        let encoder_path = self.config.model_path.join("text_encoder");
        self.load_text_encoder_from(&encoder_path)
    }

    /// Load text encoder from a specific path
    pub fn load_text_encoder_from(&mut self, path: impl AsRef<Path>) -> Result<(), InferenceError> {
        let path = path.as_ref();
        self.text_encoder_path = Some(path.to_path_buf());

        let vb = Self::load_weights(path, self.dtype, &self.device)?;

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
        self.loras_applied = false; // Text encoder changed, LoRAs need re-apply
        Ok(())
    }

    /// Load VAE from config.model_path
    pub fn load_vae(&mut self) -> Result<(), InferenceError> {
        let vae_path = self.config.model_path.join("vae");
        self.load_vae_from(&vae_path)
    }

    /// Load VAE from a specific path
    pub fn load_vae_from(&mut self, path: impl AsRef<Path>) -> Result<(), InferenceError> {
        let path = path.as_ref();
        let vb = Self::load_weights(path, self.dtype, &self.device)?;

        // Try to load VAE config from config.json
        let vae_config = self.load_vae_config(path)?;

        let vae = stable_diffusion::vae::AutoEncoderKL::new(
            vb,
            3,  // in_channels
            3,  // out_channels
            vae_config,
        ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        self.vae = Some(vae);
        Ok(())
    }

    /// Load VAE config from config.json or use defaults
    fn load_vae_config(&self, path: &Path) -> Result<stable_diffusion::vae::AutoEncoderKLConfig, InferenceError> {
        let config_path = path.join("config.json");

        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| InferenceError::IoError(e.to_string()))?;

            // Parse just the fields we need
            #[derive(serde::Deserialize)]
            struct VaeConfigJson {
                #[serde(default)]
                block_out_channels: Option<Vec<usize>>,
                #[serde(default)]
                layers_per_block: Option<usize>,
                #[serde(default)]
                latent_channels: Option<usize>,
                #[serde(default)]
                norm_num_groups: Option<usize>,
            }

            if let Ok(json_config) = serde_json::from_str::<VaeConfigJson>(&config_str) {
                return Ok(stable_diffusion::vae::AutoEncoderKLConfig {
                    block_out_channels: json_config.block_out_channels.unwrap_or_else(|| vec![64, 128, 256, 512]),
                    layers_per_block: json_config.layers_per_block.unwrap_or(2),
                    latent_channels: json_config.latent_channels.unwrap_or(4),
                    norm_num_groups: json_config.norm_num_groups.unwrap_or(32),
                    use_quant_conv: true,
                    use_post_quant_conv: true,
                });
            }
        }

        // Fall back to version-based defaults
        Ok(match self.config.version {
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
        })
    }

    /// Load UNet from config.model_path
    pub fn load_unet(&mut self) -> Result<(), InferenceError> {
        let unet_path = self.config.model_path.join("unet");
        self.load_unet_from(&unet_path)
    }

    /// Load UNet from a specific path
    pub fn load_unet_from(&mut self, path: impl AsRef<Path>) -> Result<(), InferenceError> {
        let path = path.as_ref();
        self.unet_path = Some(path.to_path_buf());

        let vb = Self::load_weights(path, self.dtype, &self.device)?;

        // Load UNet config from config.json or use defaults
        let unet_config = self.load_unet_config(path)?;

        let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
            vb,
            4,  // in_channels
            4,  // out_channels
            self.config.use_flash_attn,
            unet_config,
        ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        self.unet = Some(unet);
        self.loras_applied = false; // UNet changed, LoRAs need re-apply
        Ok(())
    }

    /// Load UNet config from config.json or use defaults
    fn load_unet_config(&self, path: &Path) -> Result<stable_diffusion::unet_2d::UNet2DConditionModelConfig, InferenceError> {
        use stable_diffusion::unet_2d::{UNet2DConditionModelConfig, BlockConfig};

        let config_path = path.join("config.json");

        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| InferenceError::IoError(e.to_string()))?;

            #[derive(serde::Deserialize)]
            struct UNetConfigJson {
                #[serde(default)]
                block_out_channels: Option<Vec<usize>>,
                #[serde(default)]
                cross_attention_dim: Option<usize>,
                #[serde(default)]
                attention_head_dim: Option<usize>,
                #[serde(default)]
                layers_per_block: Option<usize>,
                #[serde(default)]
                use_linear_projection: Option<bool>,
            }

            if let Ok(json_config) = serde_json::from_str::<UNetConfigJson>(&config_str) {
                let blocks = json_config.block_out_channels.unwrap_or_else(|| vec![320, 640, 1280, 1280]);
                let n_blocks = blocks.len();
                let attention_head_dim = json_config.attention_head_dim.unwrap_or(8);

                return Ok(UNet2DConditionModelConfig {
                    blocks: blocks.into_iter().enumerate().map(|(i, out_channels)| {
                        BlockConfig {
                            out_channels,
                            // Last block has no attention, others use 1 transformer block
                            use_cross_attn: if i < n_blocks - 1 { Some(1) } else { None },
                            attention_head_dim,
                        }
                    }).collect(),
                    center_input_sample: false,
                    cross_attention_dim: json_config.cross_attention_dim.unwrap_or(768),
                    downsample_padding: 1,
                    flip_sin_to_cos: true,
                    freq_shift: 0.0,
                    layers_per_block: json_config.layers_per_block.unwrap_or(2),
                    mid_block_scale_factor: 1.0,
                    norm_eps: 1e-5,
                    norm_num_groups: 32,
                    sliced_attention_size: None,
                    use_linear_projection: json_config.use_linear_projection.unwrap_or(false),
                });
            }
        }

        // Fall back to SD 1.5 default
        Ok(stable_diffusion::unet_2d::UNet2DConditionModelConfig::default())
    }

    /// Helper to load weights from a directory (tries safetensors then .bin)
    fn load_weights(
        path: &Path,
        dtype: DType,
        device: &Device,
    ) -> Result<nn::VarBuilder<'static>, InferenceError> {
        // Common file names to try
        let safetensors_names = [
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
        ];
        let pytorch_names = [
            "pytorch_model.bin",
            "diffusion_pytorch_model.bin",
        ];

        // Try safetensors first
        for name in &safetensors_names {
            let file = path.join(name);
            if file.exists() {
                return unsafe {
                    nn::VarBuilder::from_mmaped_safetensors(
                        &[file],
                        dtype,
                        device,
                    ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))
                };
            }
        }

        // Try pytorch .bin
        for name in &pytorch_names {
            let file = path.join(name);
            if file.exists() {
                return nn::VarBuilder::from_pth(&file, dtype, device)
                    .map_err(|e| InferenceError::ModelLoadError(e.to_string()));
            }
        }

        Err(InferenceError::ModelNotFound(
            format!("No model file found in {}", path.display())
        ))
    }

    // ==================== LoRA Management ====================

    /// Add a LoRA from a safetensors file
    ///
    /// The LoRA will be applied during generation with the specified weight.
    pub fn add_lora(&mut self, path: impl AsRef<Path>, weight: f32) -> Result<(), InferenceError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(InferenceError::ModelNotFound(path.display().to_string()));
        }

        // Check if already loaded
        if self.loras.iter().any(|l| l.path == path) {
            return self.set_lora_weight(path, weight);
        }

        let tensors = Self::load_lora_tensors(path, self.dtype, &self.device)?;

        self.loras.push(LoadedLora {
            path: path.to_path_buf(),
            weight: weight.clamp(0.0, 1.0),
            tensors,
        });

        Ok(())
    }

    /// Set the weight of an already-loaded LoRA
    pub fn set_lora_weight(&mut self, path: impl AsRef<Path>, weight: f32) -> Result<(), InferenceError> {
        let path = path.as_ref();

        for lora in &mut self.loras {
            if lora.path == path {
                lora.weight = weight.clamp(0.0, 1.0);
                return Ok(());
            }
        }

        Err(InferenceError::ModelNotFound(
            format!("LoRA not loaded: {}", path.display())
        ))
    }

    /// Remove a LoRA by path
    pub fn remove_lora(&mut self, path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        let initial_len = self.loras.len();
        self.loras.retain(|l| l.path != path);
        self.loras.len() < initial_len
    }

    /// Clear all loaded LoRAs
    pub fn clear_loras(&mut self) {
        self.loras.clear();
    }

    /// Get the list of loaded LoRAs
    pub fn loras(&self) -> &[LoadedLora] {
        &self.loras
    }

    /// Get the number of loaded LoRAs
    pub fn lora_count(&self) -> usize {
        self.loras.len()
    }

    /// Get the current LoRA mode
    pub fn lora_mode(&self) -> LoraMode {
        self.lora_mode
    }

    /// Set the LoRA mode
    ///
    /// Note: Only `LoraMode::Fused` is currently supported.
    /// Setting to `LoraMode::Dynamic` will return an error.
    pub fn set_lora_mode(&mut self, mode: LoraMode) -> Result<(), InferenceError> {
        match mode {
            LoraMode::Fused => {
                self.lora_mode = mode;
                Ok(())
            }
            LoraMode::Dynamic => {
                Err(InferenceError::FeatureDisabled(
                    "Dynamic LoRA mode not yet implemented".to_string()
                ))
            }
        }
    }

    /// Check if LoRAs have been applied to the model weights
    pub fn are_loras_applied(&self) -> bool {
        self.loras_applied
    }

    /// Apply all loaded LoRAs to the model weights (Fused mode)
    ///
    /// This merges LoRA weights into the base model weights. After calling this,
    /// generation will use the merged weights with zero per-inference overhead.
    ///
    /// The original weights can be restored by calling `unapply_loras()`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// pipeline.add_lora("style.safetensors", 0.8)?;
    /// pipeline.add_lora("detail.safetensors", 0.5)?;
    /// pipeline.apply_loras()?;  // Merge into model weights
    ///
    /// // Generate many images with no LoRA overhead
    /// for seed in 0..100 {
    ///     pipeline.generate("pixel art warrior", None, seed)?;
    /// }
    /// ```
    pub fn apply_loras(&mut self) -> Result<(), InferenceError> {
        if self.loras.is_empty() {
            return Ok(()); // Nothing to apply
        }

        if self.loras_applied {
            // Already applied - unapply first to reset to base weights
            self.unapply_loras()?;
        }

        // Clone the LoRA data we need to avoid borrow conflicts
        let lora_data: Vec<(f32, HashMap<String, LoraLayer>)> = self.loras
            .iter()
            .map(|l| (l.weight, l.tensors.clone()))
            .collect();

        // Collect all LoRA layers with their weights
        let mut lora_deltas: HashMap<String, Vec<(f32, LoraLayer)>> = HashMap::new();
        for (weight, tensors) in lora_data {
            for (layer_name, layer) in tensors {
                lora_deltas
                    .entry(layer_name)
                    .or_default()
                    .push((weight, layer));
            }
        }

        // Apply LoRAs to UNet if loaded
        if let Some(unet_path) = self.unet_path.clone() {
            self.apply_loras_to_unet(&unet_path, &lora_deltas)?;
        }

        // Apply LoRAs to text encoder if loaded
        if let Some(te_path) = self.text_encoder_path.clone() {
            self.apply_loras_to_text_encoder(&te_path, &lora_deltas)?;
        }

        self.loras_applied = true;
        Ok(())
    }

    /// Apply LoRA deltas to UNet weights and rebuild
    fn apply_loras_to_unet(
        &mut self,
        base_path: &Path,
        lora_deltas: &HashMap<String, Vec<(f32, LoraLayer)>>,
    ) -> Result<(), InferenceError> {
        // Load base weights as HashMap
        let base_tensors = Self::load_tensors_as_map(base_path, self.dtype, &self.device)?;

        // Apply LoRA deltas
        let merged_tensors = self.merge_lora_weights(base_tensors, lora_deltas, "unet")?;

        // Rebuild UNet from merged weights
        let vb = nn::VarBuilder::from_tensors(merged_tensors, self.dtype, &self.device);

        let unet_config = stable_diffusion::unet_2d::UNet2DConditionModelConfig::default();
        let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
            vb,
            4,
            4,
            self.config.use_flash_attn,
            unet_config,
        ).map_err(|e| InferenceError::ModelLoadError(
            format!("Failed to rebuild UNet with LoRA weights: {}", e)
        ))?;

        self.unet = Some(unet);
        Ok(())
    }

    /// Apply LoRA deltas to text encoder weights and rebuild
    fn apply_loras_to_text_encoder(
        &mut self,
        base_path: &Path,
        lora_deltas: &HashMap<String, Vec<(f32, LoraLayer)>>,
    ) -> Result<(), InferenceError> {
        // Load base weights as HashMap
        let base_tensors = Self::load_tensors_as_map(base_path, self.dtype, &self.device)?;

        // Apply LoRA deltas
        let merged_tensors = self.merge_lora_weights(base_tensors, lora_deltas, "text_encoder")?;

        // Rebuild text encoder from merged weights
        let vb = nn::VarBuilder::from_tensors(merged_tensors, self.dtype, &self.device);

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
        ).map_err(|e| InferenceError::ModelLoadError(
            format!("Failed to rebuild text encoder with LoRA weights: {}", e)
        ))?;

        self.text_encoder = Some(text_encoder);
        Ok(())
    }

    /// Merge LoRA deltas into base weights
    fn merge_lora_weights(
        &self,
        mut base_tensors: HashMap<String, Tensor>,
        lora_deltas: &HashMap<String, Vec<(f32, LoraLayer)>>,
        prefix: &str,
    ) -> Result<HashMap<String, Tensor>, InferenceError> {
        for (layer_name, layers) in lora_deltas {
            // Skip layers that don't match our prefix (unet vs text_encoder)
            if !layer_name.starts_with(prefix) {
                continue;
            }

            // Find the corresponding base weight
            // LoRA names like "unet.down_blocks.0.attentions.0.to_q" need to map to
            // the actual weight tensor name (often with ".weight" suffix)
            let weight_key = format!("{}.weight", layer_name);
            let alt_key = layer_name.clone();

            let base_key = if base_tensors.contains_key(&weight_key) {
                weight_key
            } else if base_tensors.contains_key(&alt_key) {
                alt_key
            } else {
                // Try to find a matching key with different naming convention
                let short_name = layer_name.strip_prefix(&format!("{}.", prefix))
                    .unwrap_or(layer_name);
                if let Some(key) = base_tensors.keys()
                    .find(|k| k.contains(short_name))
                    .cloned()
                {
                    key
                } else {
                    // No matching base weight found - skip this LoRA layer
                    continue;
                }
            };

            // Compute combined delta from all LoRAs affecting this layer
            let mut combined_delta: Option<Tensor> = None;
            for (weight, layer) in layers {
                let delta = layer.compute_delta(*weight)
                    .map_err(|e| InferenceError::InferenceError(
                        format!("Failed to compute LoRA delta: {}", e)
                    ))?;

                combined_delta = Some(match combined_delta {
                    Some(existing) => (&existing + &delta)
                        .map_err(|e| InferenceError::InferenceError(e.to_string()))?,
                    None => delta,
                });
            }

            // Add delta to base weights
            if let Some(delta) = combined_delta {
                if let Some(base_weight) = base_tensors.get(&base_key) {
                    // Reshape delta if needed to match base weight shape
                    let merged = (base_weight + &delta)
                        .map_err(|e| InferenceError::InferenceError(
                            format!("Failed to merge LoRA into {}: {}", base_key, e)
                        ))?;
                    base_tensors.insert(base_key, merged);
                }
            }
        }

        Ok(base_tensors)
    }

    /// Load tensors from a model file as a HashMap
    fn load_tensors_as_map(
        path: &Path,
        dtype: DType,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>, InferenceError> {
        let safetensors_names = [
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
        ];

        // Try safetensors
        for name in &safetensors_names {
            let file = path.join(name);
            if file.exists() {
                let tensors = candle_core::safetensors::load(&file, device)
                    .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

                // Convert to target dtype
                let mut result = HashMap::new();
                for (name, tensor) in tensors {
                    let converted = tensor.to_dtype(dtype)
                        .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
                    result.insert(name, converted);
                }
                return Ok(result);
            }
        }

        Err(InferenceError::ModelNotFound(
            format!("No safetensors file found in {}", path.display())
        ))
    }

    /// Unapply LoRAs and restore original model weights
    ///
    /// This reloads the models from their original paths, discarding
    /// any merged LoRA weights.
    pub fn unapply_loras(&mut self) -> Result<(), InferenceError> {
        if !self.loras_applied {
            return Ok(()); // Nothing to unapply
        }

        // Reload UNet from original path
        if let Some(path) = self.unet_path.clone() {
            let vb = Self::load_weights(&path, self.dtype, &self.device)?;
            let unet_config = stable_diffusion::unet_2d::UNet2DConditionModelConfig::default();
            let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
                vb,
                4,
                4,
                self.config.use_flash_attn,
                unet_config,
            ).map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            self.unet = Some(unet);
        }

        // Reload text encoder from original path
        if let Some(path) = self.text_encoder_path.clone() {
            let vb = Self::load_weights(&path, self.dtype, &self.device)?;
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
        }

        self.loras_applied = false;
        Ok(())
    }

    /// Load LoRA tensors from a safetensors file
    fn load_lora_tensors(
        path: &Path,
        dtype: DType,
        device: &Device,
    ) -> Result<HashMap<String, LoraLayer>, InferenceError> {
        let tensors = candle_core::safetensors::load(path, device)
            .map_err(|e| InferenceError::ModelLoadError(
                format!("Failed to load LoRA: {}", e)
            ))?;

        let mut layers = HashMap::new();

        // Parse LoRA tensors - they come in pairs: layer.lora_down.weight, layer.lora_up.weight
        // Also look for alpha values
        let mut lora_downs: HashMap<String, Tensor> = HashMap::new();
        let mut lora_ups: HashMap<String, Tensor> = HashMap::new();
        let mut alphas: HashMap<String, f32> = HashMap::new();

        for (name, tensor) in tensors {
            let tensor = tensor.to_dtype(dtype)
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

            if name.contains(".lora_down.") || name.ends_with(".lora_A.weight") {
                let layer_name = extract_layer_name(&name);
                lora_downs.insert(layer_name, tensor);
            } else if name.contains(".lora_up.") || name.ends_with(".lora_B.weight") {
                let layer_name = extract_layer_name(&name);
                lora_ups.insert(layer_name, tensor);
            } else if name.contains(".alpha") {
                let layer_name = extract_layer_name(&name);
                // Alpha is typically stored as a scalar tensor
                if let Ok(alpha) = tensor.to_scalar::<f32>() {
                    alphas.insert(layer_name, alpha);
                }
            }
        }

        // Match up the pairs
        for (layer_name, lora_down) in lora_downs {
            if let Some(lora_up) = lora_ups.remove(&layer_name) {
                let rank = lora_down.dim(0)
                    .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
                let alpha = alphas.get(&layer_name).copied().unwrap_or(rank as f32);

                layers.insert(layer_name, LoraLayer {
                    lora_down,
                    lora_up,
                    alpha,
                    rank,
                });
            }
        }

        if layers.is_empty() {
            return Err(InferenceError::ModelLoadError(
                "No valid LoRA layers found in file".to_string()
            ));
        }

        Ok(layers)
    }

    // ==================== Generation ====================

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

            // Note: LoRA application would happen here during UNet forward pass
            // For now, we use the base UNet - full LoRA integration requires
            // modifying the UNet forward pass to apply LoRA deltas
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
        let latent = Tensor::randn(0f32, 1f32, shape, &self.device)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;
        // Convert to model dtype
        latent.to_dtype(self.dtype)
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

/// Extract the base layer name from a LoRA tensor name
#[cfg(feature = "candle")]
fn extract_layer_name(name: &str) -> String {
    // Remove common suffixes to get the layer name
    name.replace(".lora_down.weight", "")
        .replace(".lora_up.weight", "")
        .replace(".lora_A.weight", "")
        .replace(".lora_B.weight", "")
        .replace(".alpha", "")
}

/// Stub implementation when candle feature is disabled
#[cfg(not(feature = "candle"))]
pub struct CandleDiffusionPipeline {
    config: DiffusionConfig,
    loras: Vec<LoadedLora>,
}

#[cfg(not(feature = "candle"))]
impl CandleDiffusionPipeline {
    pub fn new(config: DiffusionConfig) -> Result<Self, InferenceError> {
        Ok(Self { config, loras: Vec::new() })
    }

    pub fn config(&self) -> &DiffusionConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut DiffusionConfig {
        &mut self.config
    }

    pub fn is_loaded(&self) -> bool {
        false
    }

    pub fn load(&mut self) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn load_checkpoint(&mut self, _path: impl AsRef<Path>) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn load_tokenizer(&mut self) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn load_tokenizer_from(&mut self, _path: impl AsRef<Path>) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn load_text_encoder(&mut self) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn load_text_encoder_from(&mut self, _path: impl AsRef<Path>) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn load_vae(&mut self) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn load_vae_from(&mut self, _path: impl AsRef<Path>) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn load_unet(&mut self) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn load_unet_from(&mut self, _path: impl AsRef<Path>) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn add_lora(&mut self, _path: impl AsRef<Path>, _weight: f32) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn set_lora_weight(&mut self, _path: impl AsRef<Path>, _weight: f32) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn remove_lora(&mut self, _path: impl AsRef<Path>) -> bool {
        false
    }

    pub fn clear_loras(&mut self) {
        self.loras.clear();
    }

    pub fn loras(&self) -> &[LoadedLora] {
        &self.loras
    }

    pub fn lora_count(&self) -> usize {
        0
    }

    pub fn lora_mode(&self) -> LoraMode {
        LoraMode::Fused
    }

    pub fn set_lora_mode(&mut self, _mode: LoraMode) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn are_loras_applied(&self) -> bool {
        false
    }

    pub fn apply_loras(&mut self) -> Result<(), InferenceError> {
        Err(InferenceError::FeatureDisabled("candle".into()))
    }

    pub fn unapply_loras(&mut self) -> Result<(), InferenceError> {
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
    fn test_config_builder() {
        let config = DiffusionConfig::pixel_art()
            .with_model_path("custom/path")
            .with_version(StableDiffusionVersion::Xl);

        assert_eq!(config.model_path, PathBuf::from("custom/path"));
        assert_eq!(config.version, StableDiffusionVersion::Xl);
    }

    #[test]
    #[cfg(feature = "candle")]
    fn test_pipeline_creation() {
        let config = DiffusionConfig::default();
        let pipeline = CandleDiffusionPipeline::new(config);
        assert!(pipeline.is_ok());

        let pipeline = pipeline.unwrap();
        assert!(!pipeline.is_loaded());
        assert_eq!(pipeline.lora_count(), 0);
    }

    #[test]
    #[cfg(feature = "candle")]
    fn test_lora_management() {
        let config = DiffusionConfig::default();
        let mut pipeline = CandleDiffusionPipeline::new(config).unwrap();

        // Can't add non-existent LoRA
        let result = pipeline.add_lora("nonexistent.safetensors", 0.5);
        assert!(result.is_err());

        // Clear works on empty
        pipeline.clear_loras();
        assert_eq!(pipeline.lora_count(), 0);
    }

    #[test]
    #[cfg(feature = "candle")]
    fn test_extract_layer_name() {
        assert_eq!(
            extract_layer_name("unet.down_blocks.0.attentions.0.lora_down.weight"),
            "unet.down_blocks.0.attentions.0"
        );
        assert_eq!(
            extract_layer_name("text_encoder.encoder.layers.0.lora_A.weight"),
            "text_encoder.encoder.layers.0"
        );
    }

    #[test]
    fn test_lora_mode_default() {
        assert_eq!(LoraMode::default(), LoraMode::Fused);
    }

    #[test]
    #[cfg(feature = "candle")]
    fn test_lora_mode_fused_only() {
        let config = DiffusionConfig::default();
        let mut pipeline = CandleDiffusionPipeline::new(config).unwrap();

        // Default mode is Fused
        assert_eq!(pipeline.lora_mode(), LoraMode::Fused);

        // Setting to Fused should succeed
        assert!(pipeline.set_lora_mode(LoraMode::Fused).is_ok());

        // Setting to Dynamic should fail (not implemented)
        assert!(pipeline.set_lora_mode(LoraMode::Dynamic).is_err());
    }

    #[test]
    #[cfg(feature = "candle")]
    fn test_lora_applied_state() {
        let config = DiffusionConfig::default();
        let mut pipeline = CandleDiffusionPipeline::new(config).unwrap();

        // Initially LoRAs are not applied
        assert!(!pipeline.are_loras_applied());

        // Apply with empty LoRA list should succeed
        assert!(pipeline.apply_loras().is_ok());
        // Still not applied since there were no LoRAs
        assert!(!pipeline.are_loras_applied());

        // Unapply should succeed even when not applied
        assert!(pipeline.unapply_loras().is_ok());
    }
}
