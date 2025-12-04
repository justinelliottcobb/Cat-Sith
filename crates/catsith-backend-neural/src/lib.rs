//! CatSith Neural Backend
//!
//! Neural network inference for high-quality sprite and scene rendering.
//! Supports Stable Diffusion via Candle, VAE-based sprite generation, and text embedders.
//!
//! # Quick Start
//!
//! The recommended approach is using the Candle-based Stable Diffusion pipeline:
//!
//! ```rust,ignore
//! use catsith_backend_neural::{CandleDiffusionPipeline, DiffusionConfig};
//!
//! // Create a pipeline for pixel art generation
//! let config = DiffusionConfig::pixel_art();
//! let mut pipeline = CandleDiffusionPipeline::new(config)?;
//!
//! // Load models (tokenizer downloaded from HuggingFace if needed)
//! pipeline.load()?;
//!
//! // Generate an image
//! let image = pipeline.generate("pixel art cat sprite", None, 42)?;
//! image.save("cat.png")?;
//! ```
//!
//! # Architecture
//!
//! The neural backend supports multiple inference approaches:
//!
//! 1. **Candle Diffusion** (recommended): Full Stable Diffusion pipeline
//!    - Text encoder (CLIP)
//!    - UNet denoiser
//!    - VAE decoder
//!    - DDIM scheduler
//!
//! 2. **Embedder**: Converts semantic descriptions to embeddings
//!
//! 3. **Sprite VAE**: Generates small sprites from embeddings
//!
//! 4. **KAN**: Kolmogorov-Arnold Networks for learned sprite generation (optional)
//!
//! # Model Loading
//!
//! Models are loaded from disk or downloaded on demand.
//! Supported formats:
//! - **Candle** (recommended): safetensors, PyTorch .bin files
//! - **ONNX**: via tract (legacy)
//! - **KAN**: native Rust with WebGPU
//!
//! ## Downloading Models
//!
//! Use the Hugging Face CLI to download Stable Diffusion models:
//!
//! ```bash
//! hf download kohbanye/pixel-art-style --local-dir models/pixel-art-style
//! ```
//!
//! # Features
//!
//! - `candle`: Enable Candle-based Stable Diffusion inference (recommended)
//! - `kan`: Enable KAN-based sprite generation
//!
//! # Supported Stable Diffusion Versions
//!
//! - **SD 1.5**: Most compatible, good for pixel art LoRAs
//! - **SD 2.1**: Improved quality
//! - **SDXL**: High resolution, best quality
//! - **SDXL Turbo**: Fast generation (4 steps)

pub mod candle;
pub mod embedder;
pub mod inference;
#[cfg(feature = "kan")]
pub mod kan;
pub mod models;
pub mod onnx;
pub mod renderer;
pub mod temporal;

// Re-export commonly used types
pub use crate::candle::{CandleDiffusionPipeline, DiffusionConfig, GeneratedImage};
pub use embedder::TextEmbedder;
pub use inference::{InferenceEngine, InferenceError};
pub use onnx::{DiffusionPipeline, OnnxRuntime, SpriteVAE};
pub use renderer::NeuralRenderer;
pub use temporal::TemporalCoherence;
