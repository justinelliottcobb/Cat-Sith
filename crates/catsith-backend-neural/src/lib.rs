//! CatSith Neural Backend
//!
//! Neural network inference for high-quality sprite and scene rendering.
//! Supports VAE-based sprite generation, diffusion models, and text embedders.
//!
//! # Architecture
//!
//! The neural backend uses a multi-model approach:
//!
//! 1. **Embedder**: Converts semantic descriptions to embeddings
//! 2. **Sprite VAE**: Generates small sprites from embeddings
//! 3. **Diffusion**: Generates high-quality images via Candle/Stable Diffusion
//! 4. **KAN**: Kolmogorov-Arnold Networks for learned sprite generation (optional)
//!
//! # Model Loading
//!
//! Models are loaded from disk or downloaded on demand.
//! Supported formats:
//! - Candle: safetensors, PyTorch .bin files (recommended)
//! - ONNX: via tract (legacy)
//! - KAN: native Rust with WebGPU
//!
//! # Features
//!
//! - `candle`: Enable Candle-based Stable Diffusion inference (recommended)
//! - `kan`: Enable KAN-based sprite generation

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
