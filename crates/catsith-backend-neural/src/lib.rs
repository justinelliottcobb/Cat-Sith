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
//! 3. **Diffusion**: Generates high-quality images (optional)
//!
//! # Model Loading
//!
//! Models are loaded from disk or downloaded on demand.
//! Supported formats: ONNX (via tract)

pub mod embedder;
pub mod inference;
pub mod models;
pub mod renderer;
pub mod temporal;

// Re-export commonly used types
pub use embedder::TextEmbedder;
pub use inference::{InferenceEngine, InferenceError};
pub use renderer::NeuralRenderer;
pub use temporal::TemporalCoherence;
