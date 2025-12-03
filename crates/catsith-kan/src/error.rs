//! Error types for KAN operations

use thiserror::Error;

/// KAN-specific errors
#[derive(Debug, Error)]
pub enum KanError {
    /// GPU initialization failed
    #[error("GPU initialization failed: {0}")]
    GpuInit(String),

    /// No suitable GPU adapter found
    #[error("No suitable GPU adapter found")]
    NoAdapter,

    /// GPU device request failed
    #[error("GPU device request failed: {0}")]
    DeviceRequest(String),

    /// Shader compilation failed
    #[error("Shader compilation failed: {0}")]
    ShaderCompile(String),

    /// Buffer mapping failed
    #[error("Buffer mapping failed")]
    BufferMap,

    /// Dimension mismatch in layer operations
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Invalid network configuration
    #[error("Invalid network configuration: {0}")]
    InvalidConfig(String),

    /// Training failed
    #[error("Training failed: {0}")]
    TrainingFailed(String),

    /// Model serialization/deserialization failed
    #[error("Model IO failed: {0}")]
    ModelIO(String),
}

/// Result type for KAN operations
pub type Result<T> = std::result::Result<T, KanError>;
