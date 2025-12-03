//! CatSith KAN - Kolmogorov-Arnold Networks for Sprite Generation
//!
//! This crate provides GPU-accelerated KAN implementations tailored for
//! generating and manipulating sprites. KANs use learnable univariate
//! functions (B-splines) instead of fixed activations, making them
//! particularly effective for smooth, continuous transformations.
//!
//! # Architecture
//!
//! Unlike traditional MLPs:
//! - **MLPs**: Fixed activations on nodes, learnable weights on edges
//! - **KANs**: Learnable B-spline functions on edges, no fixed activations
//!
//! This makes KANs well-suited for:
//! - Smooth color gradients and transitions
//! - Latent space interpolation between sprite states
//! - Style transformation with interpretable learned functions
//!
//! # Sprite Generation
//!
//! The [`SpriteKAN`] architecture maps from a latent code to pixel colors:
//! ```text
//! [latent_code, x, y] -> KAN -> [r, g, b, a]
//! ```
//!
//! This allows generating sprites of any resolution from compact latent codes.

pub mod bspline;
pub mod error;
pub mod gpu;
pub mod kan;
pub mod kan_layer;
pub mod sprite;
pub mod univariate;

// Re-exports
pub use bspline::BSpline;
pub use error::{KanError, Result};
pub use gpu::GpuContext;
pub use kan::KAN;
pub use kan_layer::KANLayer;
pub use sprite::{SpriteKAN, SpriteLatent};
pub use univariate::UnivariateFunction;

/// Shader source code
pub mod shaders {
    pub const BSPLINE: &str = include_str!("shaders/bspline.wgsl");
    pub const UNIVARIATE: &str = include_str!("shaders/univariate.wgsl");
    pub const WEIGHT_UPDATE: &str = include_str!("shaders/weight_update.wgsl");
    pub const KAN_LAYER: &str = include_str!("shaders/kan_layer.wgsl");
    pub const SPRITE_GENERATE: &str = include_str!("shaders/sprite_generate.wgsl");
}
