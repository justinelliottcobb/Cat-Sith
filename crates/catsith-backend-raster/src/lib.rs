//! CatSith Raster Backend
//!
//! Traditional sprite-based 2D rendering using pre-made sprite atlases.

pub mod atlas;
pub mod compositor;
pub mod renderer;

pub use atlas::{SpriteAtlas, SpriteRegion};
pub use compositor::LayerCompositor;
pub use renderer::RasterRenderer;
