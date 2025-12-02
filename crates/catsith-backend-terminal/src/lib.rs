//! CatSith Terminal Backend
//!
//! ASCII/Unicode character-based rendering for terminal output.
//! Supports true color, 256 color, and basic 16-color modes.

pub mod color;
pub mod output;
pub mod renderer;
pub mod sprites;

// Re-export commonly used types
pub use color::ColorMapper;
pub use output::TerminalOutput;
pub use renderer::TerminalRenderer;
pub use sprites::SpriteGenerator;
