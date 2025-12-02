//! Render intent - the contract between scene and renderer
//!
//! RenderIntent captures everything needed to render an entity
//! without specifying HOW to render it.

use crate::entity::SemanticEntity;
use crate::style::PlayerStyle;
use serde::{Deserialize, Serialize};

/// Everything needed to render a single entity
#[derive(Debug, Clone)]
pub struct RenderIntent {
    /// The entity to render
    pub entity: SemanticEntity,

    /// Player identity (if available)
    pub identity: Option<EntityIdentity>,

    /// Player's style preferences
    pub style: PlayerStyle,

    /// Previous frame's render (for temporal coherence)
    pub previous: Option<PreviousRender>,

    /// Target output specification
    pub target: RenderTarget,
}

impl RenderIntent {
    /// Create a new render intent for an entity
    pub fn new(entity: SemanticEntity, style: PlayerStyle, target: RenderTarget) -> Self {
        Self {
            entity,
            identity: None,
            style,
            previous: None,
            target,
        }
    }

    /// Set the entity identity
    pub fn with_identity(mut self, identity: EntityIdentity) -> Self {
        self.identity = Some(identity);
        self
    }

    /// Set the previous render for temporal coherence
    pub fn with_previous(mut self, previous: PreviousRender) -> Self {
        self.previous = Some(previous);
        self
    }
}

/// Cached entity identity for consistent rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityIdentity {
    /// Content hash for cache lookup
    pub hash: [u8; 32],

    /// Display name
    pub name: Option<String>,

    /// Semantic description for neural rendering
    /// e.g., "Crimson fighter with gold trim, battle-scarred hull"
    pub description: Option<String>,

    /// Color preferences
    pub colors: Option<ColorScheme>,

    /// LoRA references for this entity
    pub lora_refs: Vec<LoraRef>,
}

impl EntityIdentity {
    /// Create a new identity with a hash
    pub fn new(hash: [u8; 32]) -> Self {
        Self {
            hash,
            name: None,
            description: None,
            colors: None,
            lora_refs: Vec::new(),
        }
    }

    /// Set the display name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the color scheme
    pub fn with_colors(mut self, colors: ColorScheme) -> Self {
        self.colors = Some(colors);
        self
    }

    /// Add a LoRA reference
    pub fn with_lora(mut self, lora_ref: LoraRef) -> Self {
        self.lora_refs.push(lora_ref);
        self
    }
}

/// Color scheme for an entity
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ColorScheme {
    /// Primary color (RGB)
    pub primary: [u8; 3],
    /// Secondary color (RGB)
    pub secondary: [u8; 3],
    /// Accent color (RGB)
    pub accent: [u8; 3],
}

impl ColorScheme {
    /// Create a new color scheme
    pub fn new(primary: [u8; 3], secondary: [u8; 3], accent: [u8; 3]) -> Self {
        Self {
            primary,
            secondary,
            accent,
        }
    }

    /// Default red team colors
    pub fn red_team() -> Self {
        Self::new([200, 50, 50], [150, 30, 30], [255, 100, 50])
    }

    /// Default blue team colors
    pub fn blue_team() -> Self {
        Self::new([50, 100, 200], [30, 70, 150], [100, 150, 255])
    }

    /// Default green/neutral colors
    pub fn neutral() -> Self {
        Self::new([100, 150, 100], [70, 120, 70], [150, 200, 100])
    }
}

/// Reference to a LoRA to apply
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraRef {
    /// Content hash of the LoRA
    pub content_hash: [u8; 32],
    /// Weight/strength (0.0 - 1.0)
    pub weight: f32,
}

impl LoraRef {
    /// Create a new LoRA reference
    pub fn new(content_hash: [u8; 32], weight: f32) -> Self {
        Self {
            content_hash,
            weight: weight.clamp(0.0, 1.0),
        }
    }
}

/// Previous render information for temporal coherence
#[derive(Debug, Clone)]
pub struct PreviousRender {
    /// Frame ID of previous render
    pub frame_id: u64,

    /// Embedding from previous render (for temporal consistency)
    pub embedding: Option<Vec<f32>>,

    /// Screen position in previous frame
    pub screen_position: Option<[f32; 2]>,
}

impl PreviousRender {
    /// Create a new previous render reference
    pub fn new(frame_id: u64) -> Self {
        Self {
            frame_id,
            embedding: None,
            screen_position: None,
        }
    }

    /// Set the embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Target output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderTarget {
    /// Output width in pixels/cells
    pub width: u32,
    /// Output height in pixels/cells
    pub height: u32,
    /// Output format
    pub format: OutputFormat,
    /// Quality tier
    pub quality: QualityTier,
}

impl RenderTarget {
    /// Create a terminal render target
    pub fn terminal(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: OutputFormat::Terminal {
                colors: ColorDepth::TrueColor,
            },
            quality: QualityTier::Minimal,
        }
    }

    /// Create an image render target
    pub fn image(width: u32, height: u32, quality: QualityTier) -> Self {
        Self {
            width,
            height,
            format: OutputFormat::Image {
                format: ImageFormat::Rgba8,
            },
            quality,
        }
    }

    /// Set the quality tier
    pub fn with_quality(mut self, quality: QualityTier) -> Self {
        self.quality = quality;
        self
    }
}

impl Default for RenderTarget {
    fn default() -> Self {
        Self::terminal(80, 24)
    }
}

/// Output format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Terminal characters with colors
    Terminal { colors: ColorDepth },
    /// Pixel image
    Image { format: ImageFormat },
    /// Raw tensor (for pipeline chaining)
    Tensor,
}

/// Terminal color depth
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ColorDepth {
    /// 2 colors (black/white)
    Monochrome,
    /// 16 ANSI colors
    Basic,
    /// 256 colors
    Extended,
    /// 24-bit RGB
    #[default]
    TrueColor,
}

/// Image format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ImageFormat {
    /// 8-bit RGBA
    #[default]
    Rgba8,
    /// 16-bit RGBA
    Rgba16,
    /// 8-bit RGB (no alpha)
    Rgb8,
}

/// Quality tier for rendering
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum QualityTier {
    /// Minimal - CPU only, fast, low quality
    #[default]
    Minimal,
    /// Low - Small models, fast inference
    Low,
    /// Medium - Balanced quality/performance
    Medium,
    /// High - Large models, slower
    High,
    /// Ultra - Maximum real-time quality
    Ultra,
    /// Cinematic - Offline rendering quality
    Cinematic,
}

impl QualityTier {
    /// Get the recommended sprite size for this quality tier
    pub fn sprite_size(&self) -> u32 {
        match self {
            Self::Minimal => 8,
            Self::Low => 16,
            Self::Medium => 32,
            Self::High => 64,
            Self::Ultra => 128,
            Self::Cinematic => 256,
        }
    }

    /// Does this tier require a GPU?
    pub fn requires_gpu(&self) -> bool {
        matches!(
            self,
            Self::Medium | Self::High | Self::Ultra | Self::Cinematic
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_tier_ordering() {
        assert!(QualityTier::Minimal < QualityTier::Low);
        assert!(QualityTier::Low < QualityTier::Medium);
        assert!(QualityTier::Ultra < QualityTier::Cinematic);
    }

    #[test]
    fn test_render_target_terminal() {
        let target = RenderTarget::terminal(80, 24);
        assert_eq!(target.width, 80);
        assert_eq!(target.height, 24);
        assert!(matches!(target.format, OutputFormat::Terminal { .. }));
    }

    #[test]
    fn test_color_scheme() {
        let scheme = ColorScheme::red_team();
        assert_eq!(scheme.primary[0], 200); // Red channel high
    }
}
