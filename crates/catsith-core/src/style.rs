//! Player style preferences
//!
//! Styles control how the semantic scene is interpreted visually,
//! allowing each player to see the same game world differently.

use crate::intent::QualityTier;
use serde::{Deserialize, Serialize};

/// Player's rendering style preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerStyle {
    /// Visual aesthetic preset
    pub aesthetic: Aesthetic,

    /// Color palette preference
    pub palette: Palette,

    /// Animation preferences
    pub animation: AnimationStyle,

    /// Accessibility options
    pub accessibility: Accessibility,

    /// Quality preference (may be overridden by hardware)
    pub preferred_quality: QualityTier,

    /// Custom style parameters
    pub custom: std::collections::HashMap<String, String>,
}

impl Default for PlayerStyle {
    fn default() -> Self {
        Self {
            aesthetic: Aesthetic::default(),
            palette: Palette::default(),
            animation: AnimationStyle::default(),
            accessibility: Accessibility::default(),
            preferred_quality: QualityTier::Medium,
            custom: std::collections::HashMap::new(),
        }
    }
}

impl PlayerStyle {
    /// Create a new style with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Terminal-optimized style
    pub fn terminal() -> Self {
        Self {
            aesthetic: Aesthetic::Retro,
            palette: Palette::HighContrast,
            animation: AnimationStyle {
                frame_rate: 30,
                motion_blur: false,
                particle_density: ParticleDensity::Low,
            },
            preferred_quality: QualityTier::Minimal,
            ..Default::default()
        }
    }

    /// Cinematic/high quality style
    pub fn cinematic() -> Self {
        Self {
            aesthetic: Aesthetic::Photorealistic,
            palette: Palette::Natural,
            animation: AnimationStyle {
                frame_rate: 60,
                motion_blur: true,
                particle_density: ParticleDensity::High,
            },
            preferred_quality: QualityTier::Cinematic,
            ..Default::default()
        }
    }

    /// Set the aesthetic
    pub fn with_aesthetic(mut self, aesthetic: Aesthetic) -> Self {
        self.aesthetic = aesthetic;
        self
    }

    /// Set the palette
    pub fn with_palette(mut self, palette: Palette) -> Self {
        self.palette = palette;
        self
    }

    /// Set preferred quality
    pub fn with_quality(mut self, quality: QualityTier) -> Self {
        self.preferred_quality = quality;
        self
    }

    /// Add a custom parameter
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }
}

/// Visual aesthetic preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Aesthetic {
    /// Clean, sharp pixel art
    #[default]
    PixelArt,
    /// Retro terminal/ASCII
    Retro,
    /// Anime/cel-shaded
    Anime,
    /// Vector/geometric
    Vector,
    /// Painterly/impressionist
    Painterly,
    /// Realistic rendering
    Photorealistic,
    /// Abstract/stylized
    Abstract,
    /// Custom (uses LoRAs)
    Custom,
}

impl Aesthetic {
    /// Get a description for neural rendering prompts
    pub fn description(&self) -> &'static str {
        match self {
            Self::PixelArt => "clean pixel art style, sharp edges, limited palette",
            Self::Retro => "retro computer graphics, ASCII-inspired, terminal aesthetic",
            Self::Anime => "anime style, cel-shaded, bold outlines",
            Self::Vector => "vector graphics, geometric shapes, clean lines",
            Self::Painterly => "painterly style, visible brushstrokes, impressionist",
            Self::Photorealistic => "photorealistic, detailed textures, realistic lighting",
            Self::Abstract => "abstract, stylized, artistic interpretation",
            Self::Custom => "",
        }
    }
}

/// Color palette preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Palette {
    /// Natural, realistic colors
    #[default]
    Natural,
    /// High contrast (accessibility)
    HighContrast,
    /// Warm tones
    Warm,
    /// Cool/blue tones
    Cool,
    /// Neon/cyberpunk
    Neon,
    /// Muted/desaturated
    Muted,
    /// Monochrome
    Monochrome,
    /// Sepia/vintage
    Sepia,
}

impl Palette {
    /// Apply palette adjustment to a color
    pub fn adjust_color(&self, rgb: [u8; 3]) -> [u8; 3] {
        match self {
            Self::Natural => rgb,
            Self::HighContrast => {
                // Increase contrast
                let adjusted: Vec<u8> = rgb
                    .iter()
                    .map(|&c| {
                        let f = c as f32 / 255.0;
                        let adjusted = (f - 0.5) * 1.5 + 0.5;
                        (adjusted.clamp(0.0, 1.0) * 255.0) as u8
                    })
                    .collect();
                [adjusted[0], adjusted[1], adjusted[2]]
            }
            Self::Warm => [
                rgb[0].saturating_add(20),
                rgb[1].saturating_add(10),
                rgb[2].saturating_sub(10),
            ],
            Self::Cool => [
                rgb[0].saturating_sub(10),
                rgb[1].saturating_add(5),
                rgb[2].saturating_add(20),
            ],
            Self::Neon => {
                // Boost saturation
                let max = *rgb.iter().max().unwrap() as f32;
                let min = *rgb.iter().min().unwrap() as f32;
                if max == min {
                    return rgb;
                }
                rgb.map(|c| {
                    let f = c as f32;
                    let boosted = (f - min) / (max - min) * 255.0;
                    boosted.clamp(0.0, 255.0) as u8
                })
            }
            Self::Muted => {
                // Reduce saturation
                let gray = (rgb[0] as u16 + rgb[1] as u16 + rgb[2] as u16) / 3;
                rgb.map(|c| ((c as u16 + gray) / 2) as u8)
            }
            Self::Monochrome => {
                let gray = ((rgb[0] as u32 * 299 + rgb[1] as u32 * 587 + rgb[2] as u32 * 114)
                    / 1000) as u8;
                [gray, gray, gray]
            }
            Self::Sepia => {
                let gray =
                    (rgb[0] as f32 * 0.299 + rgb[1] as f32 * 0.587 + rgb[2] as f32 * 0.114) as u8;
                [
                    (gray as f32 * 1.2).min(255.0) as u8,
                    (gray as f32 * 1.0) as u8,
                    (gray as f32 * 0.8) as u8,
                ]
            }
        }
    }
}

/// Animation style preferences
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AnimationStyle {
    /// Target frame rate
    pub frame_rate: u32,
    /// Enable motion blur
    pub motion_blur: bool,
    /// Particle effect density
    pub particle_density: ParticleDensity,
}

impl Default for AnimationStyle {
    fn default() -> Self {
        Self {
            frame_rate: 60,
            motion_blur: false,
            particle_density: ParticleDensity::Normal,
        }
    }
}

/// Particle effect density
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ParticleDensity {
    /// Minimal particles
    Low,
    /// Normal particle count
    #[default]
    Normal,
    /// Extra particles
    High,
    /// Maximum particles
    Ultra,
}

/// Accessibility options
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Accessibility {
    /// Color blindness mode
    pub color_blind_mode: ColorBlindMode,
    /// Screen shake intensity multiplier (0.0 = disabled)
    pub screen_shake: f32,
    /// Flash intensity multiplier (0.0 = disabled)
    pub flash_intensity: f32,
    /// High contrast mode
    pub high_contrast: bool,
    /// Large text/UI
    pub large_ui: bool,
}

impl Accessibility {
    /// Create default accessibility settings
    pub fn new() -> Self {
        Self {
            color_blind_mode: ColorBlindMode::None,
            screen_shake: 1.0,
            flash_intensity: 1.0,
            high_contrast: false,
            large_ui: false,
        }
    }

    /// Disable screen effects (shake, flash)
    pub fn no_effects() -> Self {
        Self {
            screen_shake: 0.0,
            flash_intensity: 0.0,
            ..Default::default()
        }
    }
}

/// Color blindness simulation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ColorBlindMode {
    #[default]
    None,
    /// Red-green (most common)
    Deuteranopia,
    /// Red-green
    Protanopia,
    /// Blue-yellow
    Tritanopia,
    /// Complete color blindness
    Achromatopsia,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_style() {
        let style = PlayerStyle::default();
        assert_eq!(style.aesthetic, Aesthetic::PixelArt);
        assert_eq!(style.palette, Palette::Natural);
    }

    #[test]
    fn test_terminal_style() {
        let style = PlayerStyle::terminal();
        assert_eq!(style.aesthetic, Aesthetic::Retro);
        assert_eq!(style.preferred_quality, QualityTier::Minimal);
    }

    #[test]
    fn test_palette_monochrome() {
        let palette = Palette::Monochrome;
        let result = palette.adjust_color([255, 0, 0]);
        // Should be grayscale
        assert_eq!(result[0], result[1]);
        assert_eq!(result[1], result[2]);
    }

    #[test]
    fn test_palette_warm() {
        let palette = Palette::Warm;
        let original = [100, 100, 100];
        let result = palette.adjust_color(original);
        // Warm should boost red, reduce blue
        assert!(result[0] > original[0]);
        assert!(result[2] < original[2]);
    }

    #[test]
    fn test_style_builder() {
        let style = PlayerStyle::new()
            .with_aesthetic(Aesthetic::Anime)
            .with_palette(Palette::Neon)
            .with_quality(QualityTier::High)
            .with_custom("theme", "cyberpunk");

        assert_eq!(style.aesthetic, Aesthetic::Anime);
        assert_eq!(style.palette, Palette::Neon);
        assert_eq!(style.custom.get("theme"), Some(&"cyberpunk".to_string()));
    }
}
