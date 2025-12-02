//! Color mapping for terminal output
//!
//! Maps RGB colors to terminal color codes based on color depth.

use catsith_core::intent::ColorDepth;
use catsith_core::style::Palette;

/// Color mapper for terminal output
#[derive(Debug, Clone)]
pub struct ColorMapper {
    /// Color depth to use
    depth: ColorDepth,
    /// Palette adjustments
    palette: Palette,
}

impl ColorMapper {
    /// Create a new color mapper
    pub fn new(depth: ColorDepth) -> Self {
        Self {
            depth,
            palette: Palette::Natural,
        }
    }

    /// Set the palette
    pub fn with_palette(mut self, palette: Palette) -> Self {
        self.palette = palette;
        self
    }

    /// Map RGB color to terminal color based on depth
    pub fn map_color(&self, rgb: [u8; 3]) -> MappedColor {
        let adjusted = self.palette.adjust_color(rgb);

        match self.depth {
            ColorDepth::Monochrome => self.to_monochrome(adjusted),
            ColorDepth::Basic => self.to_basic_16(adjusted),
            ColorDepth::Extended => self.to_256(adjusted),
            ColorDepth::TrueColor => MappedColor::TrueColor(adjusted),
        }
    }

    /// Convert to monochrome (on/off)
    fn to_monochrome(&self, rgb: [u8; 3]) -> MappedColor {
        let brightness = (rgb[0] as u16 + rgb[1] as u16 + rgb[2] as u16) / 3;
        if brightness > 127 {
            MappedColor::Basic(15) // White
        } else {
            MappedColor::Basic(0) // Black
        }
    }

    /// Convert to 16-color ANSI
    fn to_basic_16(&self, rgb: [u8; 3]) -> MappedColor {
        // Find closest basic color
        let basic_colors: [(u8, [u8; 3]); 16] = [
            (0, [0, 0, 0]),        // Black
            (1, [128, 0, 0]),      // Dark Red
            (2, [0, 128, 0]),      // Dark Green
            (3, [128, 128, 0]),    // Dark Yellow
            (4, [0, 0, 128]),      // Dark Blue
            (5, [128, 0, 128]),    // Dark Magenta
            (6, [0, 128, 128]),    // Dark Cyan
            (7, [192, 192, 192]),  // Light Gray
            (8, [128, 128, 128]),  // Dark Gray
            (9, [255, 0, 0]),      // Red
            (10, [0, 255, 0]),     // Green
            (11, [255, 255, 0]),   // Yellow
            (12, [0, 0, 255]),     // Blue
            (13, [255, 0, 255]),   // Magenta
            (14, [0, 255, 255]),   // Cyan
            (15, [255, 255, 255]), // White
        ];

        let mut best_idx = 0;
        let mut best_dist = u32::MAX;

        for (idx, color) in &basic_colors {
            let dist = color_distance(rgb, *color);
            if dist < best_dist {
                best_dist = dist;
                best_idx = *idx;
            }
        }

        MappedColor::Basic(best_idx)
    }

    /// Convert to 256-color mode
    fn to_256(&self, rgb: [u8; 3]) -> MappedColor {
        // 256-color mode: 16 basic + 216 color cube + 24 grayscale

        // Check if grayscale is a good match
        let gray_diff = (rgb[0] as i32 - rgb[1] as i32).abs()
            + (rgb[1] as i32 - rgb[2] as i32).abs()
            + (rgb[0] as i32 - rgb[2] as i32).abs();

        if gray_diff < 30 {
            // Use grayscale ramp (232-255)
            let avg = (rgb[0] as u16 + rgb[1] as u16 + rgb[2] as u16) / 3;
            let gray_idx = ((avg as f32 / 255.0 * 23.0) as u8).min(23);
            return MappedColor::Extended(232 + gray_idx);
        }

        // Use 6x6x6 color cube (16-231)
        let r = (rgb[0] as f32 / 255.0 * 5.0).round() as u8;
        let g = (rgb[1] as f32 / 255.0 * 5.0).round() as u8;
        let b = (rgb[2] as f32 / 255.0 * 5.0).round() as u8;

        let idx = 16 + 36 * r + 6 * g + b;
        MappedColor::Extended(idx)
    }

    /// Get ANSI escape sequence for foreground color
    pub fn fg_escape(&self, rgb: [u8; 3]) -> String {
        match self.map_color(rgb) {
            MappedColor::Basic(idx) => {
                if idx < 8 {
                    format!("\x1b[{}m", 30 + idx)
                } else {
                    format!("\x1b[{}m", 82 + idx) // 90-97 for bright colors
                }
            }
            MappedColor::Extended(idx) => format!("\x1b[38;5;{}m", idx),
            MappedColor::TrueColor([r, g, b]) => format!("\x1b[38;2;{};{};{}m", r, g, b),
        }
    }

    /// Get ANSI escape sequence for background color
    pub fn bg_escape(&self, rgb: [u8; 3]) -> String {
        match self.map_color(rgb) {
            MappedColor::Basic(idx) => {
                if idx < 8 {
                    format!("\x1b[{}m", 40 + idx)
                } else {
                    format!("\x1b[{}m", 92 + idx) // 100-107 for bright colors
                }
            }
            MappedColor::Extended(idx) => format!("\x1b[48;5;{}m", idx),
            MappedColor::TrueColor([r, g, b]) => format!("\x1b[48;2;{};{};{}m", r, g, b),
        }
    }

    /// Get color depth
    pub fn depth(&self) -> ColorDepth {
        self.depth
    }
}

impl Default for ColorMapper {
    fn default() -> Self {
        Self::new(ColorDepth::TrueColor)
    }
}

/// Mapped terminal color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappedColor {
    /// Basic 16-color (0-15)
    Basic(u8),
    /// Extended 256-color (0-255)
    Extended(u8),
    /// True color RGB
    TrueColor([u8; 3]),
}

/// Calculate color distance (squared Euclidean)
fn color_distance(a: [u8; 3], b: [u8; 3]) -> u32 {
    let dr = a[0] as i32 - b[0] as i32;
    let dg = a[1] as i32 - b[1] as i32;
    let db = a[2] as i32 - b[2] as i32;
    (dr * dr + dg * dg + db * db) as u32
}

/// Predefined color constants
pub mod colors {
    pub const BLACK: [u8; 3] = [0, 0, 0];
    pub const WHITE: [u8; 3] = [255, 255, 255];
    pub const RED: [u8; 3] = [255, 0, 0];
    pub const GREEN: [u8; 3] = [0, 255, 0];
    pub const BLUE: [u8; 3] = [0, 0, 255];
    pub const YELLOW: [u8; 3] = [255, 255, 0];
    pub const CYAN: [u8; 3] = [0, 255, 255];
    pub const MAGENTA: [u8; 3] = [255, 0, 255];

    pub const DARK_GRAY: [u8; 3] = [64, 64, 64];
    pub const LIGHT_GRAY: [u8; 3] = [192, 192, 192];

    pub const SHIP_GREEN: [u8; 3] = [64, 192, 128];
    pub const SHIP_ORANGE: [u8; 3] = [192, 128, 64];
    pub const DAMAGE_RED: [u8; 3] = [192, 64, 64];
    pub const SHIELD_BLUE: [u8; 3] = [64, 128, 192];
    pub const THRUST_ORANGE: [u8; 3] = [255, 128, 0];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_true_color_passthrough() {
        let mapper = ColorMapper::new(ColorDepth::TrueColor);
        let color = [100, 150, 200];

        match mapper.map_color(color) {
            MappedColor::TrueColor(c) => assert_eq!(c, color),
            _ => panic!("Expected true color"),
        }
    }

    #[test]
    fn test_monochrome() {
        let mapper = ColorMapper::new(ColorDepth::Monochrome);

        // Bright colors should map to white
        match mapper.map_color([200, 200, 200]) {
            MappedColor::Basic(15) => {}
            other => panic!("Expected white (15), got {:?}", other),
        }

        // Dark colors should map to black
        match mapper.map_color([50, 50, 50]) {
            MappedColor::Basic(0) => {}
            other => panic!("Expected black (0), got {:?}", other),
        }
    }

    #[test]
    fn test_basic_16_red() {
        let mapper = ColorMapper::new(ColorDepth::Basic);

        // Pure red should map to red
        match mapper.map_color([255, 0, 0]) {
            MappedColor::Basic(9) => {} // Bright red
            other => panic!("Expected bright red (9), got {:?}", other),
        }
    }

    #[test]
    fn test_256_grayscale() {
        let mapper = ColorMapper::new(ColorDepth::Extended);

        // Gray should use grayscale ramp
        match mapper.map_color([128, 128, 128]) {
            MappedColor::Extended(idx) => {
                assert!(idx >= 232, "Expected grayscale range (232-255)");
            }
            other => panic!("Expected extended color, got {:?}", other),
        }
    }

    #[test]
    fn test_escape_sequences() {
        let mapper = ColorMapper::new(ColorDepth::TrueColor);
        let escape = mapper.fg_escape([100, 150, 200]);
        assert!(escape.contains("38;2;100;150;200"));
    }

    #[test]
    fn test_palette_adjustment() {
        let mapper = ColorMapper::new(ColorDepth::TrueColor).with_palette(Palette::Warm);

        match mapper.map_color([100, 100, 100]) {
            MappedColor::TrueColor([r, _g, _b]) => {
                // Warm palette should boost red
                assert!(r > 100);
            }
            _ => panic!("Expected true color"),
        }
    }
}
