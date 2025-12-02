//! Render output types
//!
//! These types represent the final rendered output from CatSith.

use crate::intent::ImageFormat;
use serde::{Deserialize, Serialize};

/// The result of rendering a scene
#[derive(Debug, Clone)]
pub enum RenderOutput {
    /// Terminal output - grid of cells
    Terminal(TerminalFrame),
    /// Image output - pixel buffer
    Image(ImageFrame),
    /// Tensor output (for pipeline chaining)
    Tensor(TensorFrame),
}

impl RenderOutput {
    /// Get the dimensions of the output
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Terminal(f) => (f.width, f.height),
            Self::Image(f) => (f.width, f.height),
            Self::Tensor(f) => {
                if f.shape.len() >= 2 {
                    (
                        f.shape[f.shape.len() - 1] as u32,
                        f.shape[f.shape.len() - 2] as u32,
                    )
                } else {
                    (0, 0)
                }
            }
        }
    }

    /// Check if output is terminal type
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal(_))
    }

    /// Check if output is image type
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image(_))
    }

    /// Get as terminal frame
    pub fn as_terminal(&self) -> Option<&TerminalFrame> {
        match self {
            Self::Terminal(f) => Some(f),
            _ => None,
        }
    }

    /// Get as image frame
    pub fn as_image(&self) -> Option<&ImageFrame> {
        match self {
            Self::Image(f) => Some(f),
            _ => None,
        }
    }
}

/// Terminal frame - 2D grid of characters with colors
#[derive(Debug, Clone)]
pub struct TerminalFrame {
    /// Width in cells
    pub width: u32,
    /// Height in cells
    pub height: u32,
    /// Cell data (row-major order)
    pub cells: Vec<TerminalCell>,
}

impl TerminalFrame {
    /// Create a new empty terminal frame
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            cells: vec![TerminalCell::default(); (width * height) as usize],
        }
    }

    /// Fill the frame with a single character
    pub fn fill(&mut self, cell: TerminalCell) {
        self.cells.fill(cell);
    }

    /// Get a cell at position
    pub fn get(&self, x: u32, y: u32) -> Option<&TerminalCell> {
        if x < self.width && y < self.height {
            Some(&self.cells[(y * self.width + x) as usize])
        } else {
            None
        }
    }

    /// Get a mutable cell at position
    pub fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut TerminalCell> {
        if x < self.width && y < self.height {
            Some(&mut self.cells[(y * self.width + x) as usize])
        } else {
            None
        }
    }

    /// Set a cell at position
    pub fn set(&mut self, x: u32, y: u32, cell: TerminalCell) {
        if x < self.width && y < self.height {
            self.cells[(y * self.width + x) as usize] = cell;
        }
    }

    /// Draw a string at position
    pub fn draw_str(&mut self, x: u32, y: u32, s: &str, fg: [u8; 3]) {
        for (i, ch) in s.chars().enumerate() {
            let px = x + i as u32;
            if px < self.width {
                self.set(px, y, TerminalCell::new(ch).with_fg(fg));
            }
        }
    }

    /// Draw a box/rectangle outline
    pub fn draw_box(&mut self, x: u32, y: u32, w: u32, h: u32, fg: [u8; 3]) {
        // Corners
        self.set(x, y, TerminalCell::new('┌').with_fg(fg));
        self.set(x + w - 1, y, TerminalCell::new('┐').with_fg(fg));
        self.set(x, y + h - 1, TerminalCell::new('└').with_fg(fg));
        self.set(x + w - 1, y + h - 1, TerminalCell::new('┘').with_fg(fg));

        // Horizontal lines
        for i in 1..w - 1 {
            self.set(x + i, y, TerminalCell::new('─').with_fg(fg));
            self.set(x + i, y + h - 1, TerminalCell::new('─').with_fg(fg));
        }

        // Vertical lines
        for i in 1..h - 1 {
            self.set(x, y + i, TerminalCell::new('│').with_fg(fg));
            self.set(x + w - 1, y + i, TerminalCell::new('│').with_fg(fg));
        }
    }

    /// Get iterator over all cells with positions
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32, &TerminalCell)> {
        self.cells.iter().enumerate().map(move |(i, cell)| {
            let x = i as u32 % self.width;
            let y = i as u32 / self.width;
            (x, y, cell)
        })
    }

    /// Convert to ANSI escape code string for terminal output
    pub fn to_ansi(&self) -> String {
        let mut output = String::new();
        let mut last_fg: Option<[u8; 3]> = None;
        let mut last_bg: Option<[u8; 3]> = None;

        for y in 0..self.height {
            for x in 0..self.width {
                let cell = self.get(x, y).unwrap();

                // Update foreground color if changed
                if last_fg != Some(cell.fg) {
                    output.push_str(&format!(
                        "\x1b[38;2;{};{};{}m",
                        cell.fg[0], cell.fg[1], cell.fg[2]
                    ));
                    last_fg = Some(cell.fg);
                }

                // Update background color if changed
                if last_bg != cell.bg {
                    if let Some(bg) = cell.bg {
                        output.push_str(&format!("\x1b[48;2;{};{};{}m", bg[0], bg[1], bg[2]));
                    } else if last_bg.is_some() {
                        output.push_str("\x1b[49m"); // Reset background
                    }
                    last_bg = cell.bg;
                }

                // Bold
                if cell.bold {
                    output.push_str("\x1b[1m");
                }

                output.push(cell.char);

                if cell.bold {
                    output.push_str("\x1b[22m"); // Reset bold
                }
            }
            output.push('\n');
        }

        output.push_str("\x1b[0m"); // Reset all
        output
    }
}

/// A single terminal cell
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalCell {
    /// The character to display
    pub char: char,
    /// Foreground color (RGB)
    pub fg: [u8; 3],
    /// Background color (RGB, None for transparent)
    pub bg: Option<[u8; 3]>,
    /// Bold text
    pub bold: bool,
}

impl Default for TerminalCell {
    fn default() -> Self {
        Self {
            char: ' ',
            fg: [200, 200, 200],
            bg: None,
            bold: false,
        }
    }
}

impl TerminalCell {
    /// Create a new cell with a character
    pub fn new(char: char) -> Self {
        Self {
            char,
            ..Default::default()
        }
    }

    /// Set foreground color
    pub fn with_fg(mut self, fg: [u8; 3]) -> Self {
        self.fg = fg;
        self
    }

    /// Set background color
    pub fn with_bg(mut self, bg: [u8; 3]) -> Self {
        self.bg = Some(bg);
        self
    }

    /// Set bold
    pub fn with_bold(mut self, bold: bool) -> Self {
        self.bold = bold;
        self
    }

    /// Create a space cell (empty)
    pub fn space() -> Self {
        Self::default()
    }

    /// Check if cell is empty (space with no background)
    pub fn is_empty(&self) -> bool {
        self.char == ' ' && self.bg.is_none()
    }
}

/// Image frame - pixel buffer
#[derive(Debug, Clone)]
pub struct ImageFrame {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Pixel format
    pub format: ImageFormat,
    /// Raw pixel data
    pub data: Vec<u8>,
}

impl ImageFrame {
    /// Create a new RGBA8 image frame
    pub fn new_rgba8(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: ImageFormat::Rgba8,
            data: vec![0; (width * height * 4) as usize],
        }
    }

    /// Create a new RGB8 image frame
    pub fn new_rgb8(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: ImageFormat::Rgb8,
            data: vec![0; (width * height * 3) as usize],
        }
    }

    /// Get bytes per pixel for this format
    pub fn bytes_per_pixel(&self) -> usize {
        match self.format {
            ImageFormat::Rgba8 => 4,
            ImageFormat::Rgba16 => 8,
            ImageFormat::Rgb8 => 3,
        }
    }

    /// Get pixel at position (returns RGBA)
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let bpp = self.bytes_per_pixel();
        let idx = ((y * self.width + x) as usize) * bpp;

        Some(match self.format {
            ImageFormat::Rgba8 => [
                self.data[idx],
                self.data[idx + 1],
                self.data[idx + 2],
                self.data[idx + 3],
            ],
            ImageFormat::Rgb8 => [self.data[idx], self.data[idx + 1], self.data[idx + 2], 255],
            ImageFormat::Rgba16 => {
                // Convert 16-bit to 8-bit
                [
                    (self.data[idx] as u16 | ((self.data[idx + 1] as u16) << 8) >> 8) as u8,
                    (self.data[idx + 2] as u16 | ((self.data[idx + 3] as u16) << 8) >> 8) as u8,
                    (self.data[idx + 4] as u16 | ((self.data[idx + 5] as u16) << 8) >> 8) as u8,
                    (self.data[idx + 6] as u16 | ((self.data[idx + 7] as u16) << 8) >> 8) as u8,
                ]
            }
        })
    }

    /// Set pixel at position (RGBA)
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }

        let bpp = self.bytes_per_pixel();
        let idx = ((y * self.width + x) as usize) * bpp;

        match self.format {
            ImageFormat::Rgba8 => {
                self.data[idx] = rgba[0];
                self.data[idx + 1] = rgba[1];
                self.data[idx + 2] = rgba[2];
                self.data[idx + 3] = rgba[3];
            }
            ImageFormat::Rgb8 => {
                self.data[idx] = rgba[0];
                self.data[idx + 1] = rgba[1];
                self.data[idx + 2] = rgba[2];
            }
            ImageFormat::Rgba16 => {
                // Convert 8-bit to 16-bit (duplicate byte)
                self.data[idx] = rgba[0];
                self.data[idx + 1] = rgba[0];
                self.data[idx + 2] = rgba[1];
                self.data[idx + 3] = rgba[1];
                self.data[idx + 4] = rgba[2];
                self.data[idx + 5] = rgba[2];
                self.data[idx + 6] = rgba[3];
                self.data[idx + 7] = rgba[3];
            }
        }
    }

    /// Fill entire image with a color
    pub fn fill(&mut self, rgba: [u8; 4]) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.set_pixel(x, y, rgba);
            }
        }
    }
}

/// Tensor frame for pipeline chaining
#[derive(Debug, Clone)]
pub struct TensorFrame {
    /// Tensor shape (e.g., [batch, channels, height, width])
    pub shape: Vec<usize>,
    /// Flattened tensor data
    pub data: Vec<f32>,
}

impl TensorFrame {
    /// Create a new tensor frame
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    /// Create from existing data
    pub fn from_data(shape: Vec<usize>, data: Vec<f32>) -> Option<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return None;
        }
        Some(Self { shape, data })
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_frame() {
        let mut frame = TerminalFrame::new(10, 5);
        assert_eq!(frame.cells.len(), 50);

        frame.set(5, 2, TerminalCell::new('X').with_fg([255, 0, 0]));
        let cell = frame.get(5, 2).unwrap();
        assert_eq!(cell.char, 'X');
        assert_eq!(cell.fg, [255, 0, 0]);
    }

    #[test]
    fn test_terminal_frame_bounds() {
        let frame = TerminalFrame::new(10, 5);
        assert!(frame.get(0, 0).is_some());
        assert!(frame.get(9, 4).is_some());
        assert!(frame.get(10, 0).is_none());
        assert!(frame.get(0, 5).is_none());
    }

    #[test]
    fn test_image_frame() {
        let mut frame = ImageFrame::new_rgba8(100, 100);
        frame.set_pixel(50, 50, [255, 0, 0, 255]);

        let pixel = frame.get_pixel(50, 50).unwrap();
        assert_eq!(pixel, [255, 0, 0, 255]);
    }

    #[test]
    fn test_tensor_frame() {
        let tensor = TensorFrame::new(vec![1, 3, 64, 64]);
        assert_eq!(tensor.len(), 3 * 64 * 64);
    }

    #[test]
    fn test_terminal_cell_default() {
        let cell = TerminalCell::default();
        assert_eq!(cell.char, ' ');
        assert!(cell.is_empty());
    }
}
