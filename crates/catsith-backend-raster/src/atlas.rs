//! Sprite atlas management
//!
//! Loads and manages sprite sheets for efficient rendering.

use std::collections::HashMap;
use thiserror::Error;

/// Atlas errors
#[derive(Debug, Error)]
pub enum AtlasError {
    #[error("Sprite not found: {0}")]
    SpriteNotFound(String),

    #[error("Atlas not loaded: {0}")]
    NotLoaded(String),

    #[error("Invalid atlas format: {0}")]
    InvalidFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A region within a sprite atlas
#[derive(Debug, Clone, Copy)]
pub struct SpriteRegion {
    /// X position in atlas (pixels)
    pub x: u32,
    /// Y position in atlas (pixels)
    pub y: u32,
    /// Width (pixels)
    pub width: u32,
    /// Height (pixels)
    pub height: u32,
    /// Anchor point X (0-1, relative to sprite)
    pub anchor_x: f32,
    /// Anchor point Y (0-1, relative to sprite)
    pub anchor_y: f32,
}

impl SpriteRegion {
    /// Create a new sprite region
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            anchor_x: 0.5,
            anchor_y: 0.5,
        }
    }

    /// Set the anchor point
    pub fn with_anchor(mut self, x: f32, y: f32) -> Self {
        self.anchor_x = x;
        self.anchor_y = y;
        self
    }

    /// Get UV coordinates (for texture mapping)
    pub fn uv_coords(&self, atlas_width: u32, atlas_height: u32) -> [[f32; 2]; 4] {
        let u0 = self.x as f32 / atlas_width as f32;
        let v0 = self.y as f32 / atlas_height as f32;
        let u1 = (self.x + self.width) as f32 / atlas_width as f32;
        let v1 = (self.y + self.height) as f32 / atlas_height as f32;

        [
            [u0, v0], // Top-left
            [u1, v0], // Top-right
            [u1, v1], // Bottom-right
            [u0, v1], // Bottom-left
        ]
    }
}

/// Sprite atlas containing multiple sprite regions
pub struct SpriteAtlas {
    /// Atlas name
    name: String,
    /// Atlas dimensions
    width: u32,
    height: u32,
    /// Pixel data (RGBA)
    pixels: Vec<u8>,
    /// Named sprite regions
    regions: HashMap<String, SpriteRegion>,
}

impl SpriteAtlas {
    /// Create a new empty atlas
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            name: name.into(),
            width,
            height,
            pixels: vec![0; (width * height * 4) as usize],
            regions: HashMap::new(),
        }
    }

    /// Load atlas from image file
    pub fn load(name: impl Into<String>, path: &str) -> Result<Self, AtlasError> {
        let img = image::open(path)
            .map_err(|e| AtlasError::InvalidFormat(e.to_string()))?
            .to_rgba8();

        let (width, height) = img.dimensions();

        Ok(Self {
            name: name.into(),
            width,
            height,
            pixels: img.into_raw(),
            regions: HashMap::new(),
        })
    }

    /// Create a test atlas with a simple pattern
    pub fn test_pattern(name: impl Into<String>, width: u32, height: u32) -> Self {
        let mut atlas = Self::new(name, width, height);

        // Fill with a checkered pattern
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                let checker = ((x / 8) + (y / 8)) % 2 == 0;

                if checker {
                    atlas.pixels[idx] = 200; // R
                    atlas.pixels[idx + 1] = 200; // G
                    atlas.pixels[idx + 2] = 200; // B
                    atlas.pixels[idx + 3] = 255; // A
                } else {
                    atlas.pixels[idx] = 100;
                    atlas.pixels[idx + 1] = 100;
                    atlas.pixels[idx + 2] = 100;
                    atlas.pixels[idx + 3] = 255;
                }
            }
        }

        atlas
    }

    /// Get atlas name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get atlas dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Define a sprite region
    pub fn define_region(&mut self, name: impl Into<String>, region: SpriteRegion) {
        self.regions.insert(name.into(), region);
    }

    /// Define a grid of sprites
    #[allow(clippy::too_many_arguments)]
    pub fn define_grid(
        &mut self,
        prefix: &str,
        start_x: u32,
        start_y: u32,
        cell_width: u32,
        cell_height: u32,
        cols: u32,
        rows: u32,
    ) {
        for row in 0..rows {
            for col in 0..cols {
                let name = format!("{}_{}", prefix, row * cols + col);
                let region = SpriteRegion::new(
                    start_x + col * cell_width,
                    start_y + row * cell_height,
                    cell_width,
                    cell_height,
                );
                self.define_region(name, region);
            }
        }
    }

    /// Get a sprite region by name
    pub fn get_region(&self, name: &str) -> Option<&SpriteRegion> {
        self.regions.get(name)
    }

    /// Get pixel at position
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y * self.width + x) * 4) as usize;
        Some([
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
            self.pixels[idx + 3],
        ])
    }

    /// Get pixels for a region
    pub fn get_region_pixels(&self, region: &SpriteRegion) -> Vec<u8> {
        let mut pixels = vec![0u8; (region.width * region.height * 4) as usize];

        for dy in 0..region.height {
            for dx in 0..region.width {
                let src_x = region.x + dx;
                let src_y = region.y + dy;

                if let Some(pixel) = self.get_pixel(src_x, src_y) {
                    let dst_idx = ((dy * region.width + dx) * 4) as usize;
                    pixels[dst_idx..dst_idx + 4].copy_from_slice(&pixel);
                }
            }
        }

        pixels
    }

    /// List all region names
    pub fn list_regions(&self) -> Vec<&str> {
        self.regions.keys().map(|s| s.as_str()).collect()
    }
}

/// Atlas manager for multiple sprite atlases
pub struct AtlasManager {
    atlases: HashMap<String, SpriteAtlas>,
}

impl AtlasManager {
    /// Create a new atlas manager
    pub fn new() -> Self {
        Self {
            atlases: HashMap::new(),
        }
    }

    /// Add an atlas
    pub fn add(&mut self, atlas: SpriteAtlas) {
        self.atlases.insert(atlas.name().to_string(), atlas);
    }

    /// Get an atlas by name
    pub fn get(&self, name: &str) -> Option<&SpriteAtlas> {
        self.atlases.get(name)
    }

    /// Remove an atlas
    pub fn remove(&mut self, name: &str) -> Option<SpriteAtlas> {
        self.atlases.remove(name)
    }

    /// List loaded atlases
    pub fn list(&self) -> Vec<&str> {
        self.atlases.keys().map(|s| s.as_str()).collect()
    }

    /// Total memory usage
    pub fn memory_usage(&self) -> usize {
        self.atlases.values().map(|a| a.pixels.len()).sum()
    }
}

impl Default for AtlasManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprite_region() {
        let region = SpriteRegion::new(0, 0, 32, 32).with_anchor(0.5, 1.0);
        assert_eq!(region.width, 32);
        assert_eq!(region.anchor_y, 1.0);
    }

    #[test]
    fn test_uv_coords() {
        let region = SpriteRegion::new(64, 64, 32, 32);
        let uvs = region.uv_coords(256, 256);

        assert_eq!(uvs[0], [0.25, 0.25]); // Top-left
        assert_eq!(uvs[2], [0.375, 0.375]); // Bottom-right
    }

    #[test]
    fn test_atlas_creation() {
        let mut atlas = SpriteAtlas::test_pattern("test", 256, 256);

        atlas.define_region("sprite1", SpriteRegion::new(0, 0, 32, 32));
        atlas.define_region("sprite2", SpriteRegion::new(32, 0, 32, 32));

        assert_eq!(atlas.list_regions().len(), 2);
        assert!(atlas.get_region("sprite1").is_some());
    }

    #[test]
    fn test_grid_definition() {
        let mut atlas = SpriteAtlas::new("test", 256, 256);
        atlas.define_grid("walk", 0, 0, 32, 32, 4, 2);

        // Should define 8 regions (4 cols * 2 rows)
        assert!(atlas.get_region("walk_0").is_some());
        assert!(atlas.get_region("walk_7").is_some());
        assert!(atlas.get_region("walk_8").is_none());
    }

    #[test]
    fn test_atlas_manager() {
        let mut manager = AtlasManager::new();

        manager.add(SpriteAtlas::test_pattern("ships", 256, 256));
        manager.add(SpriteAtlas::test_pattern("effects", 128, 128));

        assert_eq!(manager.list().len(), 2);
        assert!(manager.get("ships").is_some());
    }
}
