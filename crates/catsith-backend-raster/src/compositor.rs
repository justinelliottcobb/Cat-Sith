//! Layer composition
//!
//! Composites multiple render layers into a final image.

use catsith_core::ImageFrame;
use rayon::prelude::*;

/// Blend mode for layer composition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Normal alpha blending
    #[default]
    Normal,
    /// Additive blending (for glows, lights)
    Additive,
    /// Multiplicative blending (for shadows)
    Multiply,
    /// Screen blending (lightens)
    Screen,
}

/// A render layer
#[derive(Debug, Clone)]
pub struct RenderLayer {
    /// Layer name
    pub name: String,
    /// Layer frame
    pub frame: ImageFrame,
    /// Blend mode
    pub blend_mode: BlendMode,
    /// Layer opacity (0-1)
    pub opacity: f32,
    /// Z-order (higher = on top)
    pub z_order: i32,
    /// Offset from origin
    pub offset: (i32, i32),
}

impl RenderLayer {
    /// Create a new layer
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            name: name.into(),
            frame: ImageFrame::new_rgba8(width, height),
            blend_mode: BlendMode::Normal,
            opacity: 1.0,
            z_order: 0,
            offset: (0, 0),
        }
    }

    /// Set blend mode
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Set opacity
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set z-order
    pub fn with_z_order(mut self, z: i32) -> Self {
        self.z_order = z;
        self
    }

    /// Set offset
    pub fn with_offset(mut self, x: i32, y: i32) -> Self {
        self.offset = (x, y);
        self
    }
}

/// Layer compositor
pub struct LayerCompositor {
    layers: Vec<RenderLayer>,
    output_width: u32,
    output_height: u32,
}

impl LayerCompositor {
    /// Create a new compositor
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            layers: Vec::new(),
            output_width: width,
            output_height: height,
        }
    }

    /// Add a layer
    pub fn add_layer(&mut self, layer: RenderLayer) {
        self.layers.push(layer);
    }

    /// Remove a layer by name
    pub fn remove_layer(&mut self, name: &str) -> Option<RenderLayer> {
        if let Some(idx) = self.layers.iter().position(|l| l.name == name) {
            Some(self.layers.remove(idx))
        } else {
            None
        }
    }

    /// Get a layer by name
    pub fn get_layer(&self, name: &str) -> Option<&RenderLayer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Get a mutable layer by name
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut RenderLayer> {
        self.layers.iter_mut().find(|l| l.name == name)
    }

    /// Clear all layers
    pub fn clear(&mut self) {
        self.layers.clear();
    }

    /// Composite all layers into a single frame
    pub fn composite(&mut self) -> ImageFrame {
        // Sort layers by z-order
        self.layers.sort_by_key(|l| l.z_order);

        let mut output = ImageFrame::new_rgba8(self.output_width, self.output_height);

        for layer in &self.layers {
            self.blend_layer(&mut output, layer);
        }

        output
    }

    /// Blend a single layer onto the output using parallel row processing
    fn blend_layer(&self, output: &mut ImageFrame, layer: &RenderLayer) {
        if layer.opacity <= 0.0 {
            return;
        }

        let output_width = output.width;
        let output_height = output.height;
        let layer_width = layer.frame.width;
        let layer_height = layer.frame.height;
        let offset_x = layer.offset.0;
        let offset_y = layer.offset.1;
        let blend_mode = layer.blend_mode;
        let opacity = layer.opacity;

        // Calculate the effective row range we need to process
        let start_row = offset_y.max(0) as u32;
        let end_row = ((offset_y + layer_height as i32) as u32).min(output_height);

        if start_row >= end_row {
            return;
        }

        // Process output rows in parallel
        // Each row in the output is independent, so we can safely parallelize
        let row_stride = (output_width * 4) as usize;

        output
            .data
            .par_chunks_mut(row_stride)
            .enumerate()
            .filter(|(out_y, _)| {
                let out_y = *out_y as u32;
                out_y >= start_row && out_y < end_row
            })
            .for_each(|(out_y, row_data)| {
                let out_y = out_y as i32;
                let layer_y = out_y - offset_y;

                if layer_y < 0 || layer_y >= layer_height as i32 {
                    return;
                }

                let layer_y = layer_y as u32;

                // Calculate x range for this row
                let start_x = offset_x.max(0) as u32;
                let end_x = ((offset_x + layer_width as i32) as u32).min(output_width);

                for out_x in start_x..end_x {
                    let layer_x = (out_x as i32 - offset_x) as u32;

                    let src = layer
                        .frame
                        .get_pixel(layer_x, layer_y)
                        .unwrap_or([0, 0, 0, 0]);

                    let dst_idx = (out_x * 4) as usize;
                    let dst = [
                        row_data[dst_idx],
                        row_data[dst_idx + 1],
                        row_data[dst_idx + 2],
                        row_data[dst_idx + 3],
                    ];

                    let blended = Self::blend_pixels_static(src, dst, blend_mode, opacity);

                    row_data[dst_idx] = blended[0];
                    row_data[dst_idx + 1] = blended[1];
                    row_data[dst_idx + 2] = blended[2];
                    row_data[dst_idx + 3] = blended[3];
                }
            });
    }

    /// Blend two pixels (static version for parallel use)
    fn blend_pixels_static(
        src: [u8; 4],
        dst: [u8; 4],
        mode: BlendMode,
        opacity: f32,
    ) -> [u8; 4] {
        let src_a = (src[3] as f32 / 255.0) * opacity;

        if src_a <= 0.0 {
            return dst;
        }

        match mode {
            BlendMode::Normal => {
                let dst_a = dst[3] as f32 / 255.0;
                let out_a = src_a + dst_a * (1.0 - src_a);

                if out_a <= 0.0 {
                    return [0, 0, 0, 0];
                }

                let blend = |s: u8, d: u8| -> u8 {
                    let s = s as f32 / 255.0;
                    let d = d as f32 / 255.0;
                    let out = (s * src_a + d * dst_a * (1.0 - src_a)) / out_a;
                    (out * 255.0).clamp(0.0, 255.0) as u8
                };

                [
                    blend(src[0], dst[0]),
                    blend(src[1], dst[1]),
                    blend(src[2], dst[2]),
                    (out_a * 255.0) as u8,
                ]
            }

            BlendMode::Additive => [
                (dst[0] as u16 + (src[0] as f32 * src_a) as u16).min(255) as u8,
                (dst[1] as u16 + (src[1] as f32 * src_a) as u16).min(255) as u8,
                (dst[2] as u16 + (src[2] as f32 * src_a) as u16).min(255) as u8,
                dst[3].max((src_a * 255.0) as u8),
            ],

            BlendMode::Multiply => {
                let mul = |s: u8, d: u8| -> u8 {
                    let s = s as f32 / 255.0;
                    let d = d as f32 / 255.0;
                    let m = s * d;
                    let result = d * (1.0 - src_a) + m * src_a;
                    (result * 255.0) as u8
                };

                [
                    mul(src[0], dst[0]),
                    mul(src[1], dst[1]),
                    mul(src[2], dst[2]),
                    dst[3],
                ]
            }

            BlendMode::Screen => {
                let screen = |s: u8, d: u8| -> u8 {
                    let s = s as f32 / 255.0;
                    let d = d as f32 / 255.0;
                    let m = 1.0 - (1.0 - s) * (1.0 - d);
                    let result = d * (1.0 - src_a) + m * src_a;
                    (result * 255.0) as u8
                };

                [
                    screen(src[0], dst[0]),
                    screen(src[1], dst[1]),
                    screen(src[2], dst[2]),
                    dst[3],
                ]
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = RenderLayer::new("test", 100, 100)
            .with_blend_mode(BlendMode::Additive)
            .with_opacity(0.5)
            .with_z_order(10);

        assert_eq!(layer.name, "test");
        assert_eq!(layer.blend_mode, BlendMode::Additive);
        assert_eq!(layer.opacity, 0.5);
        assert_eq!(layer.z_order, 10);
    }

    #[test]
    fn test_compositor() {
        let mut compositor = LayerCompositor::new(100, 100);

        let mut bg = RenderLayer::new("background", 100, 100).with_z_order(0);
        bg.frame.fill([100, 100, 100, 255]);

        let mut fg = RenderLayer::new("foreground", 50, 50)
            .with_z_order(1)
            .with_offset(25, 25);
        fg.frame.fill([255, 0, 0, 255]);

        compositor.add_layer(bg);
        compositor.add_layer(fg);

        let output = compositor.composite();

        // Background pixel
        assert_eq!(output.get_pixel(0, 0), Some([100, 100, 100, 255]));

        // Foreground pixel (red, blended over gray)
        let fg_pixel = output.get_pixel(50, 50).unwrap();
        assert_eq!(fg_pixel[0], 255); // Red
    }

    #[test]
    fn test_additive_blend() {
        let dst = [100, 100, 100, 255];
        let src = [50, 50, 50, 255];

        let result = LayerCompositor::blend_pixels_static(src, dst, BlendMode::Additive, 1.0);

        // Additive should add values
        assert_eq!(result[0], 150);
    }
}
