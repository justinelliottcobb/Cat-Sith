//! Raster renderer pipeline stage

use async_trait::async_trait;
use catsith_core::entity::{EntityType, EnvironmentType, SemanticEntity, ShipClass};
use catsith_core::scene::Viewport;
use catsith_core::{ImageFrame, RenderOutput};
use catsith_pipeline::lora::LoraStack;
use catsith_pipeline::stage::{PipelineStage, RenderContext, StageError};

use crate::atlas::{AtlasManager, SpriteAtlas, SpriteRegion};
use crate::compositor::{BlendMode, LayerCompositor, RenderLayer};

/// Raster renderer configuration
#[derive(Debug, Clone)]
pub struct RasterRendererConfig {
    /// Output width
    pub width: u32,
    /// Output height
    pub height: u32,
    /// Default sprite size
    pub sprite_size: u32,
}

impl Default for RasterRendererConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            sprite_size: 32,
        }
    }
}

/// Traditional 2D raster renderer
pub struct RasterRenderer {
    config: RasterRendererConfig,
    atlas_manager: AtlasManager,
    compositor: LayerCompositor,
}

impl RasterRenderer {
    /// Create a new raster renderer
    pub fn new(config: RasterRendererConfig) -> Self {
        let compositor = LayerCompositor::new(config.width, config.height);
        let mut atlas_manager = AtlasManager::new();

        // Create default test atlas
        let mut default_atlas = SpriteAtlas::test_pattern("default", 256, 256);
        Self::define_default_sprites(&mut default_atlas);
        atlas_manager.add(default_atlas);

        Self {
            config,
            atlas_manager,
            compositor,
        }
    }

    /// Define default sprite regions
    fn define_default_sprites(atlas: &mut SpriteAtlas) {
        // Ship sprites (8 directions x 3 classes = 24 sprites)
        atlas.define_grid("fighter", 0, 0, 32, 32, 8, 1);
        atlas.define_grid("bomber", 0, 32, 32, 32, 8, 1);
        atlas.define_grid("scout", 0, 64, 32, 32, 8, 1);

        // Environment sprites
        atlas.define_region("asteroid", SpriteRegion::new(0, 128, 32, 32));
        atlas.define_region("debris", SpriteRegion::new(32, 128, 32, 32));
        atlas.define_region("station", SpriteRegion::new(64, 128, 64, 64));

        // Projectiles
        atlas.define_region(
            "bullet",
            SpriteRegion::new(0, 192, 8, 8).with_anchor(0.5, 0.5),
        );
        atlas.define_region(
            "missile",
            SpriteRegion::new(8, 192, 16, 8).with_anchor(0.5, 0.5),
        );
    }

    /// Get configuration
    pub fn config(&self) -> &RasterRendererConfig {
        &self.config
    }

    /// Load a sprite atlas
    pub fn load_atlas(&mut self, atlas: SpriteAtlas) {
        self.atlas_manager.add(atlas);
    }

    /// Render a scene
    fn render_scene(&mut self, context: &RenderContext) -> Result<ImageFrame, StageError> {
        self.compositor.clear();

        // Create background layer
        let mut bg_layer = RenderLayer::new("background", self.config.width, self.config.height)
            .with_z_order(-100);

        let bg_color = context
            .scene
            .environment
            .background_color
            .unwrap_or([20, 20, 40]);
        bg_layer
            .frame
            .fill([bg_color[0], bg_color[1], bg_color[2], 255]);

        self.compositor.add_layer(bg_layer);

        // Create entity layer
        let mut entity_layer =
            RenderLayer::new("entities", self.config.width, self.config.height).with_z_order(0);

        // Sort entities by Y for proper layering
        let mut entities: Vec<_> = context.scene.entities.iter().collect();
        entities.sort_by(|a, b| a.position[1].partial_cmp(&b.position[1]).unwrap());

        // Render each entity
        for entity in entities {
            self.render_entity(entity, &mut entity_layer.frame, &context.scene.viewport);
        }

        self.compositor.add_layer(entity_layer);

        // Create effects layer
        let mut effects_layer = RenderLayer::new("effects", self.config.width, self.config.height)
            .with_z_order(100)
            .with_blend_mode(BlendMode::Additive);

        // Render events (explosions, beams)
        for event in &context.scene.events {
            self.render_event(event, &mut effects_layer.frame, &context.scene.viewport);
        }

        self.compositor.add_layer(effects_layer);

        // Composite all layers
        Ok(self.compositor.composite())
    }

    /// Render a single entity
    fn render_entity(&self, entity: &SemanticEntity, frame: &mut ImageFrame, viewport: &Viewport) {
        let screen_pos = self.world_to_screen(entity.position, viewport);

        // Get sprite name for this entity
        let sprite_name = self.entity_sprite_name(entity);

        // Get sprite from atlas
        if let Some(atlas) = self.atlas_manager.get("default") {
            if let Some(region) = atlas.get_region(&sprite_name) {
                self.blit_sprite(frame, atlas, region, screen_pos);
            } else {
                // Draw placeholder
                self.draw_placeholder(frame, screen_pos, self.config.sprite_size);
            }
        }
    }

    /// Get sprite name for an entity
    fn entity_sprite_name(&self, entity: &SemanticEntity) -> String {
        match &entity.entity_type {
            EntityType::Ship { class, .. } => {
                let class_name = match class {
                    ShipClass::Fighter => "fighter",
                    ShipClass::Bomber => "bomber",
                    ShipClass::Scout => "scout",
                    _ => "fighter",
                };
                format!("{}_{}", class_name, entity.direction_index())
            }
            EntityType::Projectile { .. } => "bullet".to_string(),
            EntityType::Environment { object_type } => match object_type {
                EnvironmentType::Asteroid => "asteroid",
                EnvironmentType::Debris => "debris",
                EnvironmentType::Station => "station",
                _ => "asteroid",
            }
            .to_string(),
            _ => "asteroid".to_string(),
        }
    }

    /// Convert world to screen coordinates
    fn world_to_screen(&self, world_pos: [f64; 2], viewport: &Viewport) -> (i32, i32) {
        let rel_x = world_pos[0] - viewport.center[0];
        let rel_y = world_pos[1] - viewport.center[1];

        let screen_x = (self.config.width as f64 / 2.0)
            + (rel_x / viewport.extent[0]) * self.config.width as f64;
        let screen_y = (self.config.height as f64 / 2.0)
            + (rel_y / viewport.extent[1]) * self.config.height as f64;

        (screen_x.round() as i32, screen_y.round() as i32)
    }

    /// Blit a sprite from atlas to frame
    fn blit_sprite(
        &self,
        frame: &mut ImageFrame,
        atlas: &SpriteAtlas,
        region: &SpriteRegion,
        pos: (i32, i32),
    ) {
        let anchor_x = (region.width as f32 * region.anchor_x) as i32;
        let anchor_y = (region.height as f32 * region.anchor_y) as i32;

        for dy in 0..region.height {
            for dx in 0..region.width {
                let fx = pos.0 + dx as i32 - anchor_x;
                let fy = pos.1 + dy as i32 - anchor_y;

                if fx < 0 || fy < 0 {
                    continue;
                }

                let fx = fx as u32;
                let fy = fy as u32;

                if fx >= frame.width || fy >= frame.height {
                    continue;
                }

                if let Some(pixel) = atlas.get_pixel(region.x + dx, region.y + dy) {
                    if pixel[3] > 0 {
                        // Simple alpha blend
                        if let Some(dst) = frame.get_pixel(fx, fy) {
                            let alpha = pixel[3] as f32 / 255.0;
                            let blended = [
                                (pixel[0] as f32 * alpha + dst[0] as f32 * (1.0 - alpha)) as u8,
                                (pixel[1] as f32 * alpha + dst[1] as f32 * (1.0 - alpha)) as u8,
                                (pixel[2] as f32 * alpha + dst[2] as f32 * (1.0 - alpha)) as u8,
                                255,
                            ];
                            frame.set_pixel(fx, fy, blended);
                        }
                    }
                }
            }
        }
    }

    /// Draw a placeholder sprite
    fn draw_placeholder(&self, frame: &mut ImageFrame, pos: (i32, i32), size: u32) {
        let half = (size / 2) as i32;

        for dy in 0..size {
            for dx in 0..size {
                let fx = pos.0 + dx as i32 - half;
                let fy = pos.1 + dy as i32 - half;

                if fx >= 0 && fy >= 0 && (fx as u32) < frame.width && (fy as u32) < frame.height {
                    // Draw a simple X pattern
                    let on_x = (dx as i32 - half).abs() == (dy as i32 - half).abs();
                    if on_x {
                        frame.set_pixel(fx as u32, fy as u32, [255, 0, 255, 255]);
                    }
                }
            }
        }
    }

    /// Render scene events
    fn render_event(
        &self,
        event: &catsith_core::scene::SceneEvent,
        frame: &mut ImageFrame,
        viewport: &Viewport,
    ) {
        use catsith_core::scene::SceneEvent;

        match event {
            SceneEvent::Explosion {
                position,
                radius,
                intensity,
                age,
            } => {
                let screen = self.world_to_screen(*position, viewport);
                let screen_radius =
                    (*radius / viewport.extent[0] * self.config.width as f64) as i32;
                let fade = (1.0 - age) as f32;
                let intensity = (*intensity as f32 * fade * 255.0) as u8;

                // Draw circular explosion
                for dy in -screen_radius..=screen_radius {
                    for dx in -screen_radius..=screen_radius {
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq <= screen_radius * screen_radius {
                            let fx = screen.0 + dx;
                            let fy = screen.1 + dy;

                            if fx >= 0
                                && fy >= 0
                                && (fx as u32) < frame.width
                                && (fy as u32) < frame.height
                            {
                                let dist_factor =
                                    1.0 - (dist_sq as f32 / (screen_radius * screen_radius) as f32);
                                let r = (intensity as f32 * dist_factor) as u8;
                                let g = (intensity as f32 * dist_factor * 0.5) as u8;

                                frame.set_pixel(fx as u32, fy as u32, [r, g, 0, 255]);
                            }
                        }
                    }
                }
            }
            SceneEvent::Beam {
                start,
                end,
                intensity,
                color,
            } => {
                let start_screen = self.world_to_screen(*start, viewport);
                let end_screen = self.world_to_screen(*end, viewport);
                let beam_intensity = (*intensity * 255.0) as u8;
                let beam_color =
                    color.unwrap_or([beam_intensity, beam_intensity / 2, beam_intensity / 4]);

                self.draw_line(frame, start_screen, end_screen, beam_color);
            }
            _ => {}
        }
    }

    /// Draw a line using Bresenham's algorithm
    fn draw_line(
        &self,
        frame: &mut ImageFrame,
        start: (i32, i32),
        end: (i32, i32),
        color: [u8; 3],
    ) {
        let dx = (end.0 - start.0).abs();
        let dy = -(end.1 - start.1).abs();
        let sx = if start.0 < end.0 { 1 } else { -1 };
        let sy = if start.1 < end.1 { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = start.0;
        let mut y = start.1;

        loop {
            if x >= 0 && y >= 0 && (x as u32) < frame.width && (y as u32) < frame.height {
                frame.set_pixel(x as u32, y as u32, [color[0], color[1], color[2], 255]);
            }

            if x == end.0 && y == end.1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }
}

#[async_trait]
impl PipelineStage for RasterRenderer {
    async fn process(&mut self, mut context: RenderContext) -> Result<RenderContext, StageError> {
        let frame = self.render_scene(&context)?;
        context.output = Some(RenderOutput::Image(frame));
        Ok(context)
    }

    fn on_lora_change(&mut self, _loras: &LoraStack) {
        // Raster renderer could swap sprite atlases based on LoRAs
    }

    fn name(&self) -> &'static str {
        "raster_renderer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catsith_core::entity::SemanticEntity;
    use catsith_core::scene::{Environment, Scene, Viewport};
    use catsith_core::style::PlayerStyle;

    #[tokio::test]
    async fn test_raster_renderer() {
        let mut renderer = RasterRenderer::new(RasterRendererConfig {
            width: 320,
            height: 240,
            ..Default::default()
        });

        let entity = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [0.0, 0.0],
        );

        let scene = Scene::new(1)
            .with_viewport(Viewport::new([0.0, 0.0], [100.0, 100.0]))
            .with_entity(entity)
            .with_environment(Environment::space());

        let context = RenderContext::new(scene, PlayerStyle::default());
        let result = renderer.process(context).await.unwrap();

        let output = result.output.unwrap();
        assert!(output.is_image());

        let frame = output.as_image().unwrap();
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
    }
}
