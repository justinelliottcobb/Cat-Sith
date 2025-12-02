//! Neural renderer pipeline stage

use async_trait::async_trait;
use catsith_core::{ImageFrame, RenderOutput};
use catsith_pipeline::lora::LoraStack;
use catsith_pipeline::stage::{PipelineStage, RenderContext, StageError};

use crate::embedder::TextEmbedder;
use crate::inference::InferenceEngine;
use crate::models::SpriteVAE;
use crate::temporal::TemporalCoherence;

/// Neural renderer configuration
#[derive(Debug, Clone)]
pub struct NeuralRendererConfig {
    /// Output width
    pub width: u32,
    /// Output height
    pub height: u32,
    /// Sprite size
    pub sprite_size: u32,
    /// Enable temporal coherence
    pub temporal_coherence: bool,
    /// Temporal blend factor
    pub blend_factor: f32,
}

impl Default for NeuralRendererConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            sprite_size: 32,
            temporal_coherence: true,
            blend_factor: 0.3,
        }
    }
}

/// Neural renderer pipeline stage
#[allow(dead_code)]
pub struct NeuralRenderer {
    config: NeuralRendererConfig,
    engine: InferenceEngine,
    embedder: TextEmbedder,
    sprite_vae: SpriteVAE,
    temporal: TemporalCoherence,
    initialized: bool,
}

impl NeuralRenderer {
    /// Create a new neural renderer
    pub fn new(config: NeuralRendererConfig) -> Self {
        Self {
            temporal: TemporalCoherence::new(config.blend_factor),
            config,
            engine: InferenceEngine::new(),
            embedder: TextEmbedder::default(),
            sprite_vae: SpriteVAE::default(),
            initialized: false,
        }
    }

    /// Initialize/load models
    pub fn initialize(&mut self) -> Result<(), StageError> {
        // In a real implementation, this would load actual models
        // For now, we just mark as initialized
        self.initialized = true;
        Ok(())
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get configuration
    pub fn config(&self) -> &NeuralRendererConfig {
        &self.config
    }

    /// Render a scene to an image
    fn render_scene(&mut self, context: &RenderContext) -> Result<ImageFrame, StageError> {
        let mut frame = ImageFrame::new_rgba8(self.config.width, self.config.height);

        // Fill with background color
        let bg_color = context
            .scene
            .environment
            .background_color
            .unwrap_or([20, 20, 40]);
        frame.fill([bg_color[0], bg_color[1], bg_color[2], 255]);

        // Process each entity
        for entity in &context.scene.entities {
            // Get semantic description
            let description = format!("{:?} {:?}", entity.entity_type, entity.archetype);

            // Generate embedding
            let embedding = self
                .embedder
                .embed(&description)
                .map_err(|e| StageError::ProcessingFailed(e.to_string()))?;

            // Apply temporal smoothing
            let smoothed = if self.config.temporal_coherence {
                self.temporal.smooth_embedding(entity.id, &embedding)
            } else {
                embedding
            };

            // Generate sprite
            let sprite = self.sprite_vae.generate_test(&smoothed);

            // Calculate screen position
            let screen_pos = self.world_to_screen(entity.position, &context.scene.viewport);

            // Blit sprite to frame
            self.blit_sprite(&mut frame, &sprite, screen_pos);
        }

        Ok(frame)
    }

    /// Convert world to screen coordinates
    fn world_to_screen(
        &self,
        world_pos: [f64; 2],
        viewport: &catsith_core::scene::Viewport,
    ) -> (i32, i32) {
        let rel_x = world_pos[0] - viewport.center[0];
        let rel_y = world_pos[1] - viewport.center[1];

        let screen_x = (self.config.width as f64 / 2.0)
            + (rel_x / viewport.extent[0]) * self.config.width as f64;
        let screen_y = (self.config.height as f64 / 2.0)
            + (rel_y / viewport.extent[1]) * self.config.height as f64;

        (screen_x.round() as i32, screen_y.round() as i32)
    }

    /// Blit sprite to frame with alpha blending
    fn blit_sprite(
        &self,
        frame: &mut ImageFrame,
        sprite: &crate::models::sprite_vae::SpriteOutput,
        pos: (i32, i32),
    ) {
        let half_w = (sprite.width / 2) as i32;
        let half_h = (sprite.height / 2) as i32;

        for sy in 0..sprite.height {
            for sx in 0..sprite.width {
                let fx = pos.0 + sx as i32 - half_w;
                let fy = pos.1 + sy as i32 - half_h;

                if fx >= 0 && fy >= 0 && (fx as u32) < frame.width && (fy as u32) < frame.height {
                    if let Some(pixel) = sprite.get_pixel(sx, sy) {
                        if pixel[3] > 0 {
                            // Simple alpha blend
                            let alpha = pixel[3] as f32 / 255.0;

                            if let Some(dst) = frame.get_pixel(fx as u32, fy as u32) {
                                let blended = [
                                    (pixel[0] as f32 * alpha + dst[0] as f32 * (1.0 - alpha)) as u8,
                                    (pixel[1] as f32 * alpha + dst[1] as f32 * (1.0 - alpha)) as u8,
                                    (pixel[2] as f32 * alpha + dst[2] as f32 * (1.0 - alpha)) as u8,
                                    255,
                                ];
                                frame.set_pixel(fx as u32, fy as u32, blended);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[async_trait]
impl PipelineStage for NeuralRenderer {
    async fn process(&mut self, mut context: RenderContext) -> Result<RenderContext, StageError> {
        if !self.initialized {
            self.initialize()?;
        }

        self.temporal.begin_frame();

        let frame = self.render_scene(&context)?;
        context.output = Some(RenderOutput::Image(frame));

        Ok(context)
    }

    fn on_lora_change(&mut self, _loras: &LoraStack) {
        // TODO: Apply LoRAs to model weights
    }

    fn name(&self) -> &'static str {
        "neural_renderer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catsith_core::entity::{EntityType, SemanticEntity, ShipClass};
    use catsith_core::scene::{Environment, Scene, Viewport};
    use catsith_core::style::PlayerStyle;

    #[tokio::test]
    async fn test_neural_renderer() {
        let mut renderer = NeuralRenderer::new(NeuralRendererConfig {
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
