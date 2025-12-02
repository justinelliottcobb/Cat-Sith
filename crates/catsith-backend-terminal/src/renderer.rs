//! Terminal renderer implementation

use async_trait::async_trait;
use catsith_core::entity::SemanticEntity;
use catsith_core::scene::{Environment, SceneEvent, Viewport};
use catsith_core::{RenderOutput, TerminalCell, TerminalFrame};
use catsith_pipeline::LoraStack;
use catsith_pipeline::stage::{PipelineStage, RenderContext, StageError};
use rayon::prelude::*;

use crate::color::ColorMapper;
use crate::sprites::SpriteGenerator;

/// Terminal renderer pipeline stage
pub struct TerminalRenderer {
    /// Sprite generator
    sprite_gen: SpriteGenerator,
    /// Color mapper
    color_mapper: ColorMapper,
    /// Output dimensions
    width: u32,
    height: u32,
}

impl TerminalRenderer {
    /// Create a new terminal renderer
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            sprite_gen: SpriteGenerator::new(),
            color_mapper: ColorMapper::default(),
            width,
            height,
        }
    }

    /// Set the color mapper
    pub fn with_color_mapper(mut self, mapper: ColorMapper) -> Self {
        self.color_mapper = mapper;
        self
    }

    /// Render an entity to the frame
    fn render_entity(
        &mut self,
        entity: &SemanticEntity,
        frame: &mut TerminalFrame,
        viewport: &Viewport,
    ) {
        // Convert world position to screen position
        let screen_pos = self.world_to_screen(entity.position, viewport);

        // Check if on screen
        if screen_pos.0 >= self.width as i32 + 3 || screen_pos.0 < -3 {
            return;
        }
        if screen_pos.1 >= self.height as i32 + 3 || screen_pos.1 < -3 {
            return;
        }

        // Get sprite for this entity
        let sprite = self.sprite_gen.generate(entity);

        // Blit sprite to frame (centered on position)
        for dy in 0..3 {
            for dx in 0..3 {
                let x = screen_pos.0 + dx as i32 - 1;
                let y = screen_pos.1 + dy as i32 - 1;

                if x >= 0 && y >= 0 && (x as u32) < frame.width && (y as u32) < frame.height {
                    let cell = &sprite.cells[dy][dx];
                    // Only draw non-space characters
                    if cell.char != ' ' {
                        frame.set(x as u32, y as u32, *cell);
                    }
                }
            }
        }
    }

    /// Convert world coordinates to screen coordinates
    fn world_to_screen(&self, world_pos: [f64; 2], viewport: &Viewport) -> (i32, i32) {
        let rel_x = world_pos[0] - viewport.center[0];
        let rel_y = world_pos[1] - viewport.center[1];

        // Apply zoom
        let zoomed_x = rel_x * viewport.zoom;
        let zoomed_y = rel_y * viewport.zoom;

        // Convert to screen coordinates
        let screen_x =
            (self.width as f64 / 2.0) + (zoomed_x / viewport.extent[0]) * self.width as f64;
        let screen_y =
            (self.height as f64 / 2.0) + (zoomed_y / viewport.extent[1]) * self.height as f64;

        (screen_x.round() as i32, screen_y.round() as i32)
    }

    /// Render environment background
    fn render_environment(&self, env: &Environment, frame: &mut TerminalFrame) {
        let bg_color = env.background_color.unwrap_or([10, 10, 20]);

        // Fill with background color (sparse for performance)
        match env.ambiance {
            catsith_core::scene::Ambiance::Void => {
                // Sparse star field
                self.render_starfield(frame, bg_color, 0.02);
            }
            catsith_core::scene::Ambiance::Nebula => {
                // Dense colorful background
                self.render_nebula(frame, bg_color);
            }
            catsith_core::scene::Ambiance::Asteroid => {
                // Rocky debris hints
                self.render_starfield(frame, bg_color, 0.01);
            }
            _ => {
                // Default: sparse dots
                self.render_starfield(frame, bg_color, 0.01);
            }
        }
    }

    /// Render sparse starfield using parallel row processing
    fn render_starfield(&self, frame: &mut TerminalFrame, _bg_color: [u8; 3], density: f64) {
        let width = frame.width;

        // Process rows in parallel
        frame
            .cells
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let y = y as u32;
                for x in 0..width {
                    let hash = (x * 13 + y * 7 + x * y) % 1000;
                    if (hash as f64 / 1000.0) < density {
                        let brightness = (hash % 3) as u8;
                        let ch = match brightness {
                            0 => '.',
                            1 => '*',
                            _ => '+',
                        };
                        let intensity = 50 + (brightness * 40);
                        row[x as usize] =
                            TerminalCell::new(ch).with_fg([intensity, intensity, intensity]);
                    }
                }
            });
    }

    /// Render nebula background using parallel row processing
    fn render_nebula(&self, frame: &mut TerminalFrame, base_color: [u8; 3]) {
        let width = frame.width;

        // Process rows in parallel
        frame
            .cells
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let y = y as u32;
                for x in 0..width {
                    let hash = (x * 17 + y * 11 + x * y * 3) % 1000;
                    if (hash as f64 / 1000.0) < 0.15 {
                        // Vary color based on position
                        let r = base_color[0].saturating_add(((x % 30) as u8).saturating_mul(2));
                        let g = base_color[1].saturating_add(((y % 20) as u8).saturating_mul(2));
                        let b =
                            base_color[2].saturating_add((((x + y) % 25) as u8).saturating_mul(2));

                        let ch = match hash % 4 {
                            0 => '~',
                            1 => '≈',
                            2 => '░',
                            _ => '.',
                        };

                        row[x as usize] = TerminalCell::new(ch).with_fg([r, g, b]);
                    }
                }
            });
    }

    /// Render scene events (explosions, beams, etc.)
    fn render_event(&self, event: &SceneEvent, frame: &mut TerminalFrame, viewport: &Viewport) {
        match event {
            SceneEvent::Explosion {
                position,
                radius,
                intensity,
                age,
            } => {
                let screen = self.world_to_screen(*position, viewport);
                let screen_radius =
                    (*radius * viewport.zoom / viewport.extent[0] * self.width as f64) as i32;
                let fade = 1.0 - age;

                // Draw explosion circle
                let intensity_byte = (*intensity * fade * 255.0) as u8;
                let color = [intensity_byte, (intensity_byte / 2), 0];

                for dy in -screen_radius..=screen_radius {
                    for dx in -screen_radius..=screen_radius {
                        if dx * dx + dy * dy <= screen_radius * screen_radius {
                            let x = screen.0 + dx;
                            let y = screen.1 + dy;
                            if x >= 0
                                && y >= 0
                                && (x as u32) < frame.width
                                && (y as u32) < frame.height
                            {
                                let ch = if *age < 0.3 {
                                    '#'
                                } else if *age < 0.6 {
                                    '*'
                                } else {
                                    '.'
                                };
                                frame.set(x as u32, y as u32, TerminalCell::new(ch).with_fg(color));
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

                let beam_color = color.unwrap_or([
                    (intensity * 255.0) as u8,
                    (intensity * 100.0) as u8,
                    (intensity * 100.0) as u8,
                ]);

                // Simple line drawing
                self.draw_line(frame, start_screen, end_screen, beam_color);
            }

            SceneEvent::Particle {
                position,
                particle_type,
                ..
            } => {
                let screen = self.world_to_screen(*position, viewport);
                if screen.0 >= 0
                    && screen.1 >= 0
                    && (screen.0 as u32) < frame.width
                    && (screen.1 as u32) < frame.height
                {
                    let (ch, color) = match particle_type.as_str() {
                        "spark" => ('*', [255, 200, 50]),
                        "debris" => ('.', [150, 150, 150]),
                        "energy" => ('+', [100, 200, 255]),
                        _ => ('.', [200, 200, 200]),
                    };
                    frame.set(
                        screen.0 as u32,
                        screen.1 as u32,
                        TerminalCell::new(ch).with_fg(color),
                    );
                }
            }

            SceneEvent::Flash { intensity, color } => {
                // Flash affects entire screen - parallelize by row
                let flash_color = color.unwrap_or([255, 255, 255]);
                let alpha = *intensity as f32;
                let inv_alpha = 1.0 - alpha;

                frame.cells.par_iter_mut().for_each(|cell| {
                    // Blend toward flash color
                    cell.fg[0] =
                        (cell.fg[0] as f32 * inv_alpha + flash_color[0] as f32 * alpha) as u8;
                    cell.fg[1] =
                        (cell.fg[1] as f32 * inv_alpha + flash_color[1] as f32 * alpha) as u8;
                    cell.fg[2] =
                        (cell.fg[2] as f32 * inv_alpha + flash_color[2] as f32 * alpha) as u8;
                });
            }

            SceneEvent::Shake { .. } => {
                // Shake would be handled by output layer
            }
        }
    }

    /// Draw a line using Bresenham's algorithm
    fn draw_line(
        &self,
        frame: &mut TerminalFrame,
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
                let ch = if dx > dy.abs() { '-' } else { '|' };
                frame.set(x as u32, y as u32, TerminalCell::new(ch).with_fg(color));
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
impl PipelineStage for TerminalRenderer {
    async fn process(&mut self, mut context: RenderContext) -> Result<RenderContext, StageError> {
        let mut frame = TerminalFrame::new(self.width, self.height);

        // Render environment first (background)
        self.render_environment(&context.scene.environment, &mut frame);

        // Sort entities by Y position for proper layering (back to front)
        let mut entities: Vec<_> = context.scene.entities.iter().collect();
        entities.sort_by(|a, b| a.position[1].partial_cmp(&b.position[1]).unwrap());

        // Render entities
        for entity in entities {
            self.render_entity(entity, &mut frame, &context.scene.viewport);
        }

        // Render events (explosions, etc.) on top
        for event in &context.scene.events {
            self.render_event(event, &mut frame, &context.scene.viewport);
        }

        context.output = Some(RenderOutput::Terminal(frame));
        Ok(context)
    }

    fn on_lora_change(&mut self, _loras: &LoraStack) {
        // Terminal renderer doesn't use LoRAs
        // Could potentially use them to change sprite sets
    }

    fn name(&self) -> &'static str {
        "terminal_renderer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catsith_core::entity::{EntityType, ShipClass};
    use catsith_core::scene::{Environment, Scene, Viewport};
    use catsith_core::style::PlayerStyle;

    #[tokio::test]
    async fn test_terminal_renderer() {
        let mut renderer = TerminalRenderer::new(80, 24);

        let entity = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [0.0, 0.0],
        );

        let scene = Scene::new(1)
            .with_viewport(Viewport::new([0.0, 0.0], [80.0, 24.0]))
            .with_entity(entity)
            .with_environment(Environment::space());

        let context = RenderContext::new(scene, PlayerStyle::default());
        let result = renderer.process(context).await.unwrap();

        let output = result.output.unwrap();
        assert!(output.is_terminal());

        let frame = output.as_terminal().unwrap();
        assert_eq!(frame.width, 80);
        assert_eq!(frame.height, 24);

        // Ship should be rendered at center
        let center_cell = frame.get(40, 12).unwrap();
        assert_eq!(center_cell.char, '#');
    }

    #[test]
    fn test_world_to_screen() {
        let renderer = TerminalRenderer::new(80, 24);
        let viewport = Viewport::new([0.0, 0.0], [80.0, 24.0]);

        // Center should map to center
        let screen = renderer.world_to_screen([0.0, 0.0], &viewport);
        assert_eq!(screen, (40, 12));

        // Top-left corner
        let screen = renderer.world_to_screen([-40.0, -12.0], &viewport);
        assert_eq!(screen, (0, 0));
    }
}
