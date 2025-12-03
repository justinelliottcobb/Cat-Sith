//! Terminal Scene Example
//!
//! Renders a single scene frame to stdout using the terminal renderer.
//! This demonstrates the basic CatSith rendering pipeline.
//!
//! Run with: cargo run --example terminal_scene

use catsith_backend_terminal::TerminalRenderer;
use catsith_core::entity::EntityFlags;
use catsith_core::scene::{Scene, Viewport};
use catsith_core::style::PlayerStyle;
use catsith_core::RenderOutput;
use catsith_domain_exospace::{ExoSpaceEntity, ExoSpaceEnvironment, ExoSpaceEvent};
use catsith_pipeline::stage::RenderContext;
use catsith_pipeline::PipelineStage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Scene dimensions
    let width = 60;
    let height = 20;
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    // Create player ship (fighter) at center, facing up
    let player = ExoSpaceEntity::fighter([center_x, center_y])
        .with_rotation(0.0) // Facing up
        .with_flags(EntityFlags::THRUSTING)
        .build();

    // Create an enemy bomber orbiting
    let enemy = ExoSpaceEntity::bomber([center_x + 15.0, center_y - 5.0])
        .with_rotation(std::f64::consts::PI) // Facing down
        .with_health(0.6)
        .with_flags(EntityFlags::DAMAGED)
        .build();

    // Create a scout ship
    let scout = ExoSpaceEntity::scout([center_x - 12.0, center_y + 4.0])
        .with_rotation(std::f64::consts::FRAC_PI_2) // Facing right
        .build();

    // Create some asteroids
    let asteroid1 = ExoSpaceEntity::asteroid([10.0, 5.0]).build();
    let asteroid2 = ExoSpaceEntity::asteroid([50.0, 15.0]).build();
    let asteroid3 = ExoSpaceEntity::asteroid([25.0, 3.0]).build();

    // Create a bullet
    let bullet = ExoSpaceEntity::bullet([center_x, center_y - 5.0])
        .with_velocity([0.0, -10.0])
        .build();

    // Create debris
    let debris = ExoSpaceEntity::debris([45.0, 10.0]).build();

    // Build the scene
    let scene = Scene::new(1)
        .with_timestamp(0.0)
        .with_viewport(Viewport::new([center_x, center_y], [width as f64, height as f64]))
        .with_entities(vec![
            player, enemy, scout, asteroid1, asteroid2, asteroid3, bullet, debris,
        ])
        .with_event(
            ExoSpaceEvent::explosion([center_x + 20.0, center_y], 4.0)
                .with_intensity(0.8)
                .build(),
        )
        .with_environment(ExoSpaceEnvironment::space());

    // Create terminal renderer
    let mut renderer = TerminalRenderer::new(width, height);

    // Create render context with terminal style
    let context = RenderContext::new(scene, PlayerStyle::terminal());

    // Render the scene
    let result = renderer.process(context).await?;

    // Output the frame
    if let Some(RenderOutput::Terminal(frame)) = result.output {
        // Print frame title
        println!("\n\x1b[1;36m=== CatSith Terminal Renderer Demo ===\x1b[0m\n");

        // Print the rendered frame with a border
        println!("\x1b[90m┌{}┐\x1b[0m", "─".repeat(width as usize));

        for line in frame.to_ansi().lines() {
            println!("\x1b[90m│\x1b[0m{}\x1b[90m│\x1b[0m", line);
        }

        println!("\x1b[90m└{}┘\x1b[0m", "─".repeat(width as usize));

        // Print legend
        println!("\n\x1b[1mLegend:\x1b[0m");
        println!(
            "  \x1b[38;2;64;192;128m#\x1b[0m Fighter (green)    \x1b[38;2;192;128;64m#\x1b[0m Bomber (orange)    \x1b[38;2;64;128;192m#\x1b[0m Scout (blue)"
        );
        println!(
            "  \x1b[38;2;128;96;64m@\x1b[0m Asteroid           \x1b[38;2;255;255;0m*\x1b[0m Bullet             \x1b[38;2;96;96;96m*\x1b[0m Debris"
        );
        println!("  \x1b[38;2;255;128;0m*\x1b[0m Thrust effect      \x1b[38;2;255;128;64m○\x1b[0m Explosion");
        println!();
    }

    Ok(())
}
