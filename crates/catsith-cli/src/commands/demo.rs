//! Demo command

use crate::DemoType;
use catsith_backend_terminal::{TerminalOutput, TerminalRenderer};
use catsith_core::RenderOutput;
use catsith_core::entity::{EntityFlags, EntityState, EntityType, SemanticEntity, ShipClass};
use catsith_core::scene::{Environment, Scene, SceneEvent, Viewport};
use catsith_core::style::PlayerStyle;
use catsith_pipeline::PipelineStage;
use catsith_pipeline::stage::RenderContext;
use std::time::{Duration, Instant};
use tracing::info;

pub async fn run(
    demo_type: DemoType,
    width: u32,
    height: u32,
    frames: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    match demo_type {
        DemoType::Terminal => run_terminal_demo(width, height, frames).await,
        DemoType::Raster => run_raster_demo(width, height, frames).await,
        DemoType::Showcase => run_showcase_demo(width, height, frames).await,
    }
}

async fn run_terminal_demo(
    width: u32,
    height: u32,
    frames: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "Running terminal demo ({}x{}, {} frames)",
        width, height, frames
    );

    let mut renderer = TerminalRenderer::new(width, height);
    let mut output = TerminalOutput::new();

    output.init()?;

    let frame_time = Duration::from_millis(33); // ~30 FPS
    let mut frame_count = 0u64;
    let start = Instant::now();

    for i in 0..frames {
        let frame_start = Instant::now();

        // Create animated scene
        let scene = create_demo_scene(i as f64 / 60.0, width as f64, height as f64);
        let context = RenderContext::new(scene, PlayerStyle::terminal());

        // Render
        let result = renderer.process(context).await?;

        // Display
        if let Some(RenderOutput::Terminal(frame)) = result.output {
            output.render(&frame)?;
        }

        frame_count += 1;

        // Frame timing
        let elapsed = frame_start.elapsed();
        if elapsed < frame_time {
            tokio::time::sleep(frame_time - elapsed).await;
        }
    }

    output.cleanup()?;

    let total_time = start.elapsed();
    let fps = frame_count as f64 / total_time.as_secs_f64();

    println!("\nDemo complete!");
    println!("Frames: {}", frame_count);
    println!("Time: {:.2}s", total_time.as_secs_f64());
    println!("Average FPS: {:.1}", fps);

    Ok(())
}

async fn run_raster_demo(
    width: u32,
    height: u32,
    frames: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "Running raster demo ({}x{}, {} frames)",
        width, height, frames
    );

    // For raster demo, we'll just create scenes and report timing
    println!(
        "Raster demo would render {} frames at {}x{}",
        frames, width, height
    );
    println!("(Image output not implemented in CLI yet)");

    Ok(())
}

async fn run_showcase_demo(
    width: u32,
    height: u32,
    _frames: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Running showcase demo");

    let mut renderer = TerminalRenderer::new(width, height);

    // Show different entity types
    let entities = vec![
        ("Fighter", ShipClass::Fighter),
        ("Bomber", ShipClass::Bomber),
        ("Scout", ShipClass::Scout),
    ];

    for (name, class) in entities {
        println!("\n{} ship (8 directions):", name);
        println!();

        for dir in 0..8 {
            let angle = dir as f64 * std::f64::consts::PI / 4.0;

            let entity = SemanticEntity::new(
                EntityType::Ship {
                    class,
                    owner_id: None,
                },
                [width as f64 / 2.0, height as f64 / 2.0],
            )
            .with_rotation(angle)
            .with_state(EntityState::full().with_flags(EntityFlags::THRUSTING));

            let scene = Scene::new(dir as u64)
                .with_viewport(Viewport::new(
                    [width as f64 / 2.0, height as f64 / 2.0],
                    [width as f64, height as f64],
                ))
                .with_entity(entity)
                .with_environment(Environment::space());

            let context = RenderContext::new(scene, PlayerStyle::terminal());
            let result = renderer.process(context).await?;

            if let Some(RenderOutput::Terminal(frame)) = result.output {
                // Print just the sprite area (center 5x5)
                let cx = width / 2;
                let cy = height / 2;

                for y in cy.saturating_sub(2)..=(cy + 2).min(height - 1) {
                    for x in cx.saturating_sub(2)..=(cx + 2).min(width - 1) {
                        if let Some(cell) = frame.get(x, y) {
                            print!(
                                "\x1b[38;2;{};{};{}m{}\x1b[0m",
                                cell.fg[0], cell.fg[1], cell.fg[2], cell.char
                            );
                        }
                    }
                }
                print!("  ");
            }
        }
        println!();
    }

    Ok(())
}

/// Create an animated demo scene
fn create_demo_scene(time: f64, width: f64, height: f64) -> Scene {
    let center_x = width / 2.0;
    let center_y = height / 2.0;

    // Player ship at center
    let player = SemanticEntity::new(
        EntityType::Ship {
            class: ShipClass::Fighter,
            owner_id: None,
        },
        [center_x, center_y],
    )
    .with_rotation(time * 2.0)
    .with_state(EntityState::full().with_flags(EntityFlags::THRUSTING));

    // Orbiting enemy
    let orbit_radius = 15.0;
    let enemy_x = center_x + orbit_radius * (time * 1.5).cos();
    let enemy_y = center_y + orbit_radius * (time * 1.5).sin();
    let enemy_angle = time * 1.5 + std::f64::consts::PI;

    let enemy = SemanticEntity::new(
        EntityType::Ship {
            class: ShipClass::Bomber,
            owner_id: None,
        },
        [enemy_x, enemy_y],
    )
    .with_rotation(enemy_angle)
    .with_state(
        EntityState::default()
            .with_health(0.7)
            .with_flags(EntityFlags::DAMAGED),
    );

    // Create some asteroids
    let mut entities = vec![player, enemy];

    for i in 0..5 {
        let ax = 10.0 + (i as f64 * 17.0 + time * 2.0) % (width - 20.0);
        let ay = 5.0 + (i as f64 * 11.0) % (height - 10.0);

        let asteroid = SemanticEntity::new(
            EntityType::Environment {
                object_type: catsith_core::entity::EnvironmentType::Asteroid,
            },
            [ax, ay],
        );
        entities.push(asteroid);
    }

    // Occasional explosion
    let mut events = Vec::new();
    if (time * 2.0) % 3.0 < 0.5 {
        events.push(SceneEvent::Explosion {
            position: [center_x + 20.0, center_y - 5.0],
            radius: 3.0,
            intensity: 1.0,
            age: ((time * 2.0) % 3.0) / 0.5,
        });
    }

    Scene::new((time * 60.0) as u64)
        .with_timestamp(time)
        .with_viewport(Viewport::new([center_x, center_y], [width, height]))
        .with_entities(entities)
        .with_environment(Environment::space())
}
