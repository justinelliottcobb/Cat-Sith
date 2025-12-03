//! Terminal Animation Example
//!
//! Renders an animated scene directly to stdout, showing ships in motion.
//! This demonstrates real-time rendering without requiring an interactive terminal.
//!
//! Run with: cargo run --example terminal_animation

use catsith_backend_terminal::TerminalRenderer;
use catsith_core::entity::EntityFlags;
use catsith_core::scene::{Scene, Viewport};
use catsith_core::style::PlayerStyle;
use catsith_core::RenderOutput;
use catsith_domain_exospace::{ExoSpaceEntity, ExoSpaceEnvironment, ExoSpaceEvent};
use catsith_pipeline::stage::RenderContext;
use catsith_pipeline::PipelineStage;
use std::time::{Duration, Instant};

fn create_scene(time: f64, width: f64, height: f64) -> Scene {
    let center_x = width / 2.0;
    let center_y = height / 2.0;

    // Player ship rotating slowly
    let player = ExoSpaceEntity::fighter([center_x, center_y])
        .with_rotation(time * 0.5)
        .with_flags(EntityFlags::THRUSTING)
        .build();

    // Enemy orbiting the player
    let orbit_radius = 12.0;
    let enemy_x = center_x + orbit_radius * (time * 1.2).cos();
    let enemy_y = center_y + orbit_radius * 0.6 * (time * 1.2).sin();
    let enemy_angle = time * 1.2 + std::f64::consts::PI;

    let enemy = ExoSpaceEntity::bomber([enemy_x, enemy_y])
        .with_rotation(enemy_angle)
        .with_health(0.5)
        .with_flags(EntityFlags::DAMAGED | EntityFlags::FIRING)
        .build();

    // Scout patrolling
    let scout_x = 8.0 + (time * 3.0).sin().abs() * (width - 16.0);
    let scout = ExoSpaceEntity::scout([scout_x, 4.0])
        .with_rotation(if (time * 3.0).cos() > 0.0 {
            std::f64::consts::FRAC_PI_2
        } else {
            -std::f64::consts::FRAC_PI_2
        })
        .build();

    // Asteroids drifting
    let mut entities = vec![player, enemy, scout];
    for i in 0..3 {
        let ax = (10.0 + i as f64 * 20.0 + time * (1.0 + i as f64 * 0.3)) % width;
        let ay = 2.0 + (i as f64 * 7.0 + time * 0.5).sin().abs() * (height - 6.0);
        entities.push(ExoSpaceEntity::asteroid([ax, ay]).build());
    }

    // Bullets fired periodically
    if ((time * 2.0) as i32) % 3 == 0 {
        let bullet_y = center_y - 4.0 - ((time * 2.0).fract() * 8.0);
        entities.push(
            ExoSpaceEntity::bullet([center_x, bullet_y])
                .with_velocity([0.0, -15.0])
                .build(),
        );
    }

    // Build scene
    let mut scene = Scene::new((time * 60.0) as u64)
        .with_timestamp(time)
        .with_viewport(Viewport::new([center_x, center_y], [width, height]))
        .with_entities(entities)
        .with_environment(ExoSpaceEnvironment::space());

    // Add explosion when enemy is hit (simulated)
    if (time * 0.7).fract() < 0.15 {
        scene = scene.with_event(
            ExoSpaceEvent::explosion([enemy_x, enemy_y], 3.0)
                .with_intensity(0.7)
                .with_age((time * 0.7).fract() / 0.15)
                .build(),
        );
    }

    scene
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 50;
    let height = 18;
    let num_frames = 30;
    let frame_delay = Duration::from_millis(100);

    let mut renderer = TerminalRenderer::new(width, height);

    println!("\x1b[1;36m=== CatSith Terminal Animation Demo ===\x1b[0m");
    println!("Rendering {} frames...\n", num_frames);

    let start = Instant::now();

    for frame_num in 0..num_frames {
        let time = frame_num as f64 / 10.0;
        let scene = create_scene(time, width as f64, height as f64);
        let context = RenderContext::new(scene, PlayerStyle::terminal());

        let result = renderer.process(context).await?;

        if let Some(RenderOutput::Terminal(frame)) = result.output {
            // Clear previous frame (move cursor up)
            if frame_num > 0 {
                print!("\x1b[{}A", height + 2);
            }

            // Print frame number
            println!(
                "\x1b[90mFrame {}/{} (t={:.1}s)\x1b[0m",
                frame_num + 1,
                num_frames,
                time
            );

            // Print frame with border
            println!("\x1b[90m┌{}┐\x1b[0m", "─".repeat(width as usize));
            for line in frame.to_ansi().lines() {
                println!("\x1b[90m│\x1b[0m{}\x1b[90m│\x1b[0m", line);
            }
            println!("\x1b[90m└{}┘\x1b[0m", "─".repeat(width as usize));
        }

        tokio::time::sleep(frame_delay).await;
    }

    let elapsed = start.elapsed();
    let fps = num_frames as f64 / elapsed.as_secs_f64();

    println!("\n\x1b[1mAnimation complete!\x1b[0m");
    println!("  Frames: {}", num_frames);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Average FPS: {:.1}", fps);

    Ok(())
}
