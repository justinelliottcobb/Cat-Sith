//! Neural Animation Demo
//!
//! Demonstrates a gameplay-style animation using neural-generated sprites.
//! This shows how CatSith can combine neural rendering with real-time animation.
//!
//! The demo can work in two modes:
//! 1. With pre-generated sprites: Uses images from output/ directory
//! 2. Without sprites: Falls back to terminal ASCII rendering
//!
//! To generate sprites first, run:
//!   cargo run --example neural_sprite -p catsith-cli --features candle --release
//!
//! Then run this demo:
//!   cargo run --example neural_animation -p catsith-cli --release

use image::{DynamicImage, GenericImageView, Rgba};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

/// A simple sprite that can be rendered to terminal
struct Sprite {
    /// Original image
    image: DynamicImage,
    /// Terminal ASCII representation (cached)
    ascii: Vec<String>,
    /// Width in terminal characters
    width: usize,
    /// Height in terminal characters
    height: usize,
}

impl Sprite {
    fn load(path: &Path, term_width: usize, term_height: usize) -> Option<Self> {
        let image = image::open(path).ok()?;
        let ascii = Self::image_to_ascii(&image, term_width, term_height);
        Some(Self {
            image,
            width: term_width,
            height: term_height,
            ascii,
        })
    }

    fn image_to_ascii(img: &DynamicImage, width: usize, height: usize) -> Vec<String> {
        let resized = img.resize_exact(
            width as u32,
            height as u32 * 2, // Multiply by 2 since terminal chars are ~2x tall
            image::imageops::FilterType::Nearest,
        );

        let mut lines = Vec::with_capacity(height);

        // Use half-block characters for better resolution
        for y in (0..height * 2).step_by(2) {
            let mut line = String::new();
            for x in 0..width {
                let top = resized.get_pixel(x as u32, y as u32);
                let bottom = if y + 1 < height * 2 {
                    resized.get_pixel(x as u32, (y + 1) as u32)
                } else {
                    top
                };

                // Use half-block character with top and bottom colors
                line.push_str(&Self::pixel_pair_to_ansi(top, bottom));
            }
            line.push_str("\x1b[0m");
            lines.push(line);
        }

        lines
    }

    fn pixel_pair_to_ansi(top: Rgba<u8>, bottom: Rgba<u8>) -> String {
        let [tr, tg, tb, ta] = top.0;
        let [br, bg, bb, ba] = bottom.0;

        // Handle transparency
        if ta < 128 && ba < 128 {
            return " ".to_string();
        } else if ta < 128 {
            // Only bottom pixel visible - use lower half block
            return format!("\x1b[38;2;{};{};{}m\u{2584}", br, bg, bb);
        } else if ba < 128 {
            // Only top pixel visible - use upper half block
            return format!("\x1b[38;2;{};{};{}m\u{2580}", tr, tg, tb);
        }

        // Both pixels visible - use upper half block with both colors
        format!(
            "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m\u{2580}",
            tr, tg, tb, br, bg, bb
        )
    }

    fn render_at(&self, buffer: &mut TerminalBuffer, x: i32, y: i32) {
        for (row, line) in self.ascii.iter().enumerate() {
            let ty = y + row as i32;
            if ty >= 0 && ty < buffer.height as i32 {
                buffer.draw_ansi_string(x, ty, line);
            }
        }
    }
}

/// A simple terminal buffer for compositing sprites
struct TerminalBuffer {
    width: usize,
    height: usize,
    cells: Vec<Vec<String>>,
}

impl TerminalBuffer {
    fn new(width: usize, height: usize) -> Self {
        let cells = vec![vec![" ".to_string(); width]; height];
        Self {
            width,
            height,
            cells,
        }
    }

    fn clear(&mut self) {
        for row in &mut self.cells {
            for cell in row {
                *cell = " ".to_string();
            }
        }
    }

    fn draw_ansi_string(&mut self, x: i32, y: i32, s: &str) {
        if y < 0 || y >= self.height as i32 {
            return;
        }

        // For ANSI strings, we just store the whole line
        // This is a simplified approach - real implementation would parse ANSI
        let row = y as usize;
        if row < self.cells.len() {
            // Store the line at position 0, effectively replacing the row
            if x == 0 {
                self.cells[row] = vec![s.to_string()];
            }
        }
    }

    fn render(&self) -> String {
        let mut output = String::new();
        for row in &self.cells {
            for cell in row {
                output.push_str(cell);
            }
            output.push('\n');
        }
        output
    }
}

/// Game state for the demo
struct GameDemo {
    player_x: f64,
    player_y: f64,
    player_vx: f64,
    enemies: Vec<(f64, f64, f64)>, // x, y, angle
    bullets: Vec<(f64, f64, f64)>, // x, y, vy
    explosions: Vec<(f64, f64, f64)>, // x, y, age
    time: f64,
    score: u32,
}

impl GameDemo {
    fn new(width: f64, height: f64) -> Self {
        Self {
            player_x: width / 2.0,
            player_y: height - 6.0,
            player_vx: 0.0,
            enemies: vec![
                (width * 0.25, 4.0, 0.0),
                (width * 0.5, 6.0, 0.0),
                (width * 0.75, 4.0, 0.0),
            ],
            bullets: Vec::new(),
            explosions: Vec::new(),
            time: 0.0,
            score: 0,
        }
    }

    fn update(&mut self, dt: f64, width: f64, height: f64) {
        self.time += dt;

        // Move player (oscillate for demo)
        self.player_vx = (self.time * 2.0).sin() * 15.0;
        self.player_x += self.player_vx * dt;
        self.player_x = self.player_x.clamp(4.0, width - 4.0);

        // Move enemies (patrol pattern)
        for (i, (ex, ey, angle)) in self.enemies.iter_mut().enumerate() {
            let phase = i as f64 * std::f64::consts::TAU / 3.0;
            *ex = width / 2.0 + (self.time * 0.5 + phase).sin() * (width * 0.35);
            *ey = 4.0 + (self.time * 0.8 + phase).cos().abs() * 8.0;
            *angle = (self.time * 2.0 + phase).sin() * 0.3;
        }

        // Fire bullets periodically
        if (self.time * 3.0).fract() < dt * 3.0 {
            self.bullets.push((self.player_x, self.player_y - 2.0, -30.0));
        }

        // Move bullets
        self.bullets.retain_mut(|(_, by, vy)| {
            *by += *vy * dt;
            *by > 0.0
        });

        // Check bullet-enemy collisions
        let mut new_explosions = Vec::new();
        self.bullets.retain(|(bx, by, _)| {
            for (ex, ey, _) in &self.enemies {
                let dx = bx - ex;
                let dy = by - ey;
                if dx.abs() < 4.0 && dy.abs() < 3.0 {
                    new_explosions.push((*ex, *ey, 0.0));
                    self.score += 100;
                    return false;
                }
            }
            true
        });
        self.explosions.extend(new_explosions);

        // Update explosions
        self.explosions.retain_mut(|(_, _, age)| {
            *age += dt;
            *age < 0.5
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[1;36m=== CatSith Neural Animation Demo ===\x1b[0m\n");

    let term_width = 60;
    let term_height = 24;
    let sprite_size = 8; // Size of sprites in terminal chars

    // Try to load pre-generated sprites
    let mut sprites: HashMap<String, Sprite> = HashMap::new();
    let sprite_dir = Path::new("output");

    if sprite_dir.exists() {
        println!("Looking for pre-generated sprites in output/...");

        for entry in std::fs::read_dir(sprite_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "png").unwrap_or(false) {
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                if let Some(sprite) = Sprite::load(&path, sprite_size, sprite_size / 2) {
                    println!("  Loaded: {} ({} x {})", name, sprite.width, sprite.height);
                    sprites.insert(name, sprite);
                }
            }
        }
    }

    if sprites.is_empty() {
        println!("\x1b[33mNo sprites found. Running in ASCII fallback mode.\x1b[0m");
        println!("To generate sprites, run:");
        println!("  cargo run --example neural_sprite -p catsith-cli --features candle --release\n");
    } else {
        println!("\x1b[32mLoaded {} sprite(s) for neural rendering!\x1b[0m\n", sprites.len());
    }

    println!("Starting animation (press Ctrl+C to stop)...\n");
    std::thread::sleep(Duration::from_secs(1));

    // Game state
    let mut game = GameDemo::new(term_width as f64, term_height as f64);
    let frame_delay = Duration::from_millis(50); // 20 FPS
    let mut last_frame = Instant::now();
    let start_time = Instant::now();
    let demo_duration = Duration::from_secs(15);
    let mut frame_count = 0;

    // Hide cursor and clear screen
    print!("\x1b[?25l\x1b[2J\x1b[H");

    loop {
        let now = Instant::now();
        let dt = (now - last_frame).as_secs_f64();
        last_frame = now;

        // Update game
        game.update(dt, term_width as f64, term_height as f64);

        // Render frame
        let mut buffer = TerminalBuffer::new(term_width, term_height);

        // Draw starfield background
        let stars = "                    *       .           *               .     *          .        *";
        for y in 0..term_height {
            let offset = ((y as f64 + game.time * 5.0) as usize * 7) % stars.len();
            let star_line: String = stars.chars().cycle().skip(offset).take(term_width).collect();
            buffer.cells[y] = vec![format!("\x1b[90m{}\x1b[0m", star_line)];
        }

        // Draw entities using sprites or ASCII fallback
        let has_sprite = !sprites.is_empty();

        // Draw player
        if has_sprite {
            if let Some(sprite) = sprites.get("spaceship") {
                sprite.render_at(&mut buffer, game.player_x as i32 - 4, game.player_y as i32 - 2);
            }
        } else {
            // ASCII fallback
            let py = game.player_y as usize;
            if py > 0 && py < term_height - 1 {
                let ship = format!("\x1b[1;36m  /\\  \x1b[0m");
                let wing = format!("\x1b[1;36m</  \\>\x1b[0m");
                let px = (game.player_x as usize).saturating_sub(3);
                buffer.cells[py - 1] = vec![format!("{:>width$}", ship, width = px + 6)];
                buffer.cells[py] = vec![format!("{:>width$}", wing, width = px + 6)];
            }
        }

        // Draw enemies
        for (ex, ey, _) in &game.enemies {
            let ex = *ex as usize;
            let ey = *ey as usize;
            if ey < term_height - 1 {
                let enemy = format!("\x1b[1;31m<*>\x1b[0m");
                if ex > 1 {
                    buffer.cells[ey] = vec![format!("{:>width$}", enemy, width = ex + 2)];
                }
            }
        }

        // Draw bullets
        for (bx, by, _) in &game.bullets {
            let bx = *bx as usize;
            let by = *by as usize;
            if by < term_height {
                let bullet = format!("\x1b[1;33m|\x1b[0m");
                buffer.cells[by] = vec![format!("{:>width$}", bullet, width = bx + 1)];
            }
        }

        // Draw explosions
        for (ex, ey, age) in &game.explosions {
            let ex = *ex as usize;
            let ey = *ey as usize;
            if ey < term_height {
                let frame = (age * 6.0) as usize;
                let explosion = match frame {
                    0 => "\x1b[1;33m*\x1b[0m",
                    1 => "\x1b[1;31m+*+\x1b[0m",
                    2 => "\x1b[1;31m\\|/\n-*-\n/|\\\x1b[0m",
                    _ => "\x1b[31m . \x1b[0m",
                };
                if ex > 1 {
                    buffer.cells[ey] = vec![format!("{:>width$}", explosion, width = ex + 2)];
                }
            }
        }

        // Render HUD
        let hud = format!(
            "\x1b[1;37mSCORE: {:06}  TIME: {:04.1}s  FPS: {:3.0}  {}\x1b[0m",
            game.score,
            game.time,
            1.0 / dt,
            if has_sprite { "[NEURAL SPRITES]" } else { "[ASCII FALLBACK]" }
        );
        buffer.cells[0] = vec![hud];

        // Move to top-left and render
        print!("\x1b[H");
        print!("{}", buffer.render());

        frame_count += 1;

        // Check if demo should end
        if start_time.elapsed() >= demo_duration {
            break;
        }

        // Frame timing
        let frame_time = now.elapsed();
        if frame_time < frame_delay {
            std::thread::sleep(frame_delay - frame_time);
        }
    }

    // Show cursor and move below the game area
    print!("\x1b[?25h\x1b[{}H", term_height + 2);

    let total_time = start_time.elapsed();
    println!("\n\x1b[1;32m=== Demo Complete! ===\x1b[0m");
    println!("  Frames rendered: {}", frame_count);
    println!("  Total time: {:.2}s", total_time.as_secs_f64());
    println!("  Average FPS: {:.1}", frame_count as f64 / total_time.as_secs_f64());
    println!("  Final score: {}", game.score);

    if sprites.is_empty() {
        println!("\n\x1b[33mTip: Generate neural sprites for enhanced visuals:\x1b[0m");
        println!("  cargo run --example neural_sprite -p catsith-cli --features candle --release");
    }

    Ok(())
}
