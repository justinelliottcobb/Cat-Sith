//! Neural Sprite Generation Example
//!
//! Generates pixel art sprites using Stable Diffusion.
//! Requires the `candle` feature and a downloaded SD model.
//!
//! Run with: cargo run --example neural_sprite --features candle
//!
//! Make sure you have a model downloaded:
//!   models/pixel-art-style/  (SD 1.5 pixel art model)

use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[1;36m=== CatSith Neural Sprite Generation ===\x1b[0m\n");

    #[cfg(not(feature = "candle"))]
    {
        eprintln!("\x1b[1;31mError: This example requires the 'candle' feature.\x1b[0m");
        eprintln!("Run with: cargo run --example neural_sprite --features candle");
        std::process::exit(1);
    }

    #[cfg(feature = "candle")]
    {
        use catsith_backend_neural::{CandleDiffusionPipeline, DiffusionConfig, StableDiffusionVersion};

        // Check for model
        let model_path = PathBuf::from("models/pixel-art-style");
        if !model_path.exists() {
            eprintln!("\x1b[1;31mError: Model not found at {:?}\x1b[0m", model_path);
            eprintln!("\nDownload a model with:");
            eprintln!("  hf download kohbanye/pixel-art-style --local-dir models/pixel-art-style");
            std::process::exit(1);
        }

        println!("Model path: {:?}", model_path);
        println!("Loading model components...\n");

        // Create pipeline configured for small sprites
        // Using 256x256 with 8 steps for faster CPU inference
        // For GPU, you can increase to 512x512 with 20 steps
        let config = DiffusionConfig {
            version: StableDiffusionVersion::V1_5,
            model_path: model_path.clone(),
            num_steps: 8,  // Fewer steps for speed (quality trade-off)
            guidance_scale: 7.5,
            width: 256,   // Smaller for faster CPU inference
            height: 256,
            use_f16: true,
            use_flash_attn: false,
        };

        let mut pipeline = CandleDiffusionPipeline::new(config)?;

        // Load each component with progress
        print!("  Loading tokenizer... ");
        let start = Instant::now();
        pipeline.load_tokenizer()?;
        println!("done ({:.1}s)", start.elapsed().as_secs_f32());

        print!("  Loading text encoder... ");
        let start = Instant::now();
        pipeline.load_text_encoder()?;
        println!("done ({:.1}s)", start.elapsed().as_secs_f32());

        print!("  Loading VAE... ");
        let start = Instant::now();
        pipeline.load_vae()?;
        println!("done ({:.1}s)", start.elapsed().as_secs_f32());

        print!("  Loading UNet... ");
        let start = Instant::now();
        pipeline.load_unet()?;
        println!("done ({:.1}s)", start.elapsed().as_secs_f32());

        println!("\n\x1b[1;32mModel loaded successfully!\x1b[0m\n");

        // Generate sprites
        // Starting with just one to test - CPU inference is slow!
        let prompts = [
            ("pixel art spaceship, side view, 16-bit retro style, game sprite", "spaceship.png"),
            // Uncomment these for more sprites (will take a long time on CPU):
            // ("pixel art asteroid, space rock, 16-bit retro style", "asteroid.png"),
            // ("pixel art explosion effect, orange and yellow flames", "explosion.png"),
        ];

        for (prompt, filename) in prompts {
            println!("Generating: {}", filename);
            println!("  Prompt: \"{}\"", prompt);

            let start = Instant::now();
            let image = pipeline.generate(
                prompt,
                Some("blurry, low quality, text, watermark"),
                42,
            )?;

            let elapsed = start.elapsed();
            println!("  Generated in {:.1}s", elapsed.as_secs_f32());

            // Save image
            let output_path = format!("output/{}", filename);
            std::fs::create_dir_all("output")?;
            image.save(&output_path)?;
            println!("  Saved to: {}\n", output_path);
        }

        println!("\x1b[1;32mAll sprites generated!\x1b[0m");
        println!("Check the output/ directory for results.");
    }

    Ok(())
}
