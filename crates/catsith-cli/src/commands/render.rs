//! Render command

use crate::OutputFormat;
use catsith_backend_terminal::TerminalRenderer;
use catsith_core::RenderOutput;
use catsith_core::scene::Scene;
use catsith_core::style::PlayerStyle;
use catsith_pipeline::PipelineStage;
use catsith_pipeline::stage::RenderContext;
use std::fs;
use tracing::info;

pub async fn run(
    input: &str,
    output: Option<&str>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Rendering scene from {}", input);

    // Load scene from JSON
    let scene_json = fs::read_to_string(input)?;
    let scene: Scene = serde_json::from_str(&scene_json)?;

    info!("Loaded scene with {} entities", scene.entities.len());

    match format {
        OutputFormat::Terminal => {
            let mut renderer = TerminalRenderer::new(80, 24);
            let context = RenderContext::new(scene, PlayerStyle::terminal());
            let result = renderer.process(context).await?;

            if let Some(RenderOutput::Terminal(frame)) = result.output {
                let ansi = frame.to_ansi();

                match output {
                    Some(path) => {
                        fs::write(path, &ansi)?;
                        println!("Output written to {}", path);
                    }
                    None => {
                        print!("{}", ansi);
                    }
                }
            }
        }

        OutputFormat::Png => {
            println!("PNG output not yet implemented");
        }

        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&scene)?;
            match output {
                Some(path) => {
                    fs::write(path, &json)?;
                    println!("Scene JSON written to {}", path);
                }
                None => {
                    println!("{}", json);
                }
            }
        }
    }

    Ok(())
}
