//! CatSith CLI - Neural Rendering Frontend
//!
//! A tool for testing and demonstrating CatSith rendering capabilities.

use clap::{Parser, Subcommand};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

mod commands;

/// CatSith - Neural Rendering Frontend
///
/// Named after the Cat Sìth of Celtic folklore - a fairy creature that appears
/// differently to different observers.
#[derive(Parser)]
#[command(name = "catsith")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Disable colored output
    #[arg(long)]
    no_color: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a rendering demo
    Demo {
        /// Demo type to run
        #[arg(value_enum, default_value = "terminal")]
        demo_type: DemoType,

        /// Output width
        #[arg(short, long, default_value = "80")]
        width: u32,

        /// Output height
        #[arg(long, default_value = "24")]
        height: u32,

        /// Number of frames to render
        #[arg(short, long, default_value = "60")]
        frames: u32,
    },

    /// Show system capabilities
    Caps,

    /// Render a single frame from a scene file
    Render {
        /// Input scene file (JSON)
        input: String,

        /// Output file
        #[arg(short, long)]
        output: Option<String>,

        /// Output format
        #[arg(short, long, default_value = "terminal")]
        format: OutputFormat,
    },

    /// Manage LoRAs
    Lora {
        #[command(subcommand)]
        action: LoraAction,
    },

    /// Start a rendering server
    Server {
        /// Listen address
        #[arg(short, long, default_value = "127.0.0.1:7878")]
        address: String,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum DemoType {
    /// Terminal ASCII rendering demo
    Terminal,
    /// Image raster rendering demo
    Raster,
    /// Scene showcase
    Showcase,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    /// Terminal ASCII output
    Terminal,
    /// PNG image output
    Png,
    /// JSON scene dump
    Json,
}

#[derive(Subcommand)]
enum LoraAction {
    /// List available LoRAs
    List,
    /// Show LoRA details
    Info { name: String },
    /// Validate a LoRA
    Validate { path: String },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_ansi(!cli.no_color)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    match cli.command {
        Commands::Demo {
            demo_type,
            width,
            height,
            frames,
        } => {
            commands::demo::run(demo_type, width, height, frames).await?;
        }

        Commands::Caps => {
            commands::caps::run();
        }

        Commands::Render {
            input,
            output,
            format,
        } => {
            commands::render::run(&input, output.as_deref(), format).await?;
        }

        Commands::Lora { action } => match action {
            LoraAction::List => commands::lora::list(),
            LoraAction::Info { name } => commands::lora::info(&name),
            LoraAction::Validate { path } => commands::lora::validate(&path)?,
        },

        Commands::Server { address } => {
            commands::server::run(&address).await?;
        }
    }

    Ok(())
}
