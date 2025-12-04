# Cat Sith

Neural rendering frontend for games and real-time applications.

Named after the Cat Sith of Celtic folklore - a fairy creature that appears differently to different observers. CatSith transforms semantic scene descriptions into visual output tailored to each player's hardware and preferences.

## Overview

```
Game Server → Semantic Scene Description → CatSith → Visual Output
                                              ↑
                                    Player Style / LoRAs / Hardware
```

CatSith consumes semantic scene descriptions and produces visual output across a massive capability spectrum - from CPU-rendered terminal ASCII to high-resolution neural-generated frames.

## Features

- **Multi-backend rendering**: Terminal (ASCII/Unicode), Raster (CPU sprites), Neural (AI-generated)
- **Semantic scene understanding**: Entities described by meaning, not pixels
- **Domain-agnostic**: Core types work for any game genre
- **Style personalization**: LoRA marketplace for unique visual experiences
- **Hardware adaptive**: Scales from terminal to GPU-accelerated rendering

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| `catsith-core` | Stable | Semantic types, scene representation, entity system |
| `catsith-pipeline` | Stable | Render pipeline stages, context flow |
| `catsith-backend-terminal` | Working | ASCII/Unicode rendering with ANSI colors |
| `catsith-backend-raster` | Working | CPU-based sprite compositing |
| `catsith-backend-neural` | Working | Candle-based Stable Diffusion inference |
| `catsith-kan` | Foundation | KAN-based sprite generation, needs training |
| `catsith-lora` | Planned | LoRA loading and application |
| `catsith-domain-exospace` | Working | Space game domain (ships, asteroids, effects) |
| `catsith-api` | Stable | Public API types |
| `catsith-cli` | Working | Demo application with examples |

## Architecture

```
crates/
├── catsith-core/           # Semantic types, entities, scenes
├── catsith-api/            # Public API surface
├── catsith-pipeline/       # Render pipeline stages
├── catsith-backend-terminal/  # Terminal ASCII renderer
├── catsith-backend-raster/    # CPU sprite renderer
├── catsith-backend-neural/    # Neural inference renderer
├── catsith-kan/            # Kolmogorov-Arnold Networks for sprites
├── catsith-lora/           # LoRA model loading
├── catsith-domain-exospace/   # Space game domain types
└── catsith-cli/            # CLI demo application
```

## Quick Start

```bash
# Build the project
cargo build --release

# Run the terminal demo
cargo run --release --bin catsith -- demo

# Run the animation example
cargo run --example terminal_animation
```

## Terminal Rendering Demo

The terminal backend renders semantic scenes as colored ASCII art:

```
┌──────────────────────────────────────────────────┐
│                    *                             │
│        ◇                                    *    │
│                        @                         │
│   *            <>                                │
│                        ●                    ◇    │
│            █                                     │
│                                *                 │
│     ◇                  ░░░                       │
└──────────────────────────────────────────────────┘
```

Entities are rendered with semantic awareness - ships show thrust effects, damaged entities flicker, explosions animate.

## Neural Rendering

The neural backend supports GPU-accelerated image generation using Hugging Face's Candle framework.

### Candle Diffusion Pipeline

The primary neural rendering approach uses Stable Diffusion models via Candle:

```rust
use catsith_backend_neural::{CandleDiffusionPipeline, DiffusionConfig};

// Create a pipeline configured for pixel art sprites
let config = DiffusionConfig::pixel_art();
let mut pipeline = CandleDiffusionPipeline::new(config)?;

// Load model components (downloads tokenizer from HuggingFace if needed)
pipeline.load()?;

// Generate an image from a text prompt
let image = pipeline.generate("pixel art spaceship sprite", None, 42)?;
image.save("spaceship.png")?;
```

Supported model versions:
- **SD 1.5** - Most compatible, good for pixel art LoRAs
- **SD 2.1** - Improved quality
- **SDXL** - High resolution, best quality
- **SDXL Turbo** - Fast generation (4 steps)

### Model Downloads

Download Stable Diffusion models using the Hugging Face CLI:

```bash
# Install HF CLI if needed
pip install huggingface_hub

# Download a pixel art model
hf download kohbanye/pixel-art-style --local-dir models/pixel-art-style
```

The pipeline supports both safetensors and PyTorch checkpoint formats.

### KAN (Kolmogorov-Arnold Networks)

An alternative approach using learnable B-spline activation functions:
- Parameter-efficient sprite generation
- Smooth latent space interpolation
- WebGPU-accelerated inference
- Interpretable learned functions

See [crates/catsith-kan/ROADMAP.md](crates/catsith-kan/ROADMAP.md) for development status.

### Legacy ONNX Support

ONNX model support is available for compatibility with converted models.
See [docs/NEURAL_MODELS.md](docs/NEURAL_MODELS.md) for model recommendations.

## Domain System

CatSith is domain-agnostic. The core types (`SemanticEntity`, `Scene`, `SceneEvent`) use categorical identifiers that domain crates interpret.

### ExoSpace Domain (included)

Space game entities:
- Ships: `fighter`, `bomber`, `scout`, `cruiser`, `station`
- Objects: `asteroid`, `debris`, `cargo`
- Effects: `bullet`, `missile`, `explosion`, `shield`
- Environment: `nebula`, `starfield`

```rust
use catsith_domain_exospace::ExoSpaceEntity;

let player = ExoSpaceEntity::fighter([100.0, 200.0])
    .with_rotation(0.5)
    .with_flags(EntityFlags::THRUSTING)
    .build();
```

### Creating Custom Domains

1. Define entity types and their visual representations
2. Create sprite/glyph mappings for each backend
3. Register with the renderer's sprite resolver

## Configuration

Player style configuration controls rendering behavior:

```rust
use catsith_core::PlayerStyle;

let style = PlayerStyle::builder()
    .quality_tier(QualityTier::High)
    .color_scheme(ColorScheme::Vibrant)
    .backend_preference(BackendType::Neural)
    .build();
```

## Documentation

- [NEURAL_MODELS.md](docs/NEURAL_MODELS.md) - Neural model recommendations and integration
- [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - Guide for training custom models
- [catsith-kan/ROADMAP.md](crates/catsith-kan/ROADMAP.md) - KAN development roadmap

## Development

```bash
# Check all crates
cargo check --workspace

# Run tests
cargo test --workspace

# Format code
cargo fmt --all

# Lint
cargo clippy --workspace
```

### Building with Optional Features

```bash
# Enable Candle-based Stable Diffusion (recommended for neural rendering)
cargo build -p catsith-backend-neural --features candle

# Enable KAN support in neural backend
cargo build -p catsith-backend-neural --features kan

# Enable both
cargo build -p catsith-backend-neural --features "candle,kan"
```

## Requirements

- Rust 1.85+ (2024 edition)
- For terminal backend: Any terminal with ANSI color support
- For neural backend (Candle): CPU or CUDA GPU (CUDA recommended for performance)
- For KAN backend: WebGPU-capable GPU

## License

MIT OR Apache-2.0

## Acknowledgments

- KAN implementation based on [kan-gpu](https://github.com/...)
- Inspired by the Kolmogorov-Arnold representation theorem
- Cat Sith folklore from Scottish/Irish mythology
