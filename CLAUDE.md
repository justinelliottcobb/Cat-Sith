# CatSith - Claude Code Guidelines

## Project Overview

CatSith is a neural rendering frontend that consumes semantic scene descriptions and produces visual output across a massive capability spectrum - from CPU-rendered terminal ASCII to 16K photorealistic cinematic frames.

Named after the Cat Sìth of Celtic folklore - a fairy creature that appears differently to different observers.

## Core Philosophy

```
Game Server → Semantic Scene Description → CatSith → Visual Output
                                              ↑
                                    Player Style / LoRAs / Hardware
```

The semantic description is canonical. What the player sees is personal interpretation filtered through their perception pipeline.

## Architecture

### Workspace Structure

- `catsith-core` - Core types, scene descriptions, entity models
- `catsith-api` - Client/server protocol for receiving scenes
- `catsith-pipeline` - Render pipeline orchestration and staging
- `catsith-backend-terminal` - ASCII/terminal rendering
- `catsith-backend-neural` - Neural network inference rendering
- `catsith-backend-raster` - Traditional sprite-based rendering
- `catsith-lora` - LoRA loading, validation, and injection
- `catsith-cli` - Command-line tool for testing/demos

### Key Abstractions

1. **Scene** - Complete semantic description of what to render
2. **Entity** - Semantic entity (ship, projectile, environment object)
3. **PipelineStage** - Composable render stage trait
4. **PlayerStyle** - User preferences for rendering (aesthetic, palette)
5. **LoRA** - Low-rank adaptations for customizing visual style

## Build & Test

```bash
# Build all crates
cargo build

# Run tests
cargo test

# Run CLI demo
cargo run -p catsith-cli -- demo terminal

# Show capabilities
cargo run -p catsith-cli -- caps
```

## Code Patterns

### Creating a Scene

```rust
use catsith_core::*;

let entity = SemanticEntity::new(
    EntityType::Ship { class: ShipClass::Fighter, owner_id: None },
    [50.0, 50.0],
)
.with_rotation(0.0)
.with_state(EntityState::full().with_flags(EntityFlags::THRUSTING));

let scene = Scene::new(frame_id)
    .with_viewport(Viewport::new([50.0, 50.0], [100.0, 100.0]))
    .with_entity(entity)
    .with_environment(Environment::space());
```

### Creating a Pipeline

```rust
use catsith_pipeline::*;
use catsith_backend_terminal::TerminalRenderer;

let pipeline = RenderPipeline::builder()
    .stage(TerminalRenderer::new(80, 24))
    .frame_budget_ms(16.0)
    .build()?;

let output = pipeline.render(&scene, &style).await?;
```

### Implementing a Pipeline Stage

```rust
#[async_trait]
impl PipelineStage for MyStage {
    async fn process(&mut self, mut context: RenderContext) -> Result<RenderContext, StageError> {
        // Process context.scene
        // Set context.output = Some(RenderOutput::...)
        Ok(context)
    }

    fn name(&self) -> &'static str {
        "my_stage"
    }
}
```

## Important Notes

1. **Semantic First**: Never include rendering hints in scene descriptions. Entities describe WHAT they are, not HOW they look.

2. **Backend Agnostic**: The same scene should render correctly on all backends (terminal, raster, neural).

3. **LoRA Safety**: Always validate LoRA hashes before loading. Never trust unverified LoRA files.

4. **Quality Tiers**: Respect hardware capabilities. Use `RenderCapabilities::detect()` to determine appropriate quality.

5. **Temporal Coherence**: For animation, maintain entity identity across frames for smooth rendering.

## File Locations

- Scene types: `crates/catsith-core/src/scene.rs`
- Entity types: `crates/catsith-core/src/entity.rs`
- Pipeline: `crates/catsith-pipeline/src/pipeline.rs`
- Stage trait: `crates/catsith-pipeline/src/stage.rs`
- Terminal sprites: `crates/catsith-backend-terminal/src/sprites.rs`
- LoRA manifest: `crates/catsith-lora/src/manifest.rs`
