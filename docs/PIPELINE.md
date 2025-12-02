# CatSith Pipeline System

## Overview

The pipeline system provides composable, staged rendering. Each stage transforms a RenderContext, passing it to the next stage until final output is produced.

## Pipeline Architecture

```
Scene + Style → [Stage 1] → [Stage 2] → ... → [Stage N] → Output
                    ↓           ↓                 ↓
               Intermediate  Intermediate    RenderOutput
                  Data         Data
```

## Creating a Pipeline

```rust
use catsith_pipeline::*;

let pipeline = RenderPipeline::builder()
    .stage(MyPreprocessStage::new())
    .stage(TerminalRenderer::new(80, 24))
    .stage(MyPostprocessStage::new())
    .frame_budget_ms(16.0)
    .temporal_coherence(true)
    .cache_size_mb(256)
    .build()?;
```

## Implementing a Stage

```rust
use async_trait::async_trait;
use catsith_pipeline::stage::*;

pub struct MyStage {
    // Stage state
}

#[async_trait]
impl PipelineStage for MyStage {
    async fn process(&mut self, mut context: RenderContext)
        -> Result<RenderContext, StageError>
    {
        // Read from context.scene, context.style
        // Optionally read from context.intermediates

        // Do processing...

        // Store intermediate results
        context.set_intermediate("my_data", Intermediate::Custom(data));

        // Or set final output
        context.output = Some(RenderOutput::Terminal(frame));

        Ok(context)
    }

    fn on_lora_change(&mut self, loras: &LoraStack) {
        // React to LoRA stack changes
    }

    fn name(&self) -> &'static str {
        "my_stage"
    }

    fn can_skip(&self, context: &RenderContext) -> bool {
        // Return true if this stage can be skipped
        false
    }
}
```

## RenderContext

The context passed through stages:

```rust
pub struct RenderContext {
    pub scene: Scene,              // Original scene
    pub style: PlayerStyle,        // Player preferences
    pub intermediates: HashMap<String, Intermediate>,
    pub output: Option<RenderOutput>,
    pub metrics: RenderMetrics,
}
```

## Intermediate Data Types

```rust
pub enum Intermediate {
    Embeddings(Vec<EntityEmbedding>),  // Neural embeddings
    Sprites(Vec<SpriteData>),          // Generated sprites
    Tensors(Vec<TensorData>),          // Raw tensor data
    Custom(Vec<u8>),                   // Arbitrary data
}
```

## Configuration

### PipelineConfig

```rust
pub struct PipelineConfig {
    pub frame_budget_ms: f64,      // Target frame time
    pub temporal_coherence: bool,   // Enable temporal smoothing
    pub cache_size_mb: u32,         // Entity cache size
    pub max_concurrent: usize,      // Parallel renders
    pub profiling: bool,            // Enable profiling
}
```

### Preset Configurations

```rust
// Low-resource terminal rendering
let config = PipelineConfig::terminal();

// High-quality real-time
let config = PipelineConfig::high_quality();

// Offline/cinematic
let config = PipelineConfig::cinematic();
```

## Entity Cache

The pipeline maintains an entity cache for:
- Reusing embeddings across frames
- Caching rendered sprites
- Temporal coherence data

```rust
let cache = pipeline.entity_cache_mut();
cache.insert(entity_id, CachedEntity::new(content_hash));

if let Some(cached) = cache.get(&entity_id) {
    // Use cached embedding/sprite
}
```

## LoRA Integration

LoRAs are managed at the pipeline level:

```rust
// Create LoRA stack
let mut stack = LoraStack::empty();
stack.push(aesthetic_lora, 0.8);
stack.push(entity_lora, 0.5);

// Apply to pipeline
pipeline.set_loras(stack);
```

Stages receive LoRA change notifications via `on_lora_change()`.

## Render Scheduling

For frame-rate targeting:

```rust
use catsith_pipeline::scheduler::*;

let mut scheduler = RenderScheduler::new(SchedulerConfig {
    target_fps: 60.0,
    min_fps: 30.0,
    adaptive_quality: true,
    ..Default::default()
});

scheduler.begin_frame();
// ... render ...
let result = scheduler.end_frame();

if result.quality_change == QualityChange::Decrease {
    // Lower quality tier
}
```

## Error Handling

```rust
#[derive(Debug, Error)]
pub enum PipelineError {
    NoStages,           // Pipeline has no stages
    NoOutput,           // No stage produced output
    StageError(String), // Stage failed
    ModelNotFound(String),
    UnsupportedFormat,
}
```

## Performance Tips

1. **Stage Ordering**: Put cheap stages first to fail fast
2. **Intermediate Caching**: Store expensive computations
3. **can_skip()**: Implement to skip unnecessary work
4. **Parallel Stages**: Use `max_concurrent` for multi-entity processing
5. **Frame Budget**: Set realistic budgets for your target hardware
