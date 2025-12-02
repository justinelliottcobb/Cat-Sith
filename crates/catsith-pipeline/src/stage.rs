//! Pipeline stage abstraction
//!
//! Stages are the building blocks of render pipelines.
//! Each stage processes a RenderContext and produces a modified context.

use crate::lora::LoraStack;
use async_trait::async_trait;
use catsith_core::{EntityId, PlayerStyle, RenderOutput, Scene};
use std::collections::HashMap;

/// A single stage in the render pipeline
#[async_trait]
pub trait PipelineStage: Send + Sync {
    /// Process the render context
    async fn process(&mut self, context: RenderContext) -> Result<RenderContext, StageError>;

    /// Called when LoRA stack changes
    fn on_lora_change(&mut self, _loras: &LoraStack) {}

    /// Stage name for debugging
    fn name(&self) -> &'static str;

    /// Whether this stage can be skipped (for optimization)
    fn can_skip(&self, _context: &RenderContext) -> bool {
        false
    }
}

/// Stage processing error
#[derive(Debug, thiserror::Error)]
pub enum StageError {
    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Missing required input: {0}")]
    MissingInput(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
}

/// Context passed through pipeline stages
pub struct RenderContext {
    /// Original scene
    pub scene: Scene,

    /// Player style preferences
    pub style: PlayerStyle,

    /// Intermediate data between stages
    pub intermediates: HashMap<String, Intermediate>,

    /// Final output (set by final stage)
    pub output: Option<RenderOutput>,

    /// Performance metrics
    pub metrics: RenderMetrics,
}

impl RenderContext {
    /// Create a new render context
    pub fn new(scene: Scene, style: PlayerStyle) -> Self {
        Self {
            scene,
            style,
            intermediates: HashMap::new(),
            output: None,
            metrics: RenderMetrics::default(),
        }
    }

    /// Get intermediate data by name
    pub fn get_intermediate(&self, name: &str) -> Option<&Intermediate> {
        self.intermediates.get(name)
    }

    /// Set intermediate data
    pub fn set_intermediate(&mut self, name: impl Into<String>, data: Intermediate) {
        self.intermediates.insert(name.into(), data);
    }

    /// Check if we have a specific intermediate
    pub fn has_intermediate(&self, name: &str) -> bool {
        self.intermediates.contains_key(name)
    }

    /// Extract the final output
    pub fn into_output(self) -> Result<RenderOutput, super::PipelineError> {
        self.output.ok_or(super::PipelineError::NoOutput)
    }

    /// Record stage timing
    pub fn record_stage_time(&mut self, stage_name: &str, duration_ms: f64) {
        self.metrics
            .stage_times
            .insert(stage_name.to_string(), duration_ms);
        self.metrics.total_time_ms += duration_ms;
    }
}

/// Intermediate data types passed between stages
#[derive(Debug, Clone)]
pub enum Intermediate {
    /// Entity embeddings from neural encoder
    Embeddings(Vec<EntityEmbedding>),
    /// Generated sprites
    Sprites(Vec<SpriteData>),
    /// Raw tensor data
    Tensors(Vec<TensorData>),
    /// Custom data as bytes
    Custom(Vec<u8>),
}

/// Embedding for a single entity
#[derive(Debug, Clone)]
pub struct EntityEmbedding {
    /// Entity this embedding belongs to
    pub entity_id: EntityId,
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Optional confidence score
    pub confidence: Option<f32>,
}

/// Generated sprite data
#[derive(Debug, Clone)]
pub struct SpriteData {
    /// Entity this sprite represents
    pub entity_id: EntityId,
    /// Pixel data (RGBA)
    pub pixels: Vec<u8>,
    /// Sprite width
    pub width: u32,
    /// Sprite height
    pub height: u32,
    /// Anchor point (normalized 0-1)
    pub anchor: [f32; 2],
}

impl SpriteData {
    /// Create new sprite data
    pub fn new(entity_id: EntityId, width: u32, height: u32) -> Self {
        Self {
            entity_id,
            pixels: vec![0; (width * height * 4) as usize],
            width,
            height,
            anchor: [0.5, 0.5],
        }
    }

    /// Set a pixel
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        if x < self.width && y < self.height {
            let idx = ((y * self.width + x) * 4) as usize;
            self.pixels[idx..idx + 4].copy_from_slice(&rgba);
        }
    }
}

/// Raw tensor data
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Tensor name/identifier
    pub name: String,
    /// Shape (e.g., [batch, channels, height, width])
    pub shape: Vec<usize>,
    /// Flattened data
    pub data: Vec<f32>,
}

/// Render performance metrics
#[derive(Debug, Clone, Default)]
pub struct RenderMetrics {
    /// Time spent in each stage
    pub stage_times: HashMap<String, f64>,
    /// Total render time
    pub total_time_ms: f64,
    /// Number of entities processed
    pub entities_processed: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
}

impl RenderMetrics {
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// A no-op stage for testing
pub struct NoOpStage {
    name: &'static str,
}

impl NoOpStage {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

#[async_trait]
impl PipelineStage for NoOpStage {
    async fn process(&mut self, context: RenderContext) -> Result<RenderContext, StageError> {
        Ok(context)
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn can_skip(&self, _context: &RenderContext) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catsith_core::Scene;

    #[test]
    fn test_render_context() {
        let scene = Scene::new(1);
        let style = PlayerStyle::default();
        let mut context = RenderContext::new(scene, style);

        assert!(!context.has_intermediate("test"));

        context.set_intermediate("test", Intermediate::Custom(vec![1, 2, 3]));
        assert!(context.has_intermediate("test"));
    }

    #[test]
    fn test_sprite_data() {
        let entity_id = EntityId::new();
        let mut sprite = SpriteData::new(entity_id, 8, 8);

        sprite.set_pixel(0, 0, [255, 0, 0, 255]);
        assert_eq!(&sprite.pixels[0..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn test_render_metrics() {
        let metrics = RenderMetrics {
            cache_hits: 80,
            cache_misses: 20,
            ..Default::default()
        };

        assert!((metrics.cache_hit_rate() - 0.8).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_noop_stage() {
        let mut stage = NoOpStage::new("test");
        let context = RenderContext::new(Scene::new(1), PlayerStyle::default());

        let result = stage.process(context).await.unwrap();
        assert_eq!(result.scene.frame_id, 1);
    }
}
