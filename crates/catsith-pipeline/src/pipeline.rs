//! Render pipeline orchestration

use crate::cache::EntityCache;
use crate::lora::LoraStack;
use crate::stage::{PipelineStage, RenderContext};
use catsith_core::{PlayerStyle, RenderOutput, Scene};
use std::time::Instant;
use thiserror::Error;
use tracing::{Level, debug, info, span};

/// Pipeline errors
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("No stages in pipeline")]
    NoStages,

    #[error("No output produced")]
    NoOutput,

    #[error("Stage '{0}' failed: {1}")]
    StageError(String, String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Unsupported output format")]
    UnsupportedFormat,

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// A complete render pipeline
pub struct RenderPipeline {
    /// Pipeline stages (in order)
    stages: Vec<Box<dyn PipelineStage>>,
    /// Active LoRA stack
    lora_stack: LoraStack,
    /// Entity embedding cache
    entity_cache: EntityCache,
    /// Configuration
    config: PipelineConfig,
}

impl RenderPipeline {
    /// Create a new pipeline builder
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Render a complete scene
    pub async fn render(
        &mut self,
        scene: &Scene,
        style: &PlayerStyle,
    ) -> Result<RenderOutput, PipelineError> {
        let span = span!(Level::DEBUG, "render_pipeline", frame_id = scene.frame_id);
        let _enter = span.enter();

        if self.stages.is_empty() {
            return Err(PipelineError::NoStages);
        }

        let start = Instant::now();
        let mut context = RenderContext::new(scene.clone(), style.clone());
        context.metrics.entities_processed = scene.entities.len();

        // Process through each stage
        for stage in &mut self.stages {
            let stage_name = stage.name();

            // Check if stage can be skipped
            if stage.can_skip(&context) {
                debug!(stage = stage_name, "Skipping stage");
                continue;
            }

            let stage_start = Instant::now();

            context = stage
                .process(context)
                .await
                .map_err(|e| PipelineError::StageError(stage_name.to_string(), e.to_string()))?;

            let stage_duration = stage_start.elapsed().as_secs_f64() * 1000.0;
            context.record_stage_time(stage_name, stage_duration);

            debug!(
                stage = stage_name,
                duration_ms = stage_duration,
                "Stage completed"
            );
        }

        let total_duration = start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            total_ms = total_duration,
            entities = context.metrics.entities_processed,
            "Pipeline completed"
        );

        context.into_output()
    }

    /// Update the LoRA stack
    pub fn set_loras(&mut self, loras: LoraStack) {
        self.lora_stack = loras;

        // Notify all stages
        for stage in &mut self.stages {
            stage.on_lora_change(&self.lora_stack);
        }
    }

    /// Get current LoRA stack
    pub fn loras(&self) -> &LoraStack {
        &self.lora_stack
    }

    /// Get the entity cache
    pub fn entity_cache(&self) -> &EntityCache {
        &self.entity_cache
    }

    /// Get mutable entity cache
    pub fn entity_cache_mut(&mut self) -> &mut EntityCache {
        &mut self.entity_cache
    }

    /// Get pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Get number of stages
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get stage names
    pub fn stage_names(&self) -> Vec<&'static str> {
        self.stages.iter().map(|s| s.name()).collect()
    }
}

/// Pipeline builder
pub struct PipelineBuilder {
    stages: Vec<Box<dyn PipelineStage>>,
    config: PipelineConfig,
}

impl PipelineBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            config: PipelineConfig::default(),
        }
    }

    /// Add a pipeline stage
    pub fn stage(mut self, stage: impl PipelineStage + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Add a boxed pipeline stage
    pub fn stage_boxed(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Set configuration
    pub fn config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Set frame time budget
    pub fn frame_budget_ms(mut self, budget: f64) -> Self {
        self.config.frame_budget_ms = budget;
        self
    }

    /// Enable/disable temporal coherence
    pub fn temporal_coherence(mut self, enabled: bool) -> Self {
        self.config.temporal_coherence = enabled;
        self
    }

    /// Set cache size
    pub fn cache_size_mb(mut self, size: u32) -> Self {
        self.config.cache_size_mb = size;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Result<RenderPipeline, PipelineError> {
        if self.stages.is_empty() {
            return Err(PipelineError::NoStages);
        }

        info!(
            stages = self.stages.len(),
            cache_mb = self.config.cache_size_mb,
            "Building render pipeline"
        );

        Ok(RenderPipeline {
            stages: self.stages,
            lora_stack: LoraStack::empty(),
            entity_cache: EntityCache::new(self.config.cache_size_mb as usize * 1024 * 1024),
            config: self.config,
        })
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum frame time budget (for LOD adjustment)
    pub frame_budget_ms: f64,
    /// Enable temporal coherence between frames
    pub temporal_coherence: bool,
    /// Entity cache size in megabytes
    pub cache_size_mb: u32,
    /// Maximum concurrent renders
    pub max_concurrent: usize,
    /// Enable profiling
    pub profiling: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            frame_budget_ms: 16.0, // 60fps target
            temporal_coherence: true,
            cache_size_mb: 256,
            max_concurrent: 2,
            profiling: false,
        }
    }
}

impl PipelineConfig {
    /// Configuration for terminal rendering (low resource)
    pub fn terminal() -> Self {
        Self {
            frame_budget_ms: 33.0, // 30fps is fine for terminal
            temporal_coherence: false,
            cache_size_mb: 64,
            max_concurrent: 1,
            profiling: false,
        }
    }

    /// Configuration for high-quality rendering
    pub fn high_quality() -> Self {
        Self {
            frame_budget_ms: 16.0,
            temporal_coherence: true,
            cache_size_mb: 512,
            max_concurrent: 4,
            profiling: false,
        }
    }

    /// Configuration for cinematic/offline rendering
    pub fn cinematic() -> Self {
        Self {
            frame_budget_ms: 1000.0, // 1 second is acceptable
            temporal_coherence: true,
            cache_size_mb: 2048,
            max_concurrent: 1,
            profiling: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::NoOpStage;

    #[tokio::test]
    async fn test_pipeline_builder() {
        let pipeline = RenderPipeline::builder()
            .stage(NoOpStage::new("stage1"))
            .stage(NoOpStage::new("stage2"))
            .frame_budget_ms(16.0)
            .build()
            .unwrap();

        assert_eq!(pipeline.stage_count(), 2);
        assert_eq!(pipeline.stage_names(), vec!["stage1", "stage2"]);
    }

    #[tokio::test]
    async fn test_empty_pipeline_fails() {
        let result = RenderPipeline::builder().build();
        assert!(matches!(result, Err(PipelineError::NoStages)));
    }

    #[tokio::test]
    async fn test_pipeline_render() {
        // Create a simple pipeline with a stage that sets output
        struct OutputStage;

        #[async_trait::async_trait]
        impl PipelineStage for OutputStage {
            async fn process(
                &mut self,
                mut context: RenderContext,
            ) -> Result<RenderContext, crate::stage::StageError> {
                context.output = Some(RenderOutput::Terminal(catsith_core::TerminalFrame::new(
                    10, 10,
                )));
                Ok(context)
            }

            fn name(&self) -> &'static str {
                "output_stage"
            }
        }

        let mut pipeline = RenderPipeline::builder()
            .stage(OutputStage)
            .build()
            .unwrap();

        let scene = Scene::new(1);
        let style = PlayerStyle::default();

        let output = pipeline.render(&scene, &style).await.unwrap();
        assert!(output.is_terminal());
    }

    #[test]
    fn test_pipeline_config() {
        let config = PipelineConfig::terminal();
        assert_eq!(config.cache_size_mb, 64);

        let config = PipelineConfig::cinematic();
        assert_eq!(config.cache_size_mb, 2048);
    }
}
