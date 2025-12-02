//! CatSith Pipeline - Render Pipeline Orchestration
//!
//! This crate provides the infrastructure for composing render pipelines
//! from multiple stages, with support for LoRA injection and caching.
//!
//! # Pipeline Architecture
//!
//! ```text
//! Scene → [Stage 1] → [Stage 2] → ... → [Stage N] → RenderOutput
//!              ↑           ↑                ↑
//!           LoRAs       LoRAs           LoRAs
//! ```
//!
//! Each stage processes a RenderContext and passes it to the next stage.

pub mod cache;
pub mod lora;
pub mod pipeline;
pub mod scheduler;
pub mod stage;

// Re-export commonly used types
pub use cache::EntityCache;
pub use lora::LoraStack;
pub use pipeline::{PipelineBuilder, PipelineConfig, PipelineError, RenderPipeline};
pub use scheduler::{RenderScheduler, SchedulerConfig};
pub use stage::{PipelineStage, RenderContext};
