//! CatSith LoRA - LoRA Management
//!
//! Handles loading, validating, and applying LoRA (Low-Rank Adaptation) weights
//! to customize the visual rendering style.
//!
//! # LoRA System Overview
//!
//! LoRAs allow players to personalize their visual experience without modifying
//! base models. Each LoRA:
//!
//! - Has a unique content hash for verification
//! - Targets specific model layers
//! - Can be combined (stacked) with other LoRAs
//! - Has a weight (0.0-1.0) controlling its influence

pub mod injector;
pub mod loader;
pub mod manifest;
pub mod registry;
pub mod validator;

pub use injector::{LoraInjector, LoraWeights};
pub use loader::{LoaderConfig, LoraLoader};
pub use manifest::{LoraCategory, LoraManifest, WeightInfo};
pub use registry::{LoraRegistry, RegistryEntry};
pub use validator::{LoraValidator, ValidationResult};
