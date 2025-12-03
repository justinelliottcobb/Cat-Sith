//! CatSith Core - Semantic rendering types and traits
//!
//! CatSith is a neural rendering frontend that consumes semantic scene descriptions
//! and produces visual output across a massive capability spectrum - from CPU-rendered
//! terminal ASCII to 16K photorealistic cinematic frames.
//!
//! Named after the Cat Sìth of Celtic folklore - a fairy creature that appears
//! differently to different observers.
//!
//! # Core Philosophy
//!
//! ```text
//! Game Server → Semantic Scene Description → CatSith → Visual Output
//!                                               ↑
//!                                     Player Style / LoRAs / Hardware
//! ```
//!
//! The semantic description is canonical. What the player sees is personal
//! interpretation filtered through their perception pipeline.
//!
//! # Domain Independence
//!
//! CatSith core types are domain-agnostic. Entity types, events, and environments
//! are described using semantic categories (`EntityKind`, `EventKind`, `AmbianceKind`)
//! combined with string-based `kind` identifiers that domain crates define.
//!
//! For example, a space game might define entities with:
//! - `kind: EntityKind::Vehicle, kind: "fighter"`
//! - `kind: EntityKind::Projectile, kind: "plasma_bolt"`
//!
//! While a fantasy game might use:
//! - `kind: EntityKind::Character, kind: "dragon"`
//! - `kind: EntityKind::Projectile, kind: "fireball"`
//!
//! Renderers look up visual representations using these identifiers.

pub mod capability;
pub mod entity;
pub mod intent;
pub mod output;
pub mod scene;
pub mod semantic;
pub mod style;

// Re-export commonly used types
pub use capability::{BackendType, GpuCapabilities, GpuVendor, RenderCapabilities};
pub use entity::{EntityFlags, EntityId, EntityState, EntityType, SemanticEntity};
pub use intent::{
    ColorScheme, EntityIdentity, ImageFormat, LoraRef, OutputFormat, PreviousRender, QualityTier,
    RenderIntent, RenderTarget,
};
pub use output::{ImageFrame, RenderOutput, TensorFrame, TerminalCell, TerminalFrame};
pub use scene::{Ambiance, Environment, IdentityRef, LightingMood, Scene, SceneEvent, Viewport};
pub use semantic::{AmbianceKind, EntityKind, EventKind, Properties, PropertyValue};
pub use style::PlayerStyle;
