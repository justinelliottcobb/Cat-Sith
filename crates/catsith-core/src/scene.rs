//! Semantic scene description
//!
//! A Scene represents a complete visual moment that CatSith should render.
//! It contains no rendering instructions - only semantic meaning.

use crate::entity::{EntityId, SemanticEntity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A complete scene to render
///
/// The game server produces this. CatSith interprets it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Unique frame identifier
    pub frame_id: u64,

    /// Scene timestamp (for temporal coherence)
    pub timestamp: f64,

    /// Camera/viewport information
    pub viewport: Viewport,

    /// All entities in the scene
    pub entities: Vec<SemanticEntity>,

    /// Ambient properties
    pub environment: Environment,

    /// Active events (explosions, effects)
    pub events: Vec<SceneEvent>,

    /// Entity identity cache reference
    /// (Full identities sent separately, referenced by ID here)
    pub identity_refs: HashMap<EntityId, IdentityRef>,
}

impl Scene {
    /// Create a new empty scene
    pub fn new(frame_id: u64) -> Self {
        Self {
            frame_id,
            timestamp: 0.0,
            viewport: Viewport::default(),
            entities: Vec::new(),
            environment: Environment::default(),
            events: Vec::new(),
            identity_refs: HashMap::new(),
        }
    }

    /// Set the timestamp
    pub fn with_timestamp(mut self, timestamp: f64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set the viewport
    pub fn with_viewport(mut self, viewport: Viewport) -> Self {
        self.viewport = viewport;
        self
    }

    /// Add an entity to the scene
    pub fn with_entity(mut self, entity: SemanticEntity) -> Self {
        self.entities.push(entity);
        self
    }

    /// Add multiple entities
    pub fn with_entities(mut self, entities: impl IntoIterator<Item = SemanticEntity>) -> Self {
        self.entities.extend(entities);
        self
    }

    /// Set the environment
    pub fn with_environment(mut self, environment: Environment) -> Self {
        self.environment = environment;
        self
    }

    /// Add an event
    pub fn with_event(mut self, event: SceneEvent) -> Self {
        self.events.push(event);
        self
    }
}

/// Camera/viewport information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Viewport {
    /// Center position in world space
    pub center: [f64; 2],

    /// Visible world units (width, height)
    pub extent: [f64; 2],

    /// Rotation (radians)
    pub rotation: f64,

    /// Focal entity (camera follows this)
    pub focus: Option<EntityId>,

    /// Zoom level (1.0 = normal)
    pub zoom: f64,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            extent: [100.0, 100.0],
            rotation: 0.0,
            focus: None,
            zoom: 1.0,
        }
    }
}

impl Viewport {
    /// Create a viewport centered at a position with given extent
    pub fn new(center: [f64; 2], extent: [f64; 2]) -> Self {
        Self {
            center,
            extent,
            ..Default::default()
        }
    }

    /// Set the focus entity
    pub fn with_focus(mut self, entity_id: EntityId) -> Self {
        self.focus = Some(entity_id);
        self
    }

    /// Set the zoom level
    pub fn with_zoom(mut self, zoom: f64) -> Self {
        self.zoom = zoom.max(0.1);
        self
    }

    /// Check if a world position is visible in this viewport
    pub fn contains(&self, world_pos: [f64; 2]) -> bool {
        let half_extent = [self.extent[0] / 2.0, self.extent[1] / 2.0];
        let rel_x = (world_pos[0] - self.center[0]).abs();
        let rel_y = (world_pos[1] - self.center[1]).abs();
        rel_x <= half_extent[0] && rel_y <= half_extent[1]
    }
}

/// Environment/atmosphere settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    /// Base ambiance type
    pub ambiance: Ambiance,

    /// Semantic descriptors
    /// e.g., "deep space", "nebula field", "asteroid belt"
    pub descriptors: Vec<String>,

    /// Lighting mood
    pub lighting: LightingMood,

    /// Background color hint (RGB)
    pub background_color: Option<[u8; 3]>,

    /// Fog/visibility distance (None = infinite)
    pub visibility: Option<f64>,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            ambiance: Ambiance::Void,
            descriptors: Vec::new(),
            lighting: LightingMood::Neutral,
            background_color: None,
            visibility: None,
        }
    }
}

impl Environment {
    /// Create a space environment
    pub fn space() -> Self {
        Self {
            ambiance: Ambiance::Void,
            descriptors: vec!["deep space".to_string()],
            lighting: LightingMood::Cold,
            background_color: Some([5, 5, 15]),
            visibility: None,
        }
    }

    /// Create a nebula environment
    pub fn nebula() -> Self {
        Self {
            ambiance: Ambiance::Nebula,
            descriptors: vec![
                "nebula field".to_string(),
                "colorful gas clouds".to_string(),
            ],
            lighting: LightingMood::Ethereal,
            background_color: Some([20, 10, 30]),
            visibility: Some(500.0),
        }
    }

    /// Add a descriptor
    pub fn with_descriptor(mut self, descriptor: impl Into<String>) -> Self {
        self.descriptors.push(descriptor.into());
        self
    }

    /// Set lighting mood
    pub fn with_lighting(mut self, lighting: LightingMood) -> Self {
        self.lighting = lighting;
        self
    }
}

/// Base ambiance type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Ambiance {
    /// Empty space
    #[default]
    Void,
    /// Colorful gas clouds
    Nebula,
    /// Rocky debris field
    Asteroid,
    /// Artificial structures
    Station,
    /// Planetary atmosphere
    Atmosphere,
    /// Non-physical/abstract space
    Abstract,
}

/// Lighting mood affects overall color and contrast
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum LightingMood {
    /// Balanced lighting
    #[default]
    Neutral,
    /// Warm, orange/yellow tones
    Warm,
    /// Cold, blue tones
    Cold,
    /// High contrast, deep shadows
    Dramatic,
    /// Soft, glowing, dreamlike
    Ethereal,
    /// Stark, bright, industrial
    Harsh,
}

/// Dynamic scene events (explosions, beams, particles)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneEvent {
    /// Explosion effect
    Explosion {
        position: [f64; 2],
        radius: f64,
        intensity: f64,
        /// 0.0 = just started, 1.0 = fading out
        age: f64,
    },

    /// Beam/laser effect
    Beam {
        start: [f64; 2],
        end: [f64; 2],
        intensity: f64,
        color: Option<[u8; 3]>,
    },

    /// Particle effect
    Particle {
        position: [f64; 2],
        velocity: [f64; 2],
        /// e.g., "spark", "debris", "energy"
        particle_type: String,
    },

    /// Screen flash effect
    Flash {
        intensity: f64,
        color: Option<[u8; 3]>,
    },

    /// Screen shake
    Shake { intensity: f64, duration: f64 },
}

/// Reference to a cached entity identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityRef {
    /// Content hash for cache lookup
    pub identity_hash: [u8; 32],
    /// Frame ID when this identity was last updated
    pub last_update: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::EntityKind;

    #[test]
    fn test_scene_builder() {
        let entity = SemanticEntity::with_kind(EntityKind::Vehicle, "fighter", [50.0, 50.0]);

        let scene = Scene::new(1)
            .with_timestamp(0.016)
            .with_viewport(Viewport::new([50.0, 50.0], [100.0, 100.0]))
            .with_entity(entity)
            .with_environment(Environment::space());

        assert_eq!(scene.frame_id, 1);
        assert_eq!(scene.entities.len(), 1);
        assert_eq!(scene.environment.ambiance, Ambiance::Void);
    }

    #[test]
    fn test_viewport_contains() {
        let viewport = Viewport::new([0.0, 0.0], [100.0, 100.0]);

        assert!(viewport.contains([0.0, 0.0]));
        assert!(viewport.contains([49.0, 49.0]));
        assert!(!viewport.contains([51.0, 0.0]));
        assert!(!viewport.contains([0.0, 51.0]));
    }

    #[test]
    fn test_scene_serialization() {
        let scene = Scene::new(42)
            .with_timestamp(1.5)
            .with_environment(Environment::nebula());

        let json = serde_json::to_string(&scene).unwrap();
        let decoded: Scene = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.frame_id, 42);
        assert_eq!(decoded.environment.ambiance, Ambiance::Nebula);
    }
}
