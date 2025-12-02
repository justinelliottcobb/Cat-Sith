//! Semantic entity types
//!
//! Entities are described by what they ARE, not how they should look.
//!
//! # Domain Independence
//!
//! Entities use a two-level classification system:
//!
//! 1. **`EntityKind`** - A universal category (Vehicle, Character, Projectile, etc.)
//! 2. **`kind`** - A domain-specific string identifier ("fighter", "dragon", etc.)
//!
//! This allows CatSith to work with any domain while renderers can look up
//! visual representations using the `kind` string.
//!
//! For backward compatibility, the old `EntityType` enum is preserved but
//! deprecated. New code should use `EntityKind` + `kind` string.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::semantic::{EntityKind, Properties};

/// Unique entity identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub Uuid);

impl EntityId {
    /// Create a new random entity ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for EntityId {
    fn default() -> Self {
        Self::new()
    }
}

/// A semantic entity in the scene
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEntity {
    /// Unique identifier for this entity
    pub id: EntityId,

    /// High-level entity category (Vehicle, Character, Projectile, etc.)
    ///
    /// This is the universal classification that all domains share.
    #[serde(default)]
    pub category: EntityKind,

    /// Domain-specific type identifier
    ///
    /// This string is used by renderers to look up visual representations.
    /// Examples: "fighter", "asteroid", "plasma_bolt", "dragon", "fireball"
    #[serde(default)]
    pub kind: String,

    /// Legacy entity type (deprecated, use `category` + `kind` instead)
    ///
    /// Preserved for backward compatibility with existing code.
    #[deprecated(note = "Use `category` and `kind` fields instead")]
    pub entity_type: EntityType,

    /// Position in world space
    pub position: [f64; 2],

    /// Velocity (for motion blur, prediction)
    pub velocity: [f64; 2],

    /// Orientation (radians)
    pub rotation: f64,

    /// Angular velocity
    pub angular_velocity: f64,

    /// Current state
    pub state: EntityState,

    /// Semantic archetype
    /// e.g., "aggressive", "damaged", "stealthy"
    pub archetype: Option<String>,

    /// Action context
    /// e.g., "pursuing", "fleeing", "patrolling"
    pub action: Option<String>,

    /// Domain-specific properties
    ///
    /// Domains can store arbitrary key-value data here.
    #[serde(default)]
    pub properties: Properties,

    /// Owner/parent entity ID (for projectiles, effects, etc.)
    #[serde(default)]
    pub owner_id: Option<EntityId>,
}

impl SemanticEntity {
    /// Create a new entity with category and kind at the specified position
    ///
    /// This is the preferred constructor for new code.
    pub fn with_kind(category: EntityKind, kind: impl Into<String>, position: [f64; 2]) -> Self {
        #[allow(deprecated)]
        Self {
            id: EntityId::new(),
            category,
            kind: kind.into(),
            entity_type: EntityType::Custom {
                type_name: String::new(),
                properties: HashMap::new(),
            },
            position,
            velocity: [0.0, 0.0],
            rotation: 0.0,
            angular_velocity: 0.0,
            state: EntityState::default(),
            archetype: None,
            action: None,
            properties: Properties::new(),
            owner_id: None,
        }
    }

    /// Create a new entity with the given type at the specified position
    ///
    /// Deprecated: Use `with_kind` instead for new code.
    #[deprecated(note = "Use SemanticEntity::with_kind() instead")]
    pub fn new(entity_type: EntityType, position: [f64; 2]) -> Self {
        // Derive category and kind from legacy EntityType
        let (category, kind) = entity_type.to_category_and_kind();

        #[allow(deprecated)]
        Self {
            id: EntityId::new(),
            category,
            kind,
            entity_type,
            position,
            velocity: [0.0, 0.0],
            rotation: 0.0,
            angular_velocity: 0.0,
            state: EntityState::default(),
            archetype: None,
            action: None,
            properties: Properties::new(),
            owner_id: None,
        }
    }

    /// Set the velocity
    pub fn with_velocity(mut self, velocity: [f64; 2]) -> Self {
        self.velocity = velocity;
        self
    }

    /// Set the rotation
    pub fn with_rotation(mut self, rotation: f64) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set the state
    pub fn with_state(mut self, state: EntityState) -> Self {
        self.state = state;
        self
    }

    /// Set the archetype
    pub fn with_archetype(mut self, archetype: impl Into<String>) -> Self {
        self.archetype = Some(archetype.into());
        self
    }

    /// Set the action
    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }

    /// Get the direction as a discrete index (0-7 for 8 directions)
    pub fn direction_index(&self) -> u8 {
        // Normalize angle to 0..2π
        let angle = self.rotation.rem_euclid(std::f64::consts::TAU);
        // Convert to 8 directions (0 = up, clockwise)
        ((angle / std::f64::consts::TAU * 8.0 + 0.5) as u8) % 8
    }

    /// Set a property value
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<crate::semantic::PropertyValue>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Set the owner entity ID
    pub fn with_owner(mut self, owner_id: EntityId) -> Self {
        self.owner_id = Some(owner_id);
        self
    }

    /// Get the sprite lookup key for this entity
    ///
    /// Returns a string that renderers can use to look up visual representations.
    /// Format: "{kind}_{direction}" or just "{kind}" if direction doesn't apply.
    pub fn sprite_key(&self) -> String {
        if self.kind.is_empty() {
            // Fall back to legacy entity_type
            #[allow(deprecated)]
            return self.entity_type.to_sprite_key(self.direction_index());
        }

        match self.category {
            EntityKind::Vehicle | EntityKind::Character => {
                format!("{}_{}", self.kind, self.direction_index())
            }
            _ => self.kind.clone(),
        }
    }
}

/// Entity type classification
///
/// **Deprecated**: This enum contains domain-specific types (Ship, WeaponType, etc.)
/// that belong in domain crates. New code should use `SemanticEntity::with_kind()`
/// with `EntityKind` + string-based `kind` identifiers.
///
/// This enum is preserved for backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    /// Player-controlled ship
    Ship {
        class: ShipClass,
        /// Links to PlayerIdentity
        owner_id: Option<Uuid>,
    },

    /// Projectile
    Projectile {
        weapon_type: WeaponType,
        /// Entity that fired this projectile
        owner_id: EntityId,
    },

    /// Environmental object
    Environment { object_type: EnvironmentType },

    /// NPC/Automaton
    Automaton { automaton_type: String },

    /// Abstract/custom entity
    Custom {
        type_name: String,
        properties: HashMap<String, String>,
    },
}

impl EntityType {
    /// Convert legacy EntityType to (EntityKind, kind_string)
    ///
    /// Used for backward compatibility when constructing entities.
    pub fn to_category_and_kind(&self) -> (EntityKind, String) {
        match self {
            Self::Ship { class, .. } => {
                let kind = match class {
                    ShipClass::Fighter => "fighter",
                    ShipClass::Bomber => "bomber",
                    ShipClass::Scout => "scout",
                    ShipClass::Cruiser => "cruiser",
                    ShipClass::Carrier => "carrier",
                    ShipClass::Station => "station",
                };
                (EntityKind::Vehicle, kind.to_string())
            }
            Self::Projectile { weapon_type, .. } => {
                let kind = match weapon_type {
                    WeaponType::Bullet => "bullet",
                    WeaponType::Bomb => "bomb",
                    WeaponType::Beam => "beam",
                    WeaponType::Missile => "missile",
                    WeaponType::Plasma => "plasma",
                    WeaponType::Laser => "laser",
                };
                (EntityKind::Projectile, kind.to_string())
            }
            Self::Environment { object_type } => {
                let kind = match object_type {
                    EnvironmentType::Asteroid => "asteroid",
                    EnvironmentType::Debris => "debris",
                    EnvironmentType::Station => "station",
                    EnvironmentType::Portal => "portal",
                    EnvironmentType::Nebula => "nebula",
                    EnvironmentType::Star => "star",
                    EnvironmentType::Planet => "planet",
                };
                (EntityKind::Environment, kind.to_string())
            }
            Self::Automaton { automaton_type } => {
                (EntityKind::Character, automaton_type.clone())
            }
            Self::Custom { type_name, .. } => (EntityKind::Custom, type_name.clone()),
        }
    }

    /// Get sprite lookup key for legacy entity types
    pub fn to_sprite_key(&self, direction: u8) -> String {
        match self {
            Self::Ship { class, .. } => {
                let name = match class {
                    ShipClass::Fighter => "fighter",
                    ShipClass::Bomber => "bomber",
                    ShipClass::Scout => "scout",
                    ShipClass::Cruiser => "cruiser",
                    ShipClass::Carrier => "carrier",
                    ShipClass::Station => "station",
                };
                format!("{}_{}", name, direction)
            }
            Self::Projectile { weapon_type, .. } => match weapon_type {
                WeaponType::Bullet => "bullet".to_string(),
                WeaponType::Bomb => "bomb".to_string(),
                WeaponType::Beam => "beam".to_string(),
                WeaponType::Missile => "missile".to_string(),
                WeaponType::Plasma => "plasma".to_string(),
                WeaponType::Laser => "laser".to_string(),
            },
            Self::Environment { object_type } => match object_type {
                EnvironmentType::Asteroid => "asteroid".to_string(),
                EnvironmentType::Debris => "debris".to_string(),
                EnvironmentType::Station => "station".to_string(),
                EnvironmentType::Portal => "portal".to_string(),
                EnvironmentType::Nebula => "nebula".to_string(),
                EnvironmentType::Star => "star".to_string(),
                EnvironmentType::Planet => "planet".to_string(),
            },
            Self::Automaton { automaton_type } => automaton_type.clone(),
            Self::Custom { type_name, .. } => type_name.clone(),
        }
    }
}

/// Ship class classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShipClass {
    Fighter,
    Bomber,
    Scout,
    Cruiser,
    Carrier,
    Station,
}

/// Weapon/projectile type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeaponType {
    Bullet,
    Bomb,
    Beam,
    Missile,
    Plasma,
    Laser,
}

/// Environment object type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnvironmentType {
    Asteroid,
    Debris,
    Station,
    Portal,
    Nebula,
    Star,
    Planet,
}

/// Current state of an entity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityState {
    /// Normalized health (0.0 - 1.0)
    pub health: Option<f32>,

    /// Normalized energy (0.0 - 1.0)
    pub energy: Option<f32>,

    /// Normalized shield (0.0 - 1.0)
    pub shield: Option<f32>,

    /// State flags
    pub flags: EntityFlags,
}

impl EntityState {
    /// Create a new entity state with full health
    pub fn full() -> Self {
        Self {
            health: Some(1.0),
            energy: Some(1.0),
            shield: Some(1.0),
            flags: EntityFlags::empty(),
        }
    }

    /// Set health
    pub fn with_health(mut self, health: f32) -> Self {
        self.health = Some(health.clamp(0.0, 1.0));
        self
    }

    /// Set energy
    pub fn with_energy(mut self, energy: f32) -> Self {
        self.energy = Some(energy.clamp(0.0, 1.0));
        self
    }

    /// Set flags
    pub fn with_flags(mut self, flags: EntityFlags) -> Self {
        self.flags = flags;
        self
    }
}

bitflags! {
    /// State flags for entities
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct EntityFlags: u32 {
        /// Engine is active
        const THRUSTING = 0b0000_0001;
        /// Weapons are firing
        const FIRING    = 0b0000_0010;
        /// Entity has taken damage
        const DAMAGED   = 0b0000_0100;
        /// Shield is active
        const SHIELDED  = 0b0000_1000;
        /// Entity is cloaked/invisible
        const CLOAKED   = 0b0001_0000;
        /// Boost/afterburner active
        const BOOSTING  = 0b0010_0000;
        /// Entity is invulnerable
        const INVULN    = 0b0100_0000;
        /// Entity is charging an ability
        const CHARGING  = 0b1000_0000;
    }
}

impl Serialize for EntityFlags {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EntityFlags {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u32::deserialize(deserializer)?;
        EntityFlags::from_bits(bits)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid EntityFlags bits: {}", bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::EntityKind;

    #[test]
    fn test_entity_direction() {
        let entity = SemanticEntity::with_kind(EntityKind::Vehicle, "fighter", [0.0, 0.0]);

        // 0 radians = up = direction 0
        assert_eq!(entity.clone().with_rotation(0.0).direction_index(), 0);

        // π/2 radians = right = direction 2
        let half_pi = std::f64::consts::FRAC_PI_2;
        assert_eq!(entity.clone().with_rotation(half_pi).direction_index(), 2);
    }

    #[test]
    fn test_entity_flags() {
        let flags = EntityFlags::THRUSTING | EntityFlags::FIRING;
        assert!(flags.contains(EntityFlags::THRUSTING));
        assert!(flags.contains(EntityFlags::FIRING));
        assert!(!flags.contains(EntityFlags::DAMAGED));
    }

    #[test]
    fn test_entity_serialization() {
        let entity = SemanticEntity::with_kind(EntityKind::Vehicle, "fighter", [100.0, 200.0])
            .with_velocity([1.0, -1.0])
            .with_state(EntityState::full().with_flags(EntityFlags::THRUSTING));

        let json = serde_json::to_string(&entity).unwrap();
        let decoded: SemanticEntity = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.position, entity.position);
        assert!(decoded.state.flags.contains(EntityFlags::THRUSTING));
    }

    #[test]
    fn test_sprite_key() {
        // Vehicles include direction in sprite key (format: "{kind}_{direction}")
        let entity = SemanticEntity::with_kind(EntityKind::Vehicle, "fighter", [0.0, 0.0]);
        assert_eq!(entity.sprite_key(), "fighter_0");

        // Projectiles don't include direction
        let entity = SemanticEntity::with_kind(EntityKind::Projectile, "missile", [0.0, 0.0]);
        assert_eq!(entity.sprite_key(), "missile");

        // Environment entities don't include direction
        let entity = SemanticEntity::with_kind(EntityKind::Environment, "asteroid", [0.0, 0.0]);
        assert_eq!(entity.sprite_key(), "asteroid");
    }

    #[test]
    #[allow(deprecated)]
    fn test_legacy_entity_type_compatibility() {
        // Test that the deprecated API still works for backward compatibility
        let entity = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [50.0, 50.0],
        );

        // Legacy entity should still produce correct sprite key (includes direction)
        assert_eq!(entity.sprite_key(), "fighter_0");
        assert_eq!(entity.category, EntityKind::Vehicle);
    }
}
