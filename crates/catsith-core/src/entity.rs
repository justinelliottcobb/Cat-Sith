//! Semantic entity types
//!
//! Entities are described by what they ARE, not how they should look.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

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

    /// What kind of thing is this?
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
}

impl SemanticEntity {
    /// Create a new entity with the given type at the specified position
    pub fn new(entity_type: EntityType, position: [f64; 2]) -> Self {
        Self {
            id: EntityId::new(),
            entity_type,
            position,
            velocity: [0.0, 0.0],
            rotation: 0.0,
            angular_velocity: 0.0,
            state: EntityState::default(),
            archetype: None,
            action: None,
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
}

/// Entity type classification
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

    #[test]
    fn test_entity_direction() {
        let entity = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [0.0, 0.0],
        );

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
        let entity = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [100.0, 200.0],
        )
        .with_velocity([1.0, -1.0])
        .with_state(EntityState::full().with_flags(EntityFlags::THRUSTING));

        let json = serde_json::to_string(&entity).unwrap();
        let decoded: SemanticEntity = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.position, entity.position);
        assert!(decoded.state.flags.contains(EntityFlags::THRUSTING));
    }
}
