//! ExoSpace entity builders
//!
//! Provides convenient builders for creating ExoSpace-specific entities
//! using the CatSith generic entity system.

use catsith_core::entity::{EntityFlags, EntityState, SemanticEntity};
use catsith_core::semantic::{EntityKind, PropertyValue};

/// Ship class in ExoSpace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShipClass {
    Fighter,
    Bomber,
    Scout,
    Cruiser,
    Carrier,
    Station,
}

impl ShipClass {
    /// Get the kind string for this ship class
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Fighter => "fighter",
            Self::Bomber => "bomber",
            Self::Scout => "scout",
            Self::Cruiser => "cruiser",
            Self::Carrier => "carrier",
            Self::Station => "station",
        }
    }

    /// Get the base color for this ship class (RGB)
    pub fn base_color(&self) -> [u8; 3] {
        match self {
            Self::Fighter => [0x40, 0xC0, 0x80], // Green
            Self::Bomber => [0xC0, 0x80, 0x40],  // Orange
            Self::Scout => [0x40, 0x80, 0xC0],   // Blue
            Self::Cruiser => [0x80, 0x40, 0xC0], // Purple
            Self::Carrier => [0xC0, 0xC0, 0x40], // Yellow
            Self::Station => [0x80, 0x80, 0x80], // Gray
        }
    }
}

/// Weapon/projectile type in ExoSpace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeaponType {
    Bullet,
    Bomb,
    Beam,
    Missile,
    Plasma,
    Laser,
}

impl WeaponType {
    /// Get the kind string for this weapon type
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Bullet => "bullet",
            Self::Bomb => "bomb",
            Self::Beam => "beam",
            Self::Missile => "missile",
            Self::Plasma => "plasma",
            Self::Laser => "laser",
        }
    }

    /// Get the base color for this weapon type (RGB)
    pub fn base_color(&self) -> [u8; 3] {
        match self {
            Self::Bullet => [0xFF, 0xFF, 0x00], // Yellow
            Self::Bomb => [0xFF, 0x80, 0x00],   // Orange
            Self::Beam => [0x00, 0xFF, 0xFF],   // Cyan
            Self::Missile => [0xFF, 0x00, 0x00], // Red
            Self::Plasma => [0x80, 0x00, 0xFF], // Purple
            Self::Laser => [0x00, 0xFF, 0x00],  // Green
        }
    }
}

/// Environment object type in ExoSpace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvironmentType {
    Asteroid,
    Debris,
    Station,
    Portal,
    Nebula,
    Star,
    Planet,
}

impl EnvironmentType {
    /// Get the kind string for this environment type
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Asteroid => "asteroid",
            Self::Debris => "debris",
            Self::Station => "station",
            Self::Portal => "portal",
            Self::Nebula => "nebula",
            Self::Star => "star",
            Self::Planet => "planet",
        }
    }
}

/// Builder for ExoSpace entities
pub struct ExoSpaceEntity;

impl ExoSpaceEntity {
    // Ship builders

    /// Create a fighter ship
    pub fn fighter(position: [f64; 2]) -> ShipBuilder {
        ShipBuilder::new(ShipClass::Fighter, position)
    }

    /// Create a bomber ship
    pub fn bomber(position: [f64; 2]) -> ShipBuilder {
        ShipBuilder::new(ShipClass::Bomber, position)
    }

    /// Create a scout ship
    pub fn scout(position: [f64; 2]) -> ShipBuilder {
        ShipBuilder::new(ShipClass::Scout, position)
    }

    /// Create a cruiser ship
    pub fn cruiser(position: [f64; 2]) -> ShipBuilder {
        ShipBuilder::new(ShipClass::Cruiser, position)
    }

    /// Create a carrier ship
    pub fn carrier(position: [f64; 2]) -> ShipBuilder {
        ShipBuilder::new(ShipClass::Carrier, position)
    }

    /// Create a station
    pub fn station(position: [f64; 2]) -> ShipBuilder {
        ShipBuilder::new(ShipClass::Station, position)
    }

    /// Create a ship of the given class
    pub fn ship(class: ShipClass, position: [f64; 2]) -> ShipBuilder {
        ShipBuilder::new(class, position)
    }

    // Projectile builders

    /// Create a bullet projectile
    pub fn bullet(position: [f64; 2]) -> ProjectileBuilder {
        ProjectileBuilder::new(WeaponType::Bullet, position)
    }

    /// Create a missile projectile
    pub fn missile(position: [f64; 2]) -> ProjectileBuilder {
        ProjectileBuilder::new(WeaponType::Missile, position)
    }

    /// Create a beam projectile
    pub fn beam(position: [f64; 2]) -> ProjectileBuilder {
        ProjectileBuilder::new(WeaponType::Beam, position)
    }

    /// Create a plasma projectile
    pub fn plasma(position: [f64; 2]) -> ProjectileBuilder {
        ProjectileBuilder::new(WeaponType::Plasma, position)
    }

    /// Create a projectile of the given type
    pub fn projectile(weapon_type: WeaponType, position: [f64; 2]) -> ProjectileBuilder {
        ProjectileBuilder::new(weapon_type, position)
    }

    // Environment builders

    /// Create an asteroid
    pub fn asteroid(position: [f64; 2]) -> EnvironmentBuilder {
        EnvironmentBuilder::new(EnvironmentType::Asteroid, position)
    }

    /// Create debris
    pub fn debris(position: [f64; 2]) -> EnvironmentBuilder {
        EnvironmentBuilder::new(EnvironmentType::Debris, position)
    }

    /// Create a portal
    pub fn portal(position: [f64; 2]) -> EnvironmentBuilder {
        EnvironmentBuilder::new(EnvironmentType::Portal, position)
    }

    /// Create an environment object
    pub fn environment(env_type: EnvironmentType, position: [f64; 2]) -> EnvironmentBuilder {
        EnvironmentBuilder::new(env_type, position)
    }
}

/// Builder for ship entities
pub struct ShipBuilder {
    class: ShipClass,
    position: [f64; 2],
    velocity: [f64; 2],
    rotation: f64,
    state: EntityState,
    archetype: Option<String>,
    action: Option<String>,
}

impl ShipBuilder {
    fn new(class: ShipClass, position: [f64; 2]) -> Self {
        Self {
            class,
            position,
            velocity: [0.0, 0.0],
            rotation: 0.0,
            state: EntityState::full(),
            archetype: None,
            action: None,
        }
    }

    /// Set velocity
    pub fn with_velocity(mut self, velocity: [f64; 2]) -> Self {
        self.velocity = velocity;
        self
    }

    /// Set rotation (radians)
    pub fn with_rotation(mut self, rotation: f64) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set health (0.0 - 1.0)
    pub fn with_health(mut self, health: f32) -> Self {
        self.state = self.state.with_health(health);
        self
    }

    /// Set flags
    pub fn with_flags(mut self, flags: EntityFlags) -> Self {
        self.state = self.state.with_flags(flags);
        self
    }

    /// Set archetype
    pub fn with_archetype(mut self, archetype: impl Into<String>) -> Self {
        self.archetype = Some(archetype.into());
        self
    }

    /// Set action
    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }

    /// Build the entity
    pub fn build(self) -> SemanticEntity {
        let mut entity =
            SemanticEntity::with_kind(EntityKind::Vehicle, self.class.kind_str(), self.position)
                .with_velocity(self.velocity)
                .with_rotation(self.rotation)
                .with_state(self.state)
                .with_property("ship_class", self.class.kind_str())
                .with_property(
                    "base_color",
                    PropertyValue::Color(self.class.base_color()),
                );

        if let Some(archetype) = self.archetype {
            entity = entity.with_archetype(archetype);
        }
        if let Some(action) = self.action {
            entity = entity.with_action(action);
        }

        entity
    }
}

/// Builder for projectile entities
pub struct ProjectileBuilder {
    weapon_type: WeaponType,
    position: [f64; 2],
    velocity: [f64; 2],
    rotation: f64,
    owner: Option<catsith_core::entity::EntityId>,
}

impl ProjectileBuilder {
    fn new(weapon_type: WeaponType, position: [f64; 2]) -> Self {
        Self {
            weapon_type,
            position,
            velocity: [0.0, 0.0],
            rotation: 0.0,
            owner: None,
        }
    }

    /// Set velocity
    pub fn with_velocity(mut self, velocity: [f64; 2]) -> Self {
        self.velocity = velocity;
        self
    }

    /// Set rotation
    pub fn with_rotation(mut self, rotation: f64) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set owner entity
    pub fn with_owner(mut self, owner: catsith_core::entity::EntityId) -> Self {
        self.owner = Some(owner);
        self
    }

    /// Build the entity
    pub fn build(self) -> SemanticEntity {
        let mut entity = SemanticEntity::with_kind(
            EntityKind::Projectile,
            self.weapon_type.kind_str(),
            self.position,
        )
        .with_velocity(self.velocity)
        .with_rotation(self.rotation)
        .with_property("weapon_type", self.weapon_type.kind_str())
        .with_property(
            "base_color",
            PropertyValue::Color(self.weapon_type.base_color()),
        );

        if let Some(owner) = self.owner {
            entity = entity.with_owner(owner);
        }

        entity
    }
}

/// Builder for environment entities
pub struct EnvironmentBuilder {
    env_type: EnvironmentType,
    position: [f64; 2],
    rotation: f64,
    scale: f64,
}

impl EnvironmentBuilder {
    fn new(env_type: EnvironmentType, position: [f64; 2]) -> Self {
        Self {
            env_type,
            position,
            rotation: 0.0,
            scale: 1.0,
        }
    }

    /// Set rotation
    pub fn with_rotation(mut self, rotation: f64) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set scale
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Build the entity
    pub fn build(self) -> SemanticEntity {
        SemanticEntity::with_kind(
            EntityKind::Environment,
            self.env_type.kind_str(),
            self.position,
        )
        .with_rotation(self.rotation)
        .with_property("scale", self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fighter_builder() {
        let entity = ExoSpaceEntity::fighter([100.0, 200.0])
            .with_velocity([1.0, 0.0])
            .with_health(0.75)
            .build();

        assert_eq!(entity.category, EntityKind::Vehicle);
        assert_eq!(entity.kind, "fighter");
        assert_eq!(entity.velocity, [1.0, 0.0]);
        assert_eq!(entity.state.health, Some(0.75));
    }

    #[test]
    fn test_projectile_builder() {
        let entity = ExoSpaceEntity::plasma([50.0, 50.0])
            .with_velocity([0.0, -10.0])
            .build();

        assert_eq!(entity.category, EntityKind::Projectile);
        assert_eq!(entity.kind, "plasma");
    }

    #[test]
    fn test_environment_builder() {
        let entity = ExoSpaceEntity::asteroid([0.0, 0.0])
            .with_scale(2.0)
            .build();

        assert_eq!(entity.category, EntityKind::Environment);
        assert_eq!(entity.kind, "asteroid");
    }
}
