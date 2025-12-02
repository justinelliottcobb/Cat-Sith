//! Semantic type system for domain-agnostic entity and event classification
//!
//! This module provides the core abstraction layer that allows CatSith to work
//! with any domain (space games, fantasy RPGs, sports sims, etc.) without
//! hardcoding domain-specific types.
//!
//! # Design Philosophy
//!
//! Instead of hardcoded enums like `ShipClass::Fighter`, we use a semantic
//! classification system:
//!
//! - `EntityKind` - What category of thing is this? (vehicle, projectile, creature, etc.)
//! - `kind` field - A string identifier that domains define (e.g., "fighter", "dragon")
//! - `archetype` - Semantic mood/behavior (e.g., "aggressive", "damaged")
//!
//! Renderers look up visual representations using these string identifiers,
//! which can be registered by domain crates or loaded from LoRAs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// High-level entity category
///
/// These are universal categories that apply across all domains.
/// The specific `kind` string provides domain-specific detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EntityKind {
    /// A controllable vehicle/vessel (ship, car, dragon mount, etc.)
    Vehicle,
    /// A character or creature (player, NPC, monster, etc.)
    Character,
    /// A projectile or thrown object (bullet, arrow, fireball, etc.)
    Projectile,
    /// An environmental/static object (asteroid, tree, building, etc.)
    #[default]
    Environment,
    /// An effect or ephemeral object (explosion particle, magic aura, etc.)
    Effect,
    /// User interface element rendered in world space
    Interface,
    /// Domain-specific category not covered above
    Custom,
}

impl EntityKind {
    /// Get a human-readable description of this category
    pub fn description(&self) -> &'static str {
        match self {
            Self::Vehicle => "controllable vehicle or vessel",
            Self::Character => "character or creature",
            Self::Projectile => "projectile or thrown object",
            Self::Environment => "environmental or static object",
            Self::Effect => "visual effect or ephemeral object",
            Self::Interface => "world-space UI element",
            Self::Custom => "domain-specific entity",
        }
    }
}

/// High-level visual event category
///
/// These are universal effect categories. Renderers interpret these
/// according to the current aesthetic and domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EventKind {
    /// Radial burst effect (explosion, magic burst, shockwave)
    #[default]
    Burst,
    /// Linear ray/beam effect (laser, lightning, arrow trail)
    Ray,
    /// Point or small particle effect (spark, muzzle flash)
    Particle,
    /// Screen-wide effect (flash, fade, color shift)
    ScreenEffect,
    /// Camera effect (shake, zoom, pan)
    CameraEffect,
    /// Ambient/environmental effect (fog, rain, aurora)
    Ambient,
}

impl EventKind {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Burst => "radial burst or explosion",
            Self::Ray => "linear ray or beam",
            Self::Particle => "point particle effect",
            Self::ScreenEffect => "screen-wide visual effect",
            Self::CameraEffect => "camera movement effect",
            Self::Ambient => "ambient environmental effect",
        }
    }
}

/// High-level environment/ambiance category
///
/// Generic categories for environment types that renderers can interpret.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AmbianceKind {
    /// Empty, minimal background (space void, empty sky, blank canvas)
    #[default]
    Empty,
    /// Dense, obscuring atmosphere (fog, nebula, smoke, underwater)
    Dense,
    /// Natural outdoor environment (forest, ocean, desert, plains)
    Natural,
    /// Artificial/constructed environment (city, dungeon, space station)
    Constructed,
    /// Abstract or non-physical space (menu, dream, digital realm)
    Abstract,
}

impl AmbianceKind {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Empty => "empty, minimal background",
            Self::Dense => "dense, obscuring atmosphere",
            Self::Natural => "natural outdoor environment",
            Self::Constructed => "artificial or constructed environment",
            Self::Abstract => "abstract or non-physical space",
        }
    }
}

/// Semantic properties bag for extensible entity data
///
/// Domains can store arbitrary key-value data here that their
/// renderers understand but the core doesn't need to know about.
pub type Properties = HashMap<String, PropertyValue>;

/// A property value that can be serialized
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertyValue {
    /// String value
    String(String),
    /// Numeric value
    Number(f64),
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Color value (RGB)
    Color([u8; 3]),
}

impl PropertyValue {
    /// Try to get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as number
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            Self::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as color
    pub fn as_color(&self) -> Option<[u8; 3]> {
        match self {
            Self::Color(c) => Some(*c),
            _ => None,
        }
    }
}

impl From<&str> for PropertyValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<String> for PropertyValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<f64> for PropertyValue {
    fn from(n: f64) -> Self {
        Self::Number(n)
    }
}

impl From<i64> for PropertyValue {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl From<bool> for PropertyValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<[u8; 3]> for PropertyValue {
    fn from(c: [u8; 3]) -> Self {
        Self::Color(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_kind() {
        assert_eq!(EntityKind::default(), EntityKind::Environment);
        assert!(!EntityKind::Vehicle.description().is_empty());
    }

    #[test]
    fn test_property_value() {
        let props: Properties = [
            ("name".to_string(), PropertyValue::from("test")),
            ("health".to_string(), PropertyValue::from(0.75)),
            ("active".to_string(), PropertyValue::from(true)),
        ]
        .into();

        assert_eq!(props.get("name").unwrap().as_str(), Some("test"));
        assert_eq!(props.get("health").unwrap().as_number(), Some(0.75));
        assert_eq!(props.get("active").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn test_serialization() {
        let kind = EntityKind::Vehicle;
        let json = serde_json::to_string(&kind).unwrap();
        let decoded: EntityKind = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, kind);
    }
}
