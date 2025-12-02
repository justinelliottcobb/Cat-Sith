//! ExoSpace Domain Types for CatSith
//!
//! This crate provides domain-specific types, entity builders, and rendering
//! configurations for the ExoSpace space combat game.
//!
//! # Entity Types
//!
//! ExoSpace defines the following entity kinds:
//!
//! - **Ships**: Fighter, Bomber, Scout, Cruiser, Carrier, Station
//! - **Projectiles**: Bullet, Bomb, Beam, Missile, Plasma, Laser
//! - **Environment**: Asteroid, Debris, Station, Portal, Nebula, Star, Planet
//!
//! # Usage
//!
//! ```ignore
//! use catsith_domain_exospace::entities::ExoSpaceEntity;
//!
//! let fighter = ExoSpaceEntity::fighter([100.0, 200.0])
//!     .with_velocity([1.0, 0.0])
//!     .build();
//! ```

pub mod entities;
pub mod environments;
pub mod events;
pub mod sprites;

pub use entities::ExoSpaceEntity;
pub use environments::ExoSpaceEnvironment;
pub use events::ExoSpaceEvent;
