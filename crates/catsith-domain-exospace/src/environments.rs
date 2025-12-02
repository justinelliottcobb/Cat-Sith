//! ExoSpace environment builders
//!
//! Provides convenient builders for creating ExoSpace-specific environments.

use catsith_core::scene::{Environment, LightingMood};
use catsith_core::semantic::AmbianceKind;

/// Builder for ExoSpace environments
pub struct ExoSpaceEnvironment;

impl ExoSpaceEnvironment {
    /// Create a deep space environment
    pub fn space() -> Environment {
        Environment {
            ambiance: catsith_core::scene::Ambiance::Void,
            descriptors: vec!["deep space".to_string(), "star field".to_string()],
            lighting: LightingMood::Cold,
            background_color: Some([5, 5, 15]),
            visibility: None,
        }
    }

    /// Create a nebula environment
    pub fn nebula() -> Environment {
        Environment {
            ambiance: catsith_core::scene::Ambiance::Nebula,
            descriptors: vec![
                "nebula field".to_string(),
                "colorful gas clouds".to_string(),
            ],
            lighting: LightingMood::Ethereal,
            background_color: Some([20, 10, 30]),
            visibility: Some(500.0),
        }
    }

    /// Create an asteroid field environment
    pub fn asteroid_field() -> Environment {
        Environment {
            ambiance: catsith_core::scene::Ambiance::Asteroid,
            descriptors: vec!["asteroid field".to_string(), "rocky debris".to_string()],
            lighting: LightingMood::Neutral,
            background_color: Some([10, 10, 15]),
            visibility: Some(800.0),
        }
    }

    /// Create a station interior environment
    pub fn station_interior() -> Environment {
        Environment {
            ambiance: catsith_core::scene::Ambiance::Station,
            descriptors: vec![
                "space station".to_string(),
                "industrial interior".to_string(),
            ],
            lighting: LightingMood::Harsh,
            background_color: Some([30, 30, 35]),
            visibility: Some(200.0),
        }
    }

    /// Create an atmospheric environment (planetary)
    pub fn atmosphere() -> Environment {
        Environment {
            ambiance: catsith_core::scene::Ambiance::Atmosphere,
            descriptors: vec![
                "planetary atmosphere".to_string(),
                "cloudy sky".to_string(),
            ],
            lighting: LightingMood::Neutral,
            background_color: Some([100, 150, 200]),
            visibility: Some(1000.0),
        }
    }
}

/// Map ExoSpace environments to generic AmbianceKind
pub fn to_ambiance_kind(env: &Environment) -> AmbianceKind {
    match env.ambiance {
        catsith_core::scene::Ambiance::Void => AmbianceKind::Empty,
        catsith_core::scene::Ambiance::Nebula => AmbianceKind::Dense,
        catsith_core::scene::Ambiance::Asteroid => AmbianceKind::Natural,
        catsith_core::scene::Ambiance::Station => AmbianceKind::Constructed,
        catsith_core::scene::Ambiance::Atmosphere => AmbianceKind::Natural,
        catsith_core::scene::Ambiance::Abstract => AmbianceKind::Abstract,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_environment() {
        let env = ExoSpaceEnvironment::space();
        assert_eq!(env.ambiance, catsith_core::scene::Ambiance::Void);
        assert!(!env.descriptors.is_empty());
    }

    #[test]
    fn test_ambiance_mapping() {
        let env = ExoSpaceEnvironment::nebula();
        assert_eq!(to_ambiance_kind(&env), AmbianceKind::Dense);
    }
}
