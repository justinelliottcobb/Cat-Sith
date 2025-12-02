//! ExoSpace event builders
//!
//! Provides convenient builders for creating ExoSpace-specific scene events.

use catsith_core::scene::SceneEvent;

/// Builder for ExoSpace events
pub struct ExoSpaceEvent;

impl ExoSpaceEvent {
    /// Create an explosion at the given position
    pub fn explosion(position: [f64; 2], radius: f64) -> ExplosionBuilder {
        ExplosionBuilder::new(position, radius)
    }

    /// Create a weapon beam between two points
    pub fn beam(start: [f64; 2], end: [f64; 2]) -> BeamBuilder {
        BeamBuilder::new(start, end)
    }

    /// Create a screen flash
    pub fn flash(intensity: f64) -> FlashBuilder {
        FlashBuilder::new(intensity)
    }

    /// Create a screen shake
    pub fn shake(intensity: f64, duration: f64) -> SceneEvent {
        SceneEvent::Shake {
            intensity,
            duration,
        }
    }

    /// Create a spark particle effect
    pub fn spark(position: [f64; 2], velocity: [f64; 2]) -> SceneEvent {
        SceneEvent::Particle {
            position,
            velocity,
            particle_type: "spark".to_string(),
        }
    }

    /// Create a debris particle effect
    pub fn debris(position: [f64; 2], velocity: [f64; 2]) -> SceneEvent {
        SceneEvent::Particle {
            position,
            velocity,
            particle_type: "debris".to_string(),
        }
    }

    /// Create an energy particle effect
    pub fn energy(position: [f64; 2], velocity: [f64; 2]) -> SceneEvent {
        SceneEvent::Particle {
            position,
            velocity,
            particle_type: "energy".to_string(),
        }
    }
}

/// Builder for explosion events
pub struct ExplosionBuilder {
    position: [f64; 2],
    radius: f64,
    intensity: f64,
    age: f64,
}

impl ExplosionBuilder {
    fn new(position: [f64; 2], radius: f64) -> Self {
        Self {
            position,
            radius,
            intensity: 1.0,
            age: 0.0,
        }
    }

    /// Set explosion intensity (0.0 - 1.0)
    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set explosion age (0.0 = just started, 1.0 = fading out)
    pub fn with_age(mut self, age: f64) -> Self {
        self.age = age.clamp(0.0, 1.0);
        self
    }

    /// Build the event
    pub fn build(self) -> SceneEvent {
        SceneEvent::Explosion {
            position: self.position,
            radius: self.radius,
            intensity: self.intensity,
            age: self.age,
        }
    }
}

/// Builder for beam events
pub struct BeamBuilder {
    start: [f64; 2],
    end: [f64; 2],
    intensity: f64,
    color: Option<[u8; 3]>,
}

impl BeamBuilder {
    fn new(start: [f64; 2], end: [f64; 2]) -> Self {
        Self {
            start,
            end,
            intensity: 1.0,
            color: None,
        }
    }

    /// Set beam intensity
    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set beam color
    pub fn with_color(mut self, color: [u8; 3]) -> Self {
        self.color = Some(color);
        self
    }

    /// Use laser color (green)
    pub fn laser(mut self) -> Self {
        self.color = Some([0, 255, 0]);
        self
    }

    /// Use plasma color (purple)
    pub fn plasma(mut self) -> Self {
        self.color = Some([128, 0, 255]);
        self
    }

    /// Build the event
    pub fn build(self) -> SceneEvent {
        SceneEvent::Beam {
            start: self.start,
            end: self.end,
            intensity: self.intensity,
            color: self.color,
        }
    }
}

/// Builder for flash events
pub struct FlashBuilder {
    intensity: f64,
    color: Option<[u8; 3]>,
}

impl FlashBuilder {
    fn new(intensity: f64) -> Self {
        Self {
            intensity: intensity.clamp(0.0, 1.0),
            color: None,
        }
    }

    /// Set flash color
    pub fn with_color(mut self, color: [u8; 3]) -> Self {
        self.color = Some(color);
        self
    }

    /// White flash (explosion)
    pub fn white(mut self) -> Self {
        self.color = Some([255, 255, 255]);
        self
    }

    /// Red flash (damage)
    pub fn red(mut self) -> Self {
        self.color = Some([255, 0, 0]);
        self
    }

    /// Build the event
    pub fn build(self) -> SceneEvent {
        SceneEvent::Flash {
            intensity: self.intensity,
            color: self.color,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explosion_builder() {
        let event = ExoSpaceEvent::explosion([100.0, 100.0], 50.0)
            .with_intensity(0.8)
            .build();

        match event {
            SceneEvent::Explosion {
                position,
                radius,
                intensity,
                ..
            } => {
                assert_eq!(position, [100.0, 100.0]);
                assert_eq!(radius, 50.0);
                assert!((intensity - 0.8).abs() < 0.001);
            }
            _ => panic!("Expected Explosion event"),
        }
    }

    #[test]
    fn test_beam_builder() {
        let event = ExoSpaceEvent::beam([0.0, 0.0], [100.0, 100.0])
            .laser()
            .build();

        match event {
            SceneEvent::Beam { color, .. } => {
                assert_eq!(color, Some([0, 255, 0]));
            }
            _ => panic!("Expected Beam event"),
        }
    }
}
