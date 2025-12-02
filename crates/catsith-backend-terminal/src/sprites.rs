//! ASCII sprite generation
//!
//! Generates character-based sprites for entities.

use catsith_core::TerminalCell;
use catsith_core::entity::{
    EntityFlags, EntityType, EnvironmentType, SemanticEntity, ShipClass, WeaponType,
};
use std::collections::HashMap;

/// Sprite generator for terminal rendering
pub struct SpriteGenerator {
    /// Cached sprites
    cache: HashMap<SpriteKey, Sprite>,
}

/// Key for sprite cache lookup
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct SpriteKey {
    entity_type: EntityTypeKey,
    direction: u8,
    flags: u32,
}

/// Simplified entity type for caching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum EntityTypeKey {
    Ship { class: u8 },
    Projectile { weapon: u8 },
    Environment { kind: u8 },
    Custom { hash: u64 },
}

/// A 3x3 terminal sprite
#[derive(Debug, Clone)]
pub struct Sprite {
    /// 3x3 grid of cells
    pub cells: [[TerminalCell; 3]; 3],
    /// Width of the sprite
    pub width: u32,
    /// Height of the sprite
    pub height: u32,
}

impl Sprite {
    /// Create a new empty sprite
    pub fn new() -> Self {
        Self {
            cells: [[TerminalCell::default(); 3]; 3],
            width: 3,
            height: 3,
        }
    }

    /// Create sprite from character grid
    pub fn from_chars(chars: [[char; 3]; 3], color: [u8; 3]) -> Self {
        let mut cells = [[TerminalCell::default(); 3]; 3];
        for (y, row) in chars.iter().enumerate() {
            for (x, &ch) in row.iter().enumerate() {
                cells[y][x] = TerminalCell::new(ch).with_fg(color);
            }
        }
        Self {
            cells,
            width: 3,
            height: 3,
        }
    }

    /// Get a cell
    pub fn get(&self, x: usize, y: usize) -> Option<&TerminalCell> {
        self.cells.get(y).and_then(|row| row.get(x))
    }

    /// Set a cell
    pub fn set(&mut self, x: usize, y: usize, cell: TerminalCell) {
        if y < 3 && x < 3 {
            self.cells[y][x] = cell;
        }
    }

    /// Apply a color to all non-space characters
    pub fn with_color(mut self, color: [u8; 3]) -> Self {
        for row in &mut self.cells {
            for cell in row {
                if cell.char != ' ' {
                    cell.fg = color;
                }
            }
        }
        self
    }

    /// Set bold on all characters
    pub fn with_bold(mut self, bold: bool) -> Self {
        for row in &mut self.cells {
            for cell in row {
                cell.bold = bold;
            }
        }
        self
    }
}

impl Default for Sprite {
    fn default() -> Self {
        Self::new()
    }
}

impl SpriteGenerator {
    /// Create a new sprite generator
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Generate a sprite for an entity
    pub fn generate(&mut self, entity: &SemanticEntity) -> &Sprite {
        let key = self.entity_to_key(entity);

        if !self.cache.contains_key(&key) {
            let sprite = self.generate_sprite(&key, entity);
            self.cache.insert(key.clone(), sprite);
        }

        self.cache.get(&key).unwrap()
    }

    /// Convert entity to cache key
    fn entity_to_key(&self, entity: &SemanticEntity) -> SpriteKey {
        let entity_type = match &entity.entity_type {
            EntityType::Ship { class, .. } => EntityTypeKey::Ship {
                class: *class as u8,
            },
            EntityType::Projectile { weapon_type, .. } => EntityTypeKey::Projectile {
                weapon: *weapon_type as u8,
            },
            EntityType::Environment { object_type } => EntityTypeKey::Environment {
                kind: *object_type as u8,
            },
            EntityType::Automaton { .. } | EntityType::Custom { .. } => EntityTypeKey::Custom {
                hash: 0, // Would compute actual hash in production
            },
        };

        SpriteKey {
            entity_type,
            direction: entity.direction_index(),
            flags: entity.state.flags.bits(),
        }
    }

    /// Generate a new sprite
    fn generate_sprite(&self, key: &SpriteKey, entity: &SemanticEntity) -> Sprite {
        match &entity.entity_type {
            EntityType::Ship { class, .. } => {
                self.generate_ship_sprite(*class, key.direction, entity.state.flags)
            }
            EntityType::Projectile { weapon_type, .. } => {
                self.generate_projectile_sprite(*weapon_type, key.direction)
            }
            EntityType::Environment { object_type } => {
                self.generate_environment_sprite(*object_type)
            }
            _ => self.generate_default_sprite(),
        }
    }

    /// Generate ship sprite with 8-direction support
    fn generate_ship_sprite(&self, class: ShipClass, direction: u8, flags: EntityFlags) -> Sprite {
        // 8 directions: 0=up, 1=up-right, 2=right, 3=down-right, 4=down, 5=down-left, 6=left, 7=up-left
        let chars = match direction {
            0 => [
                // Up
                [' ', '^', ' '],
                ['<', '#', '>'],
                ['/', ' ', '\\'],
            ],
            1 => [
                // Up-right
                [' ', '/', '/'],
                [' ', '#', '/'],
                ['/', ' ', ' '],
            ],
            2 => [
                // Right
                ['\\', ' ', ' '],
                ['#', '#', '>'],
                ['/', ' ', ' '],
            ],
            3 => [
                // Down-right
                ['\\', ' ', ' '],
                [' ', '#', '\\'],
                [' ', '\\', '\\'],
            ],
            4 => [
                // Down
                ['\\', ' ', '/'],
                ['<', '#', '>'],
                [' ', 'v', ' '],
            ],
            5 => [
                // Down-left
                [' ', ' ', '/'],
                ['/', '#', ' '],
                ['/', '/', ' '],
            ],
            6 => [
                // Left
                [' ', ' ', '/'],
                ['<', '#', '#'],
                [' ', ' ', '\\'],
            ],
            7 => [
                // Up-left
                ['\\', '\\', ' '],
                ['\\', '#', ' '],
                [' ', ' ', '\\'],
            ],
            _ => [[' '; 3]; 3],
        };

        // Color based on class
        let base_color = match class {
            ShipClass::Fighter => [0x40, 0xC0, 0x80], // Green
            ShipClass::Bomber => [0xC0, 0x80, 0x40],  // Orange
            ShipClass::Scout => [0x80, 0x80, 0xC0],   // Blue-gray
            ShipClass::Cruiser => [0xA0, 0xA0, 0xA0], // Gray
            ShipClass::Carrier => [0xC0, 0xC0, 0x80], // Yellow-gray
            ShipClass::Station => [0x80, 0x80, 0x80], // Dark gray
        };

        // Modify color based on state
        let color = if flags.contains(EntityFlags::DAMAGED) {
            [0xC0, 0x40, 0x40] // Red when damaged
        } else if flags.contains(EntityFlags::SHIELDED) {
            [0x40, 0x80, 0xC0] // Blue when shielded
        } else if flags.contains(EntityFlags::CLOAKED) {
            [0x40, 0x40, 0x40] // Dark when cloaked
        } else {
            base_color
        };

        let mut sprite = Sprite::from_chars(chars, color);

        // Add thrust effect
        if flags.contains(EntityFlags::THRUSTING) {
            // Add thrust indicator opposite to direction
            let thrust_pos = match direction {
                0 => Some((1, 2)), // Below center
                1 => Some((0, 2)), // Bottom-left
                2 => Some((0, 1)), // Left of center
                3 => Some((0, 0)), // Top-left
                4 => Some((1, 0)), // Above center
                5 => Some((2, 0)), // Top-right
                6 => Some((2, 1)), // Right of center
                7 => Some((2, 2)), // Bottom-right
                _ => None,
            };

            if let Some((x, y)) = thrust_pos {
                sprite.set(x, y, TerminalCell::new('*').with_fg([0xFF, 0x80, 0x00]));
            }
        }

        // Bold when boosting
        if flags.contains(EntityFlags::BOOSTING) {
            sprite = sprite.with_bold(true);
        }

        sprite
    }

    /// Generate projectile sprite
    fn generate_projectile_sprite(&self, weapon_type: WeaponType, direction: u8) -> Sprite {
        let (char_h, char_v, color) = match weapon_type {
            WeaponType::Bullet => ('-', '|', [0xFF, 0xFF, 0x00]),
            WeaponType::Bomb => ('o', 'o', [0xFF, 0x80, 0x00]),
            WeaponType::Beam => ('=', '║', [0x00, 0xFF, 0xFF]),
            WeaponType::Missile => ('>', '^', [0xFF, 0x40, 0x40]),
            WeaponType::Plasma => ('*', '*', [0xFF, 0x00, 0xFF]),
            WeaponType::Laser => ('.', '.', [0xFF, 0x00, 0x00]),
        };

        let ch = if direction == 0 || direction == 4 {
            char_v
        } else {
            char_h
        };

        let mut sprite = Sprite::new();
        sprite.set(1, 1, TerminalCell::new(ch).with_fg(color));

        sprite
    }

    /// Generate environment object sprite
    fn generate_environment_sprite(&self, object_type: EnvironmentType) -> Sprite {
        match object_type {
            EnvironmentType::Asteroid => Sprite::from_chars(
                [[' ', '.', ' '], ['.', '@', '.'], [' ', '.', ' ']],
                [0x80, 0x80, 0x80],
            ),
            EnvironmentType::Debris => Sprite::from_chars(
                [['.', ' ', '.'], [' ', '%', ' '], ['.', ' ', '.']],
                [0x60, 0x60, 0x60],
            ),
            EnvironmentType::Station => Sprite::from_chars(
                [['[', '=', ']'], ['|', '#', '|'], ['[', '=', ']']],
                [0xA0, 0xA0, 0xA0],
            ),
            EnvironmentType::Portal => Sprite::from_chars(
                [['(', '~', ')'], ['(', 'O', ')'], ['(', '~', ')']],
                [0x80, 0x00, 0xFF],
            ),
            EnvironmentType::Nebula => Sprite::from_chars(
                [['~', '~', '~'], ['~', '~', '~'], ['~', '~', '~']],
                [0x80, 0x40, 0xA0],
            ),
            EnvironmentType::Star => Sprite::from_chars(
                [[' ', '*', ' '], ['*', '☼', '*'], [' ', '*', ' ']],
                [0xFF, 0xFF, 0x80],
            ),
            EnvironmentType::Planet => Sprite::from_chars(
                [[' ', '_', ' '], ['(', 'O', ')'], [' ', '‾', ' ']],
                [0x40, 0x80, 0x40],
            ),
        }
    }

    /// Generate default/unknown entity sprite
    fn generate_default_sprite(&self) -> Sprite {
        Sprite::from_chars(
            [[' ', '?', ' '], ['[', '?', ']'], [' ', '?', ' ']],
            [0x80, 0x80, 0x80],
        )
    }

    /// Clear the sprite cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

impl Default for SpriteGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catsith_core::entity::EntityState;

    #[test]
    fn test_sprite_generation() {
        let mut generator = SpriteGenerator::new();

        let entity = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [0.0, 0.0],
        );

        let sprite = generator.generate(&entity);
        assert_eq!(sprite.width, 3);
        assert_eq!(sprite.height, 3);

        // Center should be '#'
        assert_eq!(sprite.cells[1][1].char, '#');
    }

    #[test]
    fn test_sprite_directions() {
        let mut generator = SpriteGenerator::new();

        // Test different directions produce different sprites
        let entity_up = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [0.0, 0.0],
        )
        .with_rotation(0.0);

        let entity_right = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [0.0, 0.0],
        )
        .with_rotation(std::f64::consts::FRAC_PI_2);

        let sprite_up = generator.generate(&entity_up).clone();
        let sprite_right = generator.generate(&entity_right);

        // Different directions should have different top-center characters
        assert_ne!(sprite_up.cells[0][1].char, sprite_right.cells[0][1].char);
    }

    #[test]
    fn test_damaged_color() {
        let mut generator = SpriteGenerator::new();

        let entity = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [0.0, 0.0],
        )
        .with_state(EntityState::default().with_flags(EntityFlags::DAMAGED));

        let sprite = generator.generate(&entity);

        // Damaged ships should be red
        assert_eq!(sprite.cells[1][1].fg, [0xC0, 0x40, 0x40]);
    }

    #[test]
    fn test_projectile_sprites() {
        let mut generator = SpriteGenerator::new();

        let bullet = SemanticEntity::new(
            EntityType::Projectile {
                weapon_type: WeaponType::Bullet,
                owner_id: catsith_core::EntityId::new(),
            },
            [0.0, 0.0],
        );

        let sprite = generator.generate(&bullet);
        assert!(sprite.cells[1][1].char == '-' || sprite.cells[1][1].char == '|');
    }

    #[test]
    fn test_sprite_cache() {
        let mut generator = SpriteGenerator::new();
        assert_eq!(generator.cache_size(), 0);

        let entity = SemanticEntity::new(
            EntityType::Ship {
                class: ShipClass::Fighter,
                owner_id: None,
            },
            [0.0, 0.0],
        );

        generator.generate(&entity);
        assert_eq!(generator.cache_size(), 1);

        // Same entity should hit cache
        generator.generate(&entity);
        assert_eq!(generator.cache_size(), 1);

        generator.clear_cache();
        assert_eq!(generator.cache_size(), 0);
    }
}
