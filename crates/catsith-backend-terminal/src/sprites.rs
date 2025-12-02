//! ASCII sprite generation
//!
//! Generates character-based sprites for entities using a registry-based
//! approach that allows domain crates to register their own sprite templates.

use catsith_core::entity::{EntityFlags, SemanticEntity};
use catsith_core::semantic::EntityKind;
use catsith_core::TerminalCell;
use std::collections::HashMap;

/// Sprite generator for terminal rendering
pub struct SpriteGenerator {
    /// Registered sprite templates by kind string
    templates: HashMap<String, SpriteTemplate>,
    /// Cached generated sprites
    cache: HashMap<SpriteKey, Sprite>,
    /// Default sprite for unknown entities
    default_sprite: Sprite,
}

/// A sprite template that can generate directional sprites
#[derive(Debug, Clone)]
pub struct SpriteTemplate {
    /// Name of this template
    pub name: String,
    /// Base characters for each direction (0-7, or single for non-directional)
    pub direction_chars: Vec<[[char; 3]; 3]>,
    /// Base color (RGB)
    pub base_color: [u8; 3],
    /// Whether this sprite uses directions
    pub directional: bool,
}

impl SpriteTemplate {
    /// Create a non-directional sprite template
    pub fn simple(name: impl Into<String>, chars: [[char; 3]; 3], color: [u8; 3]) -> Self {
        Self {
            name: name.into(),
            direction_chars: vec![chars],
            base_color: color,
            directional: false,
        }
    }

    /// Create a single-character centered sprite
    pub fn single(name: impl Into<String>, ch: char, color: [u8; 3]) -> Self {
        Self::simple(name, [[' ', ' ', ' '], [' ', ch, ' '], [' ', ' ', ' ']], color)
    }

    /// Create a directional sprite template (8 directions)
    pub fn directional(
        name: impl Into<String>,
        directions: Vec<[[char; 3]; 3]>,
        color: [u8; 3],
    ) -> Self {
        Self {
            name: name.into(),
            direction_chars: directions,
            base_color: color,
            directional: true,
        }
    }

    /// Generate a sprite for this template
    pub fn generate(&self, direction: u8, flags: EntityFlags) -> Sprite {
        let chars = if self.directional && !self.direction_chars.is_empty() {
            let idx = (direction as usize) % self.direction_chars.len();
            self.direction_chars[idx]
        } else {
            self.direction_chars.first().copied().unwrap_or([[' '; 3]; 3])
        };

        // Apply state-based color modifications
        let color = self.apply_state_color(flags);

        let mut sprite = Sprite::from_chars(chars, color);

        // Apply state-based effects
        if flags.contains(EntityFlags::THRUSTING) {
            self.apply_thrust_effect(&mut sprite, direction);
        }
        if flags.contains(EntityFlags::BOOSTING) {
            sprite = sprite.with_bold(true);
        }

        sprite
    }

    fn apply_state_color(&self, flags: EntityFlags) -> [u8; 3] {
        if flags.contains(EntityFlags::DAMAGED) {
            [0xC0, 0x40, 0x40] // Red when damaged
        } else if flags.contains(EntityFlags::SHIELDED) {
            [0x40, 0x80, 0xC0] // Blue when shielded
        } else if flags.contains(EntityFlags::CLOAKED) {
            [0x40, 0x40, 0x40] // Dark when cloaked
        } else {
            self.base_color
        }
    }

    fn apply_thrust_effect(&self, sprite: &mut Sprite, direction: u8) {
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
}

/// Key for sprite cache lookup
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct SpriteKey {
    kind: String,
    direction: u8,
    flags: u32,
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
    /// Create a new sprite generator with default templates
    pub fn new() -> Self {
        let mut generator = Self {
            templates: HashMap::new(),
            cache: HashMap::new(),
            default_sprite: Sprite::from_chars(
                [[' ', '?', ' '], ['[', '?', ']'], [' ', '?', ' ']],
                [0x80, 0x80, 0x80],
            ),
        };

        // Register default/fallback templates
        generator.register_defaults();

        generator
    }

    /// Register default sprite templates (ExoSpace-compatible for backward compat)
    fn register_defaults(&mut self) {
        // Ship directions (8 directions, clockwise from up)
        let ship_directions = vec![
            [[' ', '^', ' '], ['<', '#', '>'], ['/', ' ', '\\']], // 0: Up
            [[' ', '/', '/'], [' ', '#', '/'], ['/', ' ', ' ']],  // 1: Up-right
            [['\\', ' ', ' '], ['#', '#', '>'], ['/', ' ', ' ']], // 2: Right
            [['\\', ' ', ' '], [' ', '#', '\\'], [' ', '\\', '\\']], // 3: Down-right
            [['\\', ' ', '/'], ['<', '#', '>'], [' ', 'v', ' ']], // 4: Down
            [[' ', ' ', '/'], ['/', '#', ' '], ['/', '/', ' ']],  // 5: Down-left
            [[' ', ' ', '/'], ['<', '#', '#'], [' ', ' ', '\\']], // 6: Left
            [['\\', '\\', ' '], ['\\', '#', ' '], [' ', ' ', '\\']], // 7: Up-left
        ];

        // Register ship classes
        self.register(SpriteTemplate::directional(
            "fighter",
            ship_directions.clone(),
            [0x40, 0xC0, 0x80],
        ));
        self.register(SpriteTemplate::directional(
            "bomber",
            ship_directions.clone(),
            [0xC0, 0x80, 0x40],
        ));
        self.register(SpriteTemplate::directional(
            "scout",
            ship_directions.clone(),
            [0x40, 0x80, 0xC0],
        ));
        self.register(SpriteTemplate::directional(
            "cruiser",
            ship_directions.clone(),
            [0x80, 0x40, 0xC0],
        ));
        self.register(SpriteTemplate::directional(
            "carrier",
            ship_directions.clone(),
            [0xC0, 0xC0, 0x40],
        ));
        self.register(SpriteTemplate::directional(
            "station",
            ship_directions,
            [0x80, 0x80, 0x80],
        ));

        // Projectiles
        self.register(SpriteTemplate::single("bullet", '*', [0xFF, 0xFF, 0x00]));
        self.register(SpriteTemplate::single("bomb", 'o', [0xFF, 0x80, 0x00]));
        self.register(SpriteTemplate::single("beam", '|', [0x00, 0xFF, 0xFF]));
        self.register(SpriteTemplate::single("missile", '>', [0xFF, 0x00, 0x00]));
        self.register(SpriteTemplate::single("plasma", 'o', [0x80, 0x00, 0xFF]));
        self.register(SpriteTemplate::single("laser", '-', [0x00, 0xFF, 0x00]));

        // Environment objects
        self.register(SpriteTemplate::simple(
            "asteroid",
            [['/', '-', '\\'], ['|', '@', '|'], ['\\', '-', '/']],
            [0x80, 0x60, 0x40],
        ));
        self.register(SpriteTemplate::simple(
            "debris",
            [['.', ' ', ','], [' ', '*', ' '], ['\'', ' ', '.']],
            [0x60, 0x60, 0x60],
        ));
        self.register(SpriteTemplate::simple(
            "portal",
            [['(', '~', ')'], ['|', 'O', '|'], ['(', '~', ')']],
            [0x40, 0x80, 0xFF],
        ));
        self.register(SpriteTemplate::simple(
            "nebula",
            [['~', '~', '~'], ['~', '≈', '~'], ['~', '~', '~']],
            [0x80, 0x40, 0xA0],
        ));
        self.register(SpriteTemplate::simple(
            "star",
            [[' ', '*', ' '], ['*', '☼', '*'], [' ', '*', ' ']],
            [0xFF, 0xFF, 0x80],
        ));
        self.register(SpriteTemplate::simple(
            "planet",
            [[' ', '_', ' '], ['(', 'O', ')'], [' ', '-', ' ']],
            [0x40, 0x80, 0x40],
        ));
    }

    /// Register a sprite template
    pub fn register(&mut self, template: SpriteTemplate) {
        self.templates.insert(template.name.clone(), template);
        // Clear cache when templates change
        self.cache.clear();
    }

    /// Generate a sprite for an entity
    pub fn generate(&mut self, entity: &SemanticEntity) -> &Sprite {
        let kind = self.get_entity_kind(entity);
        let key = SpriteKey {
            kind: kind.clone(),
            direction: entity.direction_index(),
            flags: entity.state.flags.bits(),
        };

        if !self.cache.contains_key(&key) {
            let sprite = self.generate_sprite(&kind, entity);
            self.cache.insert(key.clone(), sprite);
        }

        self.cache.get(&key).unwrap()
    }

    /// Get the kind string for an entity
    fn get_entity_kind(&self, entity: &SemanticEntity) -> String {
        // Use the new kind field if set, otherwise fall back to sprite_key()
        if !entity.kind.is_empty() {
            entity.kind.clone()
        } else {
            // Extract base kind from sprite_key (remove direction suffix)
            let key = entity.sprite_key();
            if let Some(idx) = key.rfind('_') {
                // Check if suffix is a number (direction)
                if key[idx + 1..].parse::<u8>().is_ok() {
                    return key[..idx].to_string();
                }
            }
            key
        }
    }

    /// Generate a new sprite
    fn generate_sprite(&self, kind: &str, entity: &SemanticEntity) -> Sprite {
        if let Some(template) = self.templates.get(kind) {
            template.generate(entity.direction_index(), entity.state.flags)
        } else {
            // Try to find a generic category fallback
            let fallback = match entity.category {
                EntityKind::Vehicle => self.templates.get("fighter"),
                EntityKind::Projectile => self.templates.get("bullet"),
                EntityKind::Environment => self.templates.get("asteroid"),
                _ => None,
            };

            fallback
                .map(|t| t.generate(entity.direction_index(), entity.state.flags))
                .unwrap_or_else(|| self.default_sprite.clone())
        }
    }

    /// Clear the sprite cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get the number of registered templates
    pub fn template_count(&self) -> usize {
        self.templates.len()
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

        let entity = SemanticEntity::with_kind(EntityKind::Vehicle, "fighter", [0.0, 0.0]);

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
        let entity_up = SemanticEntity::with_kind(EntityKind::Vehicle, "fighter", [0.0, 0.0])
            .with_rotation(0.0);

        let entity_right = SemanticEntity::with_kind(EntityKind::Vehicle, "fighter", [0.0, 0.0])
            .with_rotation(std::f64::consts::FRAC_PI_2);

        let sprite_up = generator.generate(&entity_up).clone();
        let sprite_right = generator.generate(&entity_right);

        // Different directions should have different top-center characters
        assert_ne!(sprite_up.cells[0][1].char, sprite_right.cells[0][1].char);
    }

    #[test]
    fn test_damaged_color() {
        let mut generator = SpriteGenerator::new();

        let entity = SemanticEntity::with_kind(EntityKind::Vehicle, "fighter", [0.0, 0.0])
            .with_state(EntityState::default().with_flags(EntityFlags::DAMAGED));

        let sprite = generator.generate(&entity);

        // Damaged ships should be red
        assert_eq!(sprite.cells[1][1].fg, [0xC0, 0x40, 0x40]);
    }

    #[test]
    fn test_projectile_sprites() {
        let mut generator = SpriteGenerator::new();

        let bullet = SemanticEntity::with_kind(EntityKind::Projectile, "bullet", [0.0, 0.0]);

        let sprite = generator.generate(&bullet);
        assert_eq!(sprite.cells[1][1].char, '*');
    }

    #[test]
    fn test_sprite_cache() {
        let mut generator = SpriteGenerator::new();
        assert_eq!(generator.cache_size(), 0);

        let entity = SemanticEntity::with_kind(EntityKind::Vehicle, "fighter", [0.0, 0.0]);

        generator.generate(&entity);
        assert_eq!(generator.cache_size(), 1);

        // Same entity should hit cache
        generator.generate(&entity);
        assert_eq!(generator.cache_size(), 1);

        generator.clear_cache();
        assert_eq!(generator.cache_size(), 0);
    }

    #[test]
    fn test_custom_template() {
        let mut generator = SpriteGenerator::new();

        // Register a custom sprite
        generator.register(SpriteTemplate::simple(
            "dragon",
            [['/', '^', '\\'], ['<', 'D', '>'], ['/', 'v', '\\']],
            [0xFF, 0x40, 0x00],
        ));

        let entity = SemanticEntity::with_kind(EntityKind::Character, "dragon", [0.0, 0.0]);
        let sprite = generator.generate(&entity);

        assert_eq!(sprite.cells[1][1].char, 'D');
        assert_eq!(sprite.cells[1][1].fg, [0xFF, 0x40, 0x00]);
    }

    #[test]
    fn test_unknown_entity_fallback() {
        let mut generator = SpriteGenerator::new();

        // Unknown vehicle should fall back to fighter template
        let entity = SemanticEntity::with_kind(EntityKind::Vehicle, "unknown_ship", [0.0, 0.0]);
        let sprite = generator.generate(&entity);

        // Should get the fighter template (with '#' center)
        assert_eq!(sprite.cells[1][1].char, '#');
    }
}
