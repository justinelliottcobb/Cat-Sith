//! ExoSpace sprite definitions
//!
//! This module contains the ASCII sprite definitions and color palettes
//! for ExoSpace entities. These can be registered with CatSith renderers.

use catsith_core::TerminalCell;

/// Color palette for ExoSpace entities
pub mod colors {
    /// Fighter ship color (green)
    pub const FIGHTER: [u8; 3] = [0x40, 0xC0, 0x80];
    /// Bomber ship color (orange)
    pub const BOMBER: [u8; 3] = [0xC0, 0x80, 0x40];
    /// Scout ship color (blue)
    pub const SCOUT: [u8; 3] = [0x40, 0x80, 0xC0];
    /// Cruiser ship color (purple)
    pub const CRUISER: [u8; 3] = [0x80, 0x40, 0xC0];
    /// Carrier ship color (yellow)
    pub const CARRIER: [u8; 3] = [0xC0, 0xC0, 0x40];
    /// Station color (gray)
    pub const STATION: [u8; 3] = [0x80, 0x80, 0x80];

    /// Bullet color (yellow)
    pub const BULLET: [u8; 3] = [0xFF, 0xFF, 0x00];
    /// Beam color (cyan)
    pub const BEAM: [u8; 3] = [0x00, 0xFF, 0xFF];
    /// Missile color (red)
    pub const MISSILE: [u8; 3] = [0xFF, 0x00, 0x00];
    /// Plasma color (purple)
    pub const PLASMA: [u8; 3] = [0x80, 0x00, 0xFF];
    /// Laser color (green)
    pub const LASER: [u8; 3] = [0x00, 0xFF, 0x00];

    /// Asteroid color (brown/gray)
    pub const ASTEROID: [u8; 3] = [0x80, 0x60, 0x40];
    /// Debris color (gray)
    pub const DEBRIS: [u8; 3] = [0x60, 0x60, 0x60];
    /// Portal color (blue glow)
    pub const PORTAL: [u8; 3] = [0x40, 0x80, 0xFF];

    /// Damage indicator color
    pub const DAMAGE: [u8; 3] = [0xC0, 0x40, 0x40];
    /// Shield indicator color
    pub const SHIELD: [u8; 3] = [0x40, 0x80, 0xC0];
    /// Thrust effect color
    pub const THRUST: [u8; 3] = [0xFF, 0x80, 0x00];
}

/// A 3x3 ASCII sprite
#[derive(Debug, Clone, Copy)]
pub struct Sprite {
    /// Character grid (3x3)
    pub cells: [[TerminalCell; 3]; 3],
}

impl Sprite {
    /// Create a sprite from character arrays
    pub fn from_chars(chars: [[char; 3]; 3], color: [u8; 3]) -> Self {
        let mut cells = [[TerminalCell::default(); 3]; 3];
        for (y, row) in chars.iter().enumerate() {
            for (x, &ch) in row.iter().enumerate() {
                cells[y][x] = TerminalCell::new(ch).with_fg(color);
            }
        }
        Self { cells }
    }

    /// Create a single-character sprite centered
    pub fn single(ch: char, color: [u8; 3]) -> Self {
        Self::from_chars([[' ', ' ', ' '], [' ', ch, ' '], [' ', ' ', ' ']], color)
    }
}

/// ASCII sprite templates for ships by direction (0-7, clockwise from up)
pub fn ship_sprite_chars(direction: u8) -> [[char; 3]; 3] {
    match direction % 8 {
        0 => [[' ', '^', ' '], ['<', '#', '>'], ['/', ' ', '\\']],  // Up
        1 => [[' ', '/', ' '], [' ', '#', '>'], [' ', '\\', ' ']],  // Up-right
        2 => [['\\', ' ', ' '], ['#', '#', '>'], ['/', ' ', ' ']],  // Right
        3 => [[' ', '\\', ' '], [' ', '#', '>'], [' ', '/', ' ']],  // Down-right
        4 => [['\\', ' ', '/'], ['<', '#', '>'], [' ', 'v', ' ']],  // Down
        5 => [[' ', '/', ' '], ['<', '#', ' '], [' ', '\\', ' ']],  // Down-left
        6 => [[' ', ' ', '/'], ['<', '#', '#'], [' ', ' ', '\\']],  // Left
        7 => [[' ', '\\', ' '], ['<', '#', ' '], [' ', '/', ' ']],  // Up-left
        _ => [[' ', '^', ' '], ['<', '#', '>'], ['/', ' ', '\\']],
    }
}

/// ASCII sprite for asteroids
pub fn asteroid_sprite() -> [[char; 3]; 3] {
    [['/', '-', '\\'], ['|', '@', '|'], ['\\', '-', '/']]
}

/// ASCII sprite for debris
pub fn debris_sprite() -> [[char; 3]; 3] {
    [['.', ' ', ','], [' ', '*', ' '], ['\'', ' ', '.']]
}

/// ASCII sprite for portals
pub fn portal_sprite() -> [[char; 3]; 3] {
    [['(', '~', ')'], ['|', 'O', '|'], ['(', '~', ')']]
}

/// ASCII sprite for stations
pub fn station_sprite() -> [[char; 3]; 3] {
    [['+', '-', '+'], ['|', '#', '|'], ['+', '-', '+']]
}

/// Get the sprite for a given entity kind and direction
pub fn get_sprite(kind: &str, direction: u8) -> Sprite {
    match kind {
        "fighter" => Sprite::from_chars(ship_sprite_chars(direction), colors::FIGHTER),
        "bomber" => Sprite::from_chars(ship_sprite_chars(direction), colors::BOMBER),
        "scout" => Sprite::from_chars(ship_sprite_chars(direction), colors::SCOUT),
        "cruiser" => Sprite::from_chars(ship_sprite_chars(direction), colors::CRUISER),
        "carrier" => Sprite::from_chars(ship_sprite_chars(direction), colors::CARRIER),
        "station" => Sprite::from_chars(station_sprite(), colors::STATION),
        "asteroid" => Sprite::from_chars(asteroid_sprite(), colors::ASTEROID),
        "debris" => Sprite::from_chars(debris_sprite(), colors::DEBRIS),
        "portal" => Sprite::from_chars(portal_sprite(), colors::PORTAL),
        "bullet" => Sprite::single('*', colors::BULLET),
        "beam" => Sprite::single('|', colors::BEAM),
        "missile" => Sprite::single('>', colors::MISSILE),
        "plasma" => Sprite::single('o', colors::PLASMA),
        "laser" => Sprite::single('-', colors::LASER),
        _ => Sprite::single('?', [0x80, 0x80, 0x80]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ship_sprite() {
        let sprite = get_sprite("fighter", 0);
        assert_eq!(sprite.cells[0][1].char, '^');
        assert_eq!(sprite.cells[1][1].char, '#');
    }

    #[test]
    fn test_projectile_sprite() {
        let sprite = get_sprite("bullet", 0);
        assert_eq!(sprite.cells[1][1].char, '*');
    }

    #[test]
    fn test_environment_sprite() {
        let sprite = get_sprite("asteroid", 0);
        assert_eq!(sprite.cells[1][1].char, '@');
    }

    #[test]
    fn test_unknown_sprite() {
        let sprite = get_sprite("unknown_entity", 0);
        assert_eq!(sprite.cells[1][1].char, '?');
    }
}
