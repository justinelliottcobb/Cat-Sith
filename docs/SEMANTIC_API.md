# CatSith Semantic API

## Overview

The Semantic API defines how game servers describe scenes to CatSith. The key principle is **semantic truth**: describe WHAT things are, not HOW they should look.

## Scene Structure

```rust
pub struct Scene {
    pub frame_id: u64,           // Unique frame identifier
    pub timestamp: f64,          // Scene time for temporal coherence
    pub viewport: Viewport,      // Camera/viewport
    pub entities: Vec<SemanticEntity>,
    pub environment: Environment,
    pub events: Vec<SceneEvent>,
    pub identity_refs: HashMap<EntityId, IdentityRef>,
}
```

### Viewport

```rust
pub struct Viewport {
    pub center: [f64; 2],        // World-space center
    pub extent: [f64; 2],        // Visible area (width, height)
    pub rotation: f64,           // Camera rotation (radians)
    pub focus: Option<EntityId>, // Entity camera follows
    pub zoom: f64,               // Zoom level (1.0 = normal)
}
```

## Entity Types

### Ships

```rust
EntityType::Ship {
    class: ShipClass,           // Fighter, Bomber, Scout, etc.
    owner_id: Option<Uuid>,     // Links to player identity
}
```

Ship classes convey combat role, not visual appearance.

### Projectiles

```rust
EntityType::Projectile {
    weapon_type: WeaponType,    // Bullet, Missile, Beam, etc.
    owner_id: EntityId,         // Who fired it
}
```

### Environment

```rust
EntityType::Environment {
    object_type: EnvironmentType, // Asteroid, Station, Portal, etc.
}
```

## Entity State

Entities have normalized state values and flags:

```rust
pub struct EntityState {
    pub health: Option<f32>,    // 0.0 - 1.0
    pub energy: Option<f32>,    // 0.0 - 1.0
    pub shield: Option<f32>,    // 0.0 - 1.0
    pub flags: EntityFlags,     // Bitflags
}

bitflags! {
    pub struct EntityFlags: u32 {
        const THRUSTING = 0b0001;
        const FIRING    = 0b0010;
        const DAMAGED   = 0b0100;
        const SHIELDED  = 0b1000;
        const CLOAKED   = 0b10000;
        const BOOSTING  = 0b100000;
    }
}
```

## Environment Settings

```rust
pub struct Environment {
    pub ambiance: Ambiance,      // Void, Nebula, Asteroid, etc.
    pub descriptors: Vec<String>, // "deep space", "colorful nebula"
    pub lighting: LightingMood,   // Neutral, Warm, Cold, Dramatic
    pub background_color: Option<[u8; 3]>,
    pub visibility: Option<f64>,  // Fog distance
}
```

## Scene Events

Dynamic effects are represented as events:

```rust
pub enum SceneEvent {
    Explosion {
        position: [f64; 2],
        radius: f64,
        intensity: f64,
        age: f64,               // 0.0 = new, 1.0 = fading
    },
    Beam {
        start: [f64; 2],
        end: [f64; 2],
        intensity: f64,
        color: Option<[u8; 3]>,
    },
    Particle { ... },
    Flash { ... },
    Shake { ... },
}
```

## Entity Identity

For consistent rendering, entities can have associated identities:

```rust
pub struct EntityIdentity {
    pub hash: [u8; 32],         // Content hash for caching
    pub name: Option<String>,   // Display name
    pub description: Option<String>, // For neural rendering
    pub colors: Option<ColorScheme>,
    pub lora_refs: Vec<LoraRef>,
}
```

The description field is used by neural renderers:
```
"Crimson fighter with gold trim, battle-scarred hull"
```

## Protocol Messages

### Scene Message

```rust
pub struct SceneMessage {
    pub scene: Scene,
}
```

### Identity Message

Sent separately for caching:

```rust
pub struct IdentityMessage {
    pub identities: HashMap<EntityId, EntityIdentity>,
}
```

## Best Practices

### DO:
- Use semantic archetypes: "aggressive", "damaged", "stealthy"
- Provide action context: "pursuing", "fleeing", "patrolling"
- Use normalized values (0.0-1.0) for health, energy
- Include velocity for motion prediction
- Set meaningful timestamps for animation

### DON'T:
- Specify colors in scene data
- Include sprite names or image references
- Describe visual details (leave that to styles)
- Hard-code render-specific values
- Send identity data every frame (cache it)

## Example Scene

```json
{
  "frame_id": 12345,
  "timestamp": 205.5,
  "viewport": {
    "center": [1000.0, 500.0],
    "extent": [800.0, 600.0],
    "zoom": 1.0
  },
  "entities": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "entity_type": {
        "Ship": {
          "class": "Fighter",
          "owner_id": "player-123"
        }
      },
      "position": [1000.0, 500.0],
      "velocity": [50.0, 0.0],
      "rotation": 0.0,
      "state": {
        "health": 0.85,
        "energy": 0.6,
        "flags": 1
      },
      "archetype": "aggressive",
      "action": "pursuing"
    }
  ],
  "environment": {
    "ambiance": "Void",
    "descriptors": ["deep space", "star field"],
    "lighting": "Cold"
  },
  "events": []
}
```
