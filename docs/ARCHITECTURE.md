# CatSith Architecture

## Overview

CatSith is a semantic rendering frontend that decouples game state from visual representation. It consumes semantic scene descriptions and produces visual output across multiple backends.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Game Server                               │
│    (ExoSpace, Tito's Horse, or any game producing scenes)       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Scene Messages (JSON/Bitcode)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CatSith API Layer                           │
│   • Scene reception and validation                               │
│   • Identity caching                                             │
│   • Protocol handling                                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestration                        │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
│   │ Stage 1 │ → │ Stage 2 │ → │ Stage 3 │ → │ Stage N │        │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
│        ↑             ↑             ↑             ↑               │
│        └─────────────┴─────────────┴─────────────┘               │
│                         LoRA Stack                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Rendering Backends                          │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Terminal │  │  Raster  │  │  Neural  │  │ (Future) │        │
│  │   ASCII  │  │ Sprites  │  │Inference │  │          │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                        Visual Output
              (Terminal, Images, Tensor Data)
```

## Crate Dependencies

```
catsith-cli
    ├── catsith-backend-terminal
    │       ├── catsith-pipeline
    │       │       └── catsith-core
    │       └── catsith-core
    ├── catsith-backend-raster
    │       ├── catsith-pipeline
    │       └── catsith-core
    ├── catsith-lora
    │       └── catsith-core
    └── catsith-api
            └── catsith-core
```

## Data Flow

### Scene Processing

1. **Scene Creation** (Game Server)
   - Game produces semantic scene with entities, viewport, environment
   - No rendering hints - only semantic meaning

2. **Scene Reception** (API Layer)
   - Deserialize scene from wire format
   - Validate structure
   - Cache entity identities

3. **Pipeline Processing**
   - Create RenderContext with scene and player style
   - Pass through each pipeline stage
   - Stages may add intermediate data (embeddings, sprites)
   - Final stage produces RenderOutput

4. **Output Delivery**
   - Terminal: ANSI escape codes to stdout
   - Image: Pixel buffer to file/display
   - Tensor: Raw data for pipeline chaining

### Entity Identity

Entities have stable identities across frames for:
- Caching rendered sprites/embeddings
- Temporal coherence in animation
- Player-specific customization

```
EntityId → IdentityRef → EntityIdentity
                              │
                              ├── Display Name
                              ├── Semantic Description
                              ├── Color Scheme
                              └── LoRA References
```

## Backend Capabilities

| Backend  | Quality Tiers | GPU Required | Use Case |
|----------|---------------|--------------|----------|
| Terminal | Minimal       | No           | SSH, low-bandwidth |
| Raster   | Low-Medium    | No           | Traditional games |
| Neural   | Medium-Ultra  | Yes*         | AI-generated visuals |

*CPU fallback available at lower quality

## LoRA System

LoRAs customize visual style without modifying base models:

```
Player → Selects LoRAs → LoRA Stack → Applied to Model Weights
                              │
                              ├── Aesthetic LoRAs (overall style)
                              ├── Entity LoRAs (specific entities)
                              ├── Effects LoRAs (explosions, beams)
                              └── Color LoRAs (palette modifications)
```

## Quality Adaptation

CatSith automatically adapts quality based on:

1. **Hardware Capabilities**
   - CPU cores/features
   - GPU presence and VRAM
   - Available memory

2. **Frame Budget**
   - Target FPS
   - Actual frame times
   - Adaptive quality adjustment

3. **Player Preferences**
   - Preferred quality tier
   - Accessibility options
   - Style preferences
