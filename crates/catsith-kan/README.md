# CatSith KAN - Kolmogorov-Arnold Networks for Sprite Generation

GPU-accelerated KAN implementation tailored for generating pixel art sprites from compact latent codes.

## Overview

Unlike traditional neural networks (MLPs) that use fixed activation functions with learnable weights, KANs flip this paradigm:

- **MLPs**: Fixed activations (ReLU, sigmoid) on nodes → Learnable scalar weights on edges
- **KANs**: Learnable univariate functions (B-splines) on edges → No fixed activations

This makes KANs particularly effective for:
- Smooth, continuous transformations (perfect for color gradients)
- Interpretable learned functions (you can visualize what each edge learned)
- Parameter-efficient representations of smooth functions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SpriteKAN                               │
├─────────────────────────────────────────────────────────────┤
│  Input: [latent_code₁...latent_codeₙ, x, y]                 │
│         └──────────┬──────────┘  └──┬──┘                    │
│              N-dim latent     normalized coords              │
│                              (-1 to 1)                       │
├─────────────────────────────────────────────────────────────┤
│                    KAN Layers                                │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Layer 1 │───▶│ Layer 2 │───▶│ Layer 3 │                  │
│  │ N+2→32  │    │  32→32  │    │  32→4   │                  │
│  └─────────┘    └─────────┘    └─────────┘                  │
│       │              │              │                        │
│       ▼              ▼              ▼                        │
│  [projection → univariate functions → sum]                   │
├─────────────────────────────────────────────────────────────┤
│  Output: [r, g, b, a] (after sigmoid)                       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```rust
use catsith_kan::{GpuContext, SpriteKAN, SpriteLatent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU
    let gpu = GpuContext::new().await?;

    // Create a sprite generator
    // - 16-dim latent code
    // - Two hidden layers of 32 units
    // - 8 univariate functions per layer
    // - 32x32 output sprites
    let sprite_kan = SpriteKAN::new(
        gpu,
        16,           // latent_dim
        &[32, 32],    // hidden layers
        &[8, 8, 8],   // functions per layer
        32, 32,       // sprite dimensions
    )?;

    // Generate a sprite from a random latent code
    let latent = SpriteLatent::random(16);
    let pixels = sprite_kan.generate(&latent)?; // Returns RGBA bytes

    // Interpolate between two sprites
    let latent_a = SpriteLatent::random(16);
    let latent_b = SpriteLatent::random(16);
    let midpoint = latent_a.slerp(&latent_b, 0.5);
    let interpolated = sprite_kan.generate(&midpoint)?;

    Ok(())
}
```

## Module Structure

```
catsith-kan/
├── src/
│   ├── lib.rs              # Public API and re-exports
│   ├── bspline.rs          # B-spline basis functions (CPU + GPU)
│   ├── univariate.rs       # Learnable univariate functions
│   ├── kan_layer.rs        # Single KAN layer implementation
│   ├── kan.rs              # Multi-layer KAN network
│   ├── sprite.rs           # SpriteKAN - sprite-specific architecture
│   ├── gpu.rs              # Shared WebGPU context
│   ├── error.rs            # Error types
│   └── shaders/
│       ├── bspline.wgsl        # B-spline evaluation
│       ├── univariate.wgsl     # Function evaluation
│       ├── weight_update.wgsl  # Gradient updates
│       ├── kan_layer.wgsl      # Layer forward pass
│       └── sprite_generate.wgsl # Batch pixel generation
└── Cargo.toml
```

## Development Roadmap

See [ROADMAP.md](./ROADMAP.md) for the complete development plan.

### Current Status: Phase 1 (Foundation) - Complete

- [x] B-spline basis functions (CPU + GPU verified)
- [x] Univariate function evaluation
- [x] KAN layer forward pass
- [x] Multi-layer network structure
- [x] SpriteKAN architecture
- [x] Basic training loop
- [x] WebGPU compute shaders

### Next Priority: Phase 2 (Training Infrastructure)

- [ ] Complete backward pass with proper gradient flow
- [ ] Model serialization (save/load weights)
- [ ] Training examples and validation

## References

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) - Original paper
- [kan-gpu](https://github.com/...) - Original Rust/WebGPU implementation this was forked from
- B-spline theory: Cox-de Boor recursion formula

## License

MIT OR Apache-2.0 (same as parent CatSith project)
