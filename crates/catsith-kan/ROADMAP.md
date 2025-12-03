# CatSith KAN Development Roadmap

This document outlines the development plan for making `catsith-kan` production-ready for sprite generation in CatSith.

## Phase 1: Foundation (Complete)

**Status: ✅ Done**

Core infrastructure ported from `kan-gpu` and adapted for sprite generation.

### Completed

- [x] B-spline basis function implementation
  - Cox-de Boor recursion (numerically stable)
  - CPU and GPU implementations verified to match
  - Partition of unity property validated

- [x] Univariate function structure
  - Learnable weights for B-spline coefficients
  - Forward evaluation (CPU + GPU)
  - Basic gradient updates

- [x] KAN layer implementation
  - Input projection matrix
  - Multiple univariate functions per layer
  - Forward pass working

- [x] Multi-layer KAN network
  - Layer chaining
  - Basic backpropagation (simplified)
  - MSE loss computation

- [x] SpriteKAN architecture
  - Latent + coordinate input design
  - RGBA output with sigmoid activation
  - Resolution-independent generation

- [x] GPU infrastructure
  - Shared `GpuContext` to avoid redundant initialization
  - WebGPU compute shaders for all operations
  - Async/await pattern for GPU operations

---

## Phase 2: Training Infrastructure

**Status: 🚧 In Progress**

**Goal:** Enable effective training on sprite datasets.

### 2.1 Complete Backward Pass

**Priority: High**

The current backward pass uses a simplified gradient propagation that doesn't properly apply the chain rule through univariate functions.

Tasks:
- [ ] Implement proper derivative computation for B-spline basis functions
- [ ] Store intermediate activations during forward pass for backprop
- [ ] Compute gradients through univariate functions (∂f/∂x = Σ wᵢ * B'ᵢ(x))
- [ ] Propagate gradients correctly between layers
- [ ] Add gradient clipping to prevent exploding gradients
- [ ] Implement gradient accumulation for batch training

**Files affected:** `kan_layer.rs`, `kan.rs`, `univariate.rs`

**Estimated effort:** 2-3 days

### 2.2 Model Serialization

**Priority: High**

Save and load trained models for reuse.

Tasks:
- [ ] Define serialization format for KAN weights
  - B-spline knot vectors
  - Univariate function weights
  - Projection matrices
  - Network architecture metadata
- [ ] Implement `save()` and `load()` methods on `KAN` and `SpriteKAN`
- [ ] Use `serde` + `bincode` or custom binary format
- [ ] Version the format for forward compatibility
- [ ] Add optional compression (zstd)

**Files affected:** New `src/serialization.rs`, `kan.rs`, `sprite.rs`

**Estimated effort:** 1-2 days

### 2.3 Training Utilities

**Priority: Medium**

Make training practical and observable.

Tasks:
- [ ] Learning rate schedulers (cosine annealing, step decay)
- [ ] Early stopping based on validation loss
- [ ] Training progress logging/callbacks
- [ ] Checkpoint saving during training
- [ ] Loss history tracking
- [ ] Batch training with data loader pattern

**Files affected:** New `src/training.rs`

**Estimated effort:** 2 days

---

## Phase 3: GPU Optimization

**Status: 📋 Planned**

**Goal:** Full GPU acceleration for training and inference.

### 3.1 Batched Sprite Generation

**Priority: High**

Currently, sprite generation evaluates one pixel at a time on CPU. This is extremely slow.

Tasks:
- [ ] Implement full KAN forward pass in WGSL shader
  - Embed B-spline evaluation in layer shader
  - Fuse all layers into single dispatch
- [ ] Generate all pixels in parallel (one thread per pixel)
- [ ] Batch multiple sprites in single GPU dispatch
- [ ] Profile and optimize memory access patterns

**Files affected:** `shaders/sprite_generate.wgsl`, `sprite.rs`

**Estimated effort:** 3-4 days

### 3.2 GPU-Accelerated Training

**Priority: Medium**

Move training entirely to GPU.

Tasks:
- [ ] Implement backward pass in WGSL
- [ ] GPU-based gradient accumulation
- [ ] Atomic weight updates or reduction patterns
- [ ] Compare GPU vs CPU training performance

**Files affected:** New shaders, `training.rs`

**Estimated effort:** 3-4 days

### 3.3 Memory Optimization

**Priority: Low**

Reduce GPU memory footprint for larger models.

Tasks:
- [ ] Buffer pooling/reuse instead of per-call allocation
- [ ] Gradient checkpointing for memory-bound training
- [ ] FP16 support for weights and activations
- [ ] Profile memory usage with various model sizes

**Estimated effort:** 2 days

---

## Phase 4: Training Data & Experiments

**Status: 📋 Planned**

**Goal:** Train actual sprite generators and validate the approach.

### 4.1 Dataset Preparation

Tasks:
- [ ] Create sprite dataset format specification
  - Image files (PNG, 32x32 or 64x64)
  - Metadata JSON (entity type, state, colors)
  - Train/validation/test splits
- [ ] Build dataset loader
- [ ] Implement data augmentation
  - Color jitter
  - Rotation (90° increments for pixel art)
  - Horizontal flip
- [ ] Create sample datasets for testing
  - Simple geometric shapes
  - ExoSpace ship silhouettes

**Estimated effort:** 2-3 days

### 4.2 Baseline Training Experiments

Tasks:
- [ ] Train on simple functions (sin, cos) to validate training works
- [ ] Train on single sprites to verify reconstruction
- [ ] Train on sprite categories (ships, asteroids, effects)
- [ ] Measure and log:
  - Training loss curves
  - Validation loss
  - Visual quality (manual inspection)
  - Generation speed

**Estimated effort:** 3-5 days (includes iteration)

### 4.3 Latent Space Analysis

Tasks:
- [ ] Visualize latent space with t-SNE/UMAP
- [ ] Test interpolation quality between sprites
- [ ] Identify semantic dimensions in latent space
- [ ] Document optimal latent dimensions for different sprite types

**Estimated effort:** 2 days

---

## Phase 5: Advanced Features

**Status: 📋 Planned (Future)**

**Goal:** Production-quality sprite generation.

### 5.1 Conditional Generation

Generate sprites based on semantic attributes.

Tasks:
- [ ] Add condition embedding input to SpriteKAN
- [ ] Train with entity type labels
- [ ] Train with state flags (damaged, firing, etc.)
- [ ] Support continuous attributes (health %, rotation)

### 5.2 Temporal Coherence

Generate consistent animation frames.

Tasks:
- [ ] Add previous-frame latent as input
- [ ] Implement frame-to-frame consistency loss
- [ ] Test on animation sequences

### 5.3 Style Transfer

Apply different visual styles to same semantic content.

Tasks:
- [ ] Separate content and style in latent space
- [ ] Train style encoder from example images
- [ ] Implement style mixing at generation time

### 5.4 Sparsity and Pruning

KAN-specific optimization: prune unimportant edges.

Tasks:
- [ ] Implement L1 regularization on B-spline coefficients
- [ ] Add pruning based on coefficient magnitude
- [ ] Measure model size reduction vs quality tradeoff
- [ ] Implement symbolic regression for interpretability

---

## Phase 6: Integration

**Status: 📋 Planned (Future)**

**Goal:** Full integration with CatSith rendering pipeline.

### 6.1 Neural Backend Integration

Tasks:
- [ ] Complete `KanSpriteGenerator` in neural backend
- [ ] Implement sprite caching with LRU eviction
- [ ] Add fallback to raster sprites when KAN unavailable
- [ ] Profile rendering performance

### 6.2 Entity-Aware Generation

Tasks:
- [ ] Map `SemanticEntity` properties to latent codes
- [ ] Generate state-dependent sprites (health, damage)
- [ ] Support entity archetype variations

### 6.3 Real-Time Animation

Tasks:
- [ ] Generate animation frames on-demand
- [ ] Implement predictive generation (pre-generate next frames)
- [ ] Profile frame generation latency
- [ ] Target: <16ms for 60fps capability

---

## Milestones & Dependencies

```
Phase 1 (Foundation) ─────────────────────────────► COMPLETE
         │
         ▼
Phase 2 (Training) ───────────────────────────────► IN PROGRESS
         │
         ├──► Phase 3 (GPU Optimization)
         │           │
         │           ▼
         └──► Phase 4 (Training Data)
                     │
                     ▼
              Phase 5 (Advanced Features)
                     │
                     ▼
              Phase 6 (Integration)
```

**Critical Path:** 2.1 → 3.1 → 4.2 → 6.1

The backward pass must work before meaningful training. GPU sprite generation must work before real-time use is practical.

---

## Resource Estimates

| Phase | Estimated Time | GPU Required |
|-------|---------------|--------------|
| Phase 2 | 5-7 days | Development GPU |
| Phase 3 | 6-8 days | Development GPU |
| Phase 4 | 7-10 days | Training GPU (optional cloud) |
| Phase 5 | 10-15 days | Training GPU |
| Phase 6 | 5-7 days | Development GPU |

**Total estimate:** 33-47 days of focused development

---

## Success Criteria

### Minimum Viable Product (MVP)

- [ ] Can train a SpriteKAN on 100 example sprites
- [ ] Generates recognizable sprites from latent codes
- [ ] Interpolation produces smooth transitions
- [ ] Generation speed < 100ms per 32x32 sprite on GPU

### Production Ready

- [ ] Training loss converges reliably
- [ ] Latent space has semantic structure
- [ ] Generation speed < 5ms per 32x32 sprite on GPU
- [ ] Model size < 10MB for portable deployment
- [ ] Integrated with CatSith neural backend

---

## Open Questions

1. **Optimal architecture:** What latent dimension and hidden layer sizes work best for pixel art?
2. **B-spline degree:** Is cubic (degree 3) optimal, or would higher degrees help?
3. **Training data volume:** How many example sprites are needed for good generalization?
4. **Comparison with VAE:** Is KAN actually better than a traditional VAE for this use case?

These will be answered through Phase 4 experimentation.

---

## Contributing

If you'd like to help with development:

1. Check the GitHub issues for tasks tagged with the current phase
2. Pick unclaimed tasks and comment to claim them
3. Follow the existing code style (run `cargo fmt` and `cargo clippy`)
4. Add tests for new functionality
5. Update this roadmap when tasks are completed
