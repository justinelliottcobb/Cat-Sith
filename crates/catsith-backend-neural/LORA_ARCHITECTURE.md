# LoRA Architecture in CatSith Neural Backend

This document describes the LoRA (Low-Rank Adaptation) implementation strategy for the
CatSith neural rendering backend.

## Background

LoRA allows customizing Stable Diffusion models by adding small learned weight deltas
to specific layers. The core operation is:

```
W' = W + (alpha / rank) * (lora_down @ lora_up) * weight
```

Where:
- `W` = original layer weights
- `lora_down` = down-projection matrix (in_features × rank)
- `lora_up` = up-projection matrix (rank × out_features)
- `alpha` = scaling factor (often equal to rank)
- `weight` = user-specified strength (0.0 to 1.0)

## Industry Approaches

### 1. Pre-Merge / Fuse (HuggingFace Diffusers)

Used by `diffusers` library via `fuse_lora()` and `merge_and_unload()`.

**How it works:**
- LoRA weights are merged directly into base model weights at load time
- Forward pass uses standard operations with no LoRA overhead
- Original weights can be cached for undo capability

**Pros:**
- Zero per-inference overhead
- Best latency for generation
- Compatible with torch.compile optimizations

**Cons:**
- Changing LoRA weights requires re-merge (can take several seconds)
- 2x memory for UNet if caching original weights for undo
- Not suitable for per-frame LoRA weight changes

**References:**
- https://huggingface.co/docs/diffusers/en/tutorials/using_peft_for_inference
- https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras

### 2. Lazy Patching / On-Demand (ComfyUI)

Used by ComfyUI's `ModelPatcher` system.

**How it works:**
- LoRA weights stored as patches keyed to layer names
- Patches applied during forward pass via wrapper objects
- `calculate_weight()` blends multiple patch contributions on-demand

**Pros:**
- Can change LoRA weights between frames without re-merge
- No need to cache original weights
- Supports dynamic weight interpolation (fade between styles)
- Memory efficient (only stores small LoRA tensors)

**Cons:**
- Overhead on every forward pass (extra matmuls per layer)
- More complex implementation
- Higher memory bandwidth usage

**References:**
- https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_patcher.py
- https://blog.comfy.org/p/masking-and-scheduling-lora-and-model-weights

## CatSith Implementation: Hybrid Approach

CatSith implements a hybrid approach optimized for real-time game rendering:

### Default Mode: Pre-Merge (`LoraMode::Fused`)

For the common case where a player selects a style and generates many frames:

```rust
let mut pipeline = CandleDiffusionPipeline::new(config)?;
pipeline.load()?;

// Add LoRAs to the stack
pipeline.add_lora("pixel-style.safetensors", 0.8)?;
pipeline.add_lora("anime-eyes.safetensors", 0.5)?;

// Merge LoRAs into model weights (one-time cost)
pipeline.apply_loras()?;

// Generate many frames with no LoRA overhead
for _ in 0..100 {
    let image = pipeline.generate("pixel art warrior", None, seed)?;
}

// Changing styles requires re-merge (loading screen)
pipeline.unapply_loras()?;
pipeline.clear_loras();
pipeline.add_lora("watercolor.safetensors", 1.0)?;
pipeline.apply_loras()?;
```

### Future: Dynamic Mode (`LoraMode::Dynamic`)

Reserved for future implementation if dynamic style blending is needed:

```rust
pipeline.set_lora_mode(LoraMode::Dynamic);

// Weights can change per-frame without re-merge
for t in 0..100 {
    let blend = t as f32 / 100.0;
    pipeline.set_lora_weight("style-a", 1.0 - blend)?;
    pipeline.set_lora_weight("style-b", blend)?;
    pipeline.generate(...)?;
}
```

## Implementation Details

### Weight Merging Algorithm

For each layer that has a LoRA:

```rust
// Compute LoRA delta: down @ up
let delta = lora_down.matmul(&lora_up)?;

// Scale by alpha/rank and user weight
let scale = (alpha / rank as f32) * user_weight;
let scaled_delta = delta * scale;

// Add to base weights
new_weights = base_weights + scaled_delta;
```

### Memory Management

- **Original weights cache**: Stored when `apply_loras()` is called
- **Cache cleared**: When `unapply_loras()` restores weights
- **Memory estimate**: ~2x UNet size when LoRAs are fused with undo capability
  - SD 1.5 UNet: ~3.4 GB → ~6.8 GB with cache
  - SDXL UNet: ~10 GB → ~20 GB with cache

### Layer Name Matching

LoRA files use various naming conventions:

```
# Common patterns
unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight
unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight

# Text encoder
text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_down.weight
```

CatSith normalizes these to match against the loaded model's parameter names.

## Limitations

1. **No per-layer weight control**: All layers use the same LoRA weight
   (ComfyUI supports per-block weights via LoRA Block Weight nodes)

2. **Single LoRA mode**: Cannot mix fused and dynamic LoRAs simultaneously

3. **UNet-only initially**: Text encoder LoRA support is secondary priority

## Future Enhancements

- [ ] Dynamic mode implementation for real-time style blending
- [ ] Per-layer/per-block weight control
- [ ] Text encoder LoRA support
- [ ] LoRA caching to avoid re-parsing safetensors files
- [ ] Async merge operations with progress callbacks
