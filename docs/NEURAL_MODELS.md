# Neural Model Integration Guide

This document outlines the neural network models suitable for CatSith's sprite generation
pipeline, integration approaches, and a training roadmap for custom models.

## Table of Contents

1. [Recommended Pre-trained Models](#recommended-pre-trained-models)
2. [Model Comparison](#model-comparison)
3. [Integration Architecture](#integration-architecture)
4. [Training Custom Models](#training-custom-models)
5. [Cost Estimates](#cost-estimates)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Recommended Pre-trained Models

### Tier 1: Production Ready (Small, Fast)

#### Segmind Tiny-SD
- **Source**: https://huggingface.co/segmind/tiny-sd
- **Parameters**: ~387M (55% fewer than SD 1.5)
- **Size**: ~350MB (FP16)
- **Speed**: 80% faster than SD 1.5
- **Quality**: Good for 512x512, acceptable for sprites
- **License**: CreativeML Open RAIL-M

```python
# Example usage with diffusers
from diffusers import DiffusionPipeline
pipeline = DiffusionPipeline.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float16)
```

#### Segmind Small-SD
- **Source**: https://github.com/segmind/distill-sd
- **Parameters**: ~559M (35% fewer than SD 1.5)
- **Size**: ~500MB (FP16)
- **Speed**: 60% faster than SD 1.5
- **Quality**: Better than Tiny-SD, still efficient

### Tier 2: Style-Specific Models

#### All-In-One-Pixel-Model
- **Source**: https://huggingface.co/PublicPrompts/All-In-One-Pixel-Model
- **Trigger Words**: `pixelsprite`, `16bitscene`
- **Use Case**: Pixel art sprites and scenes
- **Note**: Not pixel-perfect, may need post-processing

#### SD_PixelArt_SpriteSheet_Generator
- **Source**: https://huggingface.co/Onodofthenorth/SD_PixelArt_SpriteSheet_Generator
- **Use Case**: Multi-angle sprite sheets
- **Note**: Generates 4 directional views

### Tier 3: High Quality (Larger)

#### Pixel Art Diffusion XL - Sprite Shaper
- **Source**: https://civitai.com/models/277680/pixel-art-diffusion-xl
- **Base**: SDXL
- **Size**: ~6GB
- **Quality**: Best pixel art quality
- **Use Case**: Offline asset generation, not real-time

### Tier 4: Lightweight Components

#### LiteVAE (NeurIPS 2024)
- **Source**: https://neurips.cc/virtual/2024/poster/93756
- **Innovation**: 6x fewer encoder parameters using wavelet transforms
- **Use Case**: Fast encoding/decoding for latent manipulation
- **Status**: Research paper, implementation may need porting

#### Standalone VAE
- **Size**: ~80MB
- **Use Case**: Latent space sprite variations
- **Speed**: Very fast (<10ms per encode/decode)

---

## Model Comparison

| Model | Params | Size | Speed | Quality | Real-time? |
|-------|--------|------|-------|---------|------------|
| Tiny-SD | 387M | 350MB | 80% faster | Good | Yes (GPU) |
| Small-SD | 559M | 500MB | 60% faster | Better | Yes (GPU) |
| Pixel-Model | 860M | 1.7GB | Baseline | Pixel Art | Marginal |
| SDXL Pixel | 3.5B | 6GB | Slow | Excellent | No |
| VAE Only | 84M | 80MB | Very Fast | N/A | Yes (CPU) |

### Recommended Configuration

For CatSith's use cases:

1. **Real-time Game Rendering** (with GPU):
   - Tiny-SD + Pixel Art LoRA
   - Target: 5-10 FPS sprite generation

2. **Asset Pipeline** (offline):
   - Small-SD or full SD 1.5 + Pixel LoRA
   - Generate sprite sheets, cache results

3. **CPU-only / Edge**:
   - VAE-only mode for sprite variations
   - Pre-generated sprite atlas with neural upscaling

---

## Integration Architecture

### Current CatSith Structure

```
catsith-backend-neural/
├── src/
│   ├── embedder.rs      # Text → embedding (STUB)
│   ├── inference.rs     # ONNX model loading (STUB)
│   ├── renderer.rs      # Pipeline stage integration
│   ├── temporal.rs      # Frame coherence (WORKING)
│   └── models/
│       ├── diffusion.rs # Diffusion scheduler (STUB)
│       └── sprite_vae.rs # VAE encode/decode (STUB)
```

### Required Changes

#### 1. Model Loading (`inference.rs`)

```rust
use tract_onnx::prelude::*;

pub struct OnnxModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl OnnxModel {
    pub fn load(path: &Path) -> Result<Self, InferenceError> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact([1, 4, 64, 64]).into())?  // Latent shape
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            input_shape: vec![1, 4, 64, 64],
            output_shape: vec![1, 3, 512, 512],
        })
    }

    pub fn run(&self, input: &Tensor) -> Result<Tensor, InferenceError> {
        let result = self.model.run(tvec!(input.clone().into()))?;
        Ok(result[0].clone().into_tensor())
    }
}
```

#### 2. VAE Integration (`models/sprite_vae.rs`)

```rust
pub struct SpriteVAE {
    encoder: OnnxModel,
    decoder: OnnxModel,
    latent_dim: usize,
}

impl SpriteVAE {
    pub fn encode(&self, image: &ImageFrame) -> Result<Tensor, InferenceError> {
        let input = self.preprocess(image);
        let latent = self.encoder.run(&input)?;
        Ok(latent)
    }

    pub fn decode(&self, latent: &Tensor) -> Result<ImageFrame, InferenceError> {
        let output = self.decoder.run(latent)?;
        self.postprocess(&output)
    }

    pub fn interpolate(&self, a: &Tensor, b: &Tensor, t: f32) -> Tensor {
        // Spherical interpolation in latent space
        let theta = a.dot(b).acos();
        let sin_theta = theta.sin();

        a * ((1.0 - t) * theta).sin() / sin_theta
          + b * (t * theta).sin() / sin_theta
    }
}
```

#### 3. Diffusion Pipeline (`models/diffusion.rs`)

```rust
pub struct DiffusionPipeline {
    unet: OnnxModel,
    vae: SpriteVAE,
    text_encoder: OnnxModel,
    scheduler: DDPMScheduler,
    num_steps: usize,
}

impl DiffusionPipeline {
    pub fn generate(&self, prompt: &str, seed: u64) -> Result<ImageFrame, InferenceError> {
        // 1. Encode text prompt
        let text_embedding = self.text_encoder.encode(prompt)?;

        // 2. Initialize random latent
        let mut latent = self.random_latent(seed);

        // 3. Denoise iteratively
        for t in self.scheduler.timesteps() {
            let noise_pred = self.unet.predict_noise(&latent, t, &text_embedding)?;
            latent = self.scheduler.step(&latent, &noise_pred, t)?;
        }

        // 4. Decode to image
        self.vae.decode(&latent)
    }
}
```

#### 4. LoRA Application

```rust
impl DiffusionPipeline {
    pub fn apply_lora(&mut self, lora: &LoraWeights) -> Result<(), InferenceError> {
        // Modify UNet weights with LoRA deltas
        for (layer_name, (lora_a, lora_b)) in &lora.layers {
            if let Some(layer) = self.unet.get_layer_mut(layer_name) {
                // W' = W + alpha * (A @ B)
                let delta = lora_a.matmul(lora_b) * lora.alpha;
                layer.add_weights(&delta)?;
            }
        }
        Ok(())
    }
}
```

### ONNX Model Conversion

Convert Hugging Face models to ONNX:

```bash
# Install optimum
pip install optimum[onnxruntime]

# Convert Tiny-SD to ONNX
optimum-cli export onnx \
    --model segmind/tiny-sd \
    --task stable-diffusion \
    tiny-sd-onnx/
```

This produces:
```
tiny-sd-onnx/
├── text_encoder/model.onnx      (~250MB)
├── unet/model.onnx              (~350MB)
├── vae_encoder/model.onnx       (~80MB)
└── vae_decoder/model.onnx       (~80MB)
```

---

## Training Custom Models

### Option A: Fine-tune Existing Model with LoRA

**Lowest cost, fastest iteration**

#### Dataset Requirements
- 500-2000 high-quality sprite images
- Consistent style (your game's art direction)
- Captions describing each sprite

#### Training Process

```bash
# Using kohya-ss/sd-scripts
accelerate launch train_network.py \
    --pretrained_model_name_or_path="segmind/tiny-sd" \
    --train_data_dir="./sprites" \
    --output_dir="./catsith-lora" \
    --network_module=networks.lora \
    --network_dim=32 \
    --network_alpha=16 \
    --resolution=512 \
    --train_batch_size=4 \
    --max_train_epochs=10 \
    --learning_rate=1e-4
```

#### Cost Estimate
- **GPU Time**: 2-4 hours on A100
- **Cloud Cost**: $8-16 (Lambda Labs) or $16-32 (AWS)
- **Dataset Prep**: 4-8 hours manual work

### Option B: Knowledge Distillation (Custom Tiny Model)

**Medium cost, best inference speed**

#### Approach
Following Segmind's distillation methodology:

1. Start with SD 1.5 as teacher
2. Remove U-Net blocks (12 → 8 or 12 → 6)
3. Train student to match teacher outputs
4. Fine-tune on sprite dataset

#### Training Script Outline

```python
# Based on segmind/distill-sd
from distill_sd import create_student_unet, DistillationTrainer

# Create smaller U-Net
student_unet = create_student_unet(
    teacher_unet,
    distill_level="sd_tiny",  # or "sd_small"
    remove_blocks=[1, 2, 7, 8]  # Remove specific blocks
)

# Distillation training
trainer = DistillationTrainer(
    teacher_model=teacher_pipeline,
    student_unet=student_unet,
    output_weight=1.0,      # Output-level KD loss
    feature_weight=0.5,     # Feature-level KD loss
)

trainer.train(
    dataset=sprite_dataset,
    epochs=50,
    batch_size=8,
    learning_rate=1e-5,
)
```

#### Cost Estimate
- **GPU Time**: 24-72 hours on A100
- **Cloud Cost**: $100-300 (Lambda Labs) or $200-600 (AWS)
- **Dataset**: 10,000-50,000 images recommended

### Option C: Train VAE from Scratch

**For specialized latent space**

#### Architecture: Sprite-Specific VAE

```python
class SpriteVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder: 64x64x3 → 64-dim latent
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim * 2),  # mu, logvar
        )

        # Decoder: 64-dim latent → 64x64x3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )
```

#### Training Objectives

1. **Reconstruction Loss**: MSE between input and output
2. **Perceptual Loss**: VGG feature matching
3. **KL Divergence**: Regularize latent space
4. **Sprite-Specific**: Direction consistency loss

```python
def sprite_direction_loss(latent, direction_labels):
    """Encourage latent space to encode direction consistently"""
    # Cluster latents by direction
    direction_centroids = compute_centroids(latent, direction_labels)

    # Minimize intra-direction variance
    intra_loss = intra_cluster_variance(latent, direction_labels)

    # Maximize inter-direction separation
    inter_loss = -inter_cluster_distance(direction_centroids)

    return intra_loss + 0.5 * inter_loss
```

#### Cost Estimate
- **GPU Time**: 8-24 hours on A100
- **Cloud Cost**: $32-100
- **Advantage**: Tiny model (~5-10MB), very fast inference

### Option D: Full Custom Diffusion Model

**Highest cost, maximum control**

Only recommended if:
- Existing models don't meet quality needs
- Specific architectural requirements
- Large training budget available

#### Architecture Considerations

1. **U-Net Size**: Start with 4-6 blocks (vs 12 in SD)
2. **Attention**: Use linear attention for speed
3. **Resolution**: Train at 64x64, upsample if needed
4. **Conditioning**: Entity type + direction embeddings

#### Cost Estimate
- **GPU Time**: 200-500 hours on A100
- **Cloud Cost**: $800-2000
- **Dataset**: 100,000+ images recommended

---

## Cost Estimates Summary

| Approach | GPU Hours | Cloud Cost | Time to Deploy |
|----------|-----------|------------|----------------|
| LoRA Fine-tune | 2-4h | $8-32 | 1 day |
| Distilled Model | 24-72h | $100-300 | 1 week |
| Custom VAE | 8-24h | $32-100 | 3 days |
| Full Custom | 200-500h | $800-2000 | 2-4 weeks |

### Cloud GPU Providers (2024 Pricing)

| Provider | A100 40GB/hr | A100 80GB/hr | Notes |
|----------|--------------|--------------|-------|
| Lambda Labs | $1.10 | $1.89 | Best value, limited availability |
| RunPod | $1.44 | $2.49 | Good availability |
| Vast.ai | $0.80-1.50 | $1.50-2.50 | Variable, community GPUs |
| AWS p4d | $3.50 | - | Most reliable, expensive |
| Google Cloud | $3.67 | - | Good for large jobs |

---

## Implementation Roadmap

### Phase 1: Integration Foundation (Week 1-2)
- [ ] Set up ONNX model loading with tract
- [ ] Implement VAE encode/decode
- [ ] Test with pre-converted Tiny-SD ONNX models
- [ ] Benchmark inference speed

### Phase 2: Basic Generation (Week 3-4)
- [ ] Implement DDPM scheduler
- [ ] Wire up text encoder → U-Net → VAE pipeline
- [ ] Add LoRA loading and application
- [ ] Generate first sprites from text prompts

### Phase 3: Optimization (Week 5-6)
- [ ] Profile and optimize hot paths
- [ ] Implement caching for repeated generations
- [ ] Add batch generation support
- [ ] Test temporal coherence with animation

### Phase 4: Custom Training (Week 7+)
- [ ] Collect and prepare sprite dataset
- [ ] Train LoRA for game's art style
- [ ] Evaluate and iterate on quality
- [ ] Optional: Train distilled model

### Phase 5: Production (Ongoing)
- [ ] Quantize models (FP16 → INT8)
- [ ] Implement model hot-swapping
- [ ] Add quality tier selection
- [ ] Performance monitoring

---

## Quick Start Checklist

1. **Download Pre-trained Models**
   ```bash
   # Install conversion tools
   pip install optimum[onnxruntime] diffusers

   # Convert Tiny-SD
   optimum-cli export onnx --model segmind/tiny-sd tiny-sd-onnx/
   ```

2. **Place Models in CatSith**
   ```
   catsith/
   └── models/
       └── tiny-sd/
           ├── text_encoder.onnx
           ├── unet.onnx
           ├── vae_encoder.onnx
           └── vae_decoder.onnx
   ```

3. **Update Configuration**
   ```toml
   # catsith.toml
   [neural]
   model_path = "models/tiny-sd"
   inference_steps = 20
   guidance_scale = 7.5
   ```

4. **Run Integration Tests**
   ```bash
   cargo test -p catsith-backend-neural
   ```

---

## References

- [Segmind Distillation Blog](https://blog.segmind.com/introducing-sd-small-and-sd-tiny-stable-diffusion-models/)
- [Segmind distill-sd GitHub](https://github.com/segmind/distill-sd)
- [ONNX Runtime Generative AI](https://onnxruntime.ai/generative-ai)
- [Hugging Face Optimum](https://huggingface.co/docs/optimum)
- [LiteVAE Paper (NeurIPS 2024)](https://neurips.cc/virtual/2024/poster/93756)
- [Tract ONNX Runtime](https://github.com/sonos/tract)
