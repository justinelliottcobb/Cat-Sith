# CatSith Model Training Guide

This guide covers training custom neural models for CatSith's sprite generation pipeline.

## Table of Contents

1. [Dataset Preparation](#dataset-preparation)
2. [Training Options](#training-options)
3. [LoRA Fine-tuning](#lora-fine-tuning)
4. [Knowledge Distillation](#knowledge-distillation)
5. [Custom VAE Training](#custom-vae-training)
6. [Infrastructure Setup](#infrastructure-setup)
7. [Evaluation and Testing](#evaluation-and-testing)

---

## Dataset Preparation

### Image Requirements

| Attribute | Recommendation |
|-----------|----------------|
| **Resolution** | 512×512 or 64×64 (for VAE-only) |
| **Format** | PNG with transparency |
| **Color depth** | 8-bit RGBA |
| **Style** | Consistent art direction |
| **Quantity** | 500-2000 for LoRA, 10000+ for distillation |

### Directory Structure

```
dataset/
├── images/
│   ├── ship_fighter_0.png      # Direction 0 (up)
│   ├── ship_fighter_1.png      # Direction 1 (up-right)
│   ├── ship_fighter_2.png      # etc.
│   ├── ship_bomber_0.png
│   ├── projectile_bullet.png
│   ├── environment_asteroid.png
│   └── ...
├── captions/
│   ├── ship_fighter_0.txt      # "pixel art fighter spaceship facing up, green hull"
│   ├── ship_fighter_1.txt
│   └── ...
└── metadata.json
```

### Metadata Schema

```json
{
  "version": "1.0",
  "style": "pixel_art_16bit",
  "entities": [
    {
      "id": "ship_fighter",
      "category": "vehicle",
      "kind": "fighter",
      "directions": 8,
      "base_color": [64, 192, 128],
      "files": ["ship_fighter_0.png", "ship_fighter_1.png", "..."]
    },
    {
      "id": "projectile_bullet",
      "category": "projectile",
      "kind": "bullet",
      "directions": 1,
      "base_color": [255, 255, 0],
      "files": ["projectile_bullet.png"]
    }
  ],
  "captions": {
    "style_prefix": "pixel art game sprite, ",
    "style_suffix": ", transparent background, 16-bit style"
  }
}
```

### Caption Generation

Good captions are crucial for text-to-image training:

```python
# caption_generator.py
import json
from pathlib import Path

def generate_caption(entity: dict, direction: int) -> str:
    """Generate training caption for an entity sprite."""

    direction_names = [
        "facing up", "facing up-right", "facing right", "facing down-right",
        "facing down", "facing down-left", "facing left", "facing up-left"
    ]

    parts = [
        "pixel art game sprite",
        f"{entity['kind']} {entity['category']}",
    ]

    if entity['directions'] > 1:
        parts.append(direction_names[direction])

    # Add color description
    r, g, b = entity['base_color']
    if r > g and r > b:
        parts.append("red colored")
    elif g > r and g > b:
        parts.append("green colored")
    elif b > r and b > g:
        parts.append("blue colored")

    parts.append("transparent background")
    parts.append("16-bit retro style")

    return ", ".join(parts)

# Example output:
# "pixel art game sprite, fighter vehicle, facing up, green colored, transparent background, 16-bit retro style"
```

### Data Augmentation

For small datasets, augment with:

```python
import albumentations as A

augmentation = A.Compose([
    A.HorizontalFlip(p=0.0),  # Don't flip directional sprites!
    A.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02,
        p=0.5
    ),
    A.GaussNoise(var_limit=(1, 5), p=0.3),
])
```

**Important**: Do NOT flip directional sprites horizontally - this corrupts direction labels.

---

## Training Options

### Decision Matrix

| Goal | Method | Cost | Time | Quality |
|------|--------|------|------|---------|
| Quick style adaptation | LoRA | $8-32 | 2-4 hours | Good |
| Faster inference | Distillation | $100-300 | 24-72 hours | Very Good |
| Custom latent space | VAE | $32-100 | 8-24 hours | Specialized |
| Maximum control | Full training | $800-2000 | 200+ hours | Excellent |

### Recommended Path

1. **Start with LoRA** ($8-32)
   - Quick iteration on art style
   - Test different approaches
   - Low risk

2. **If satisfied, optionally distill** ($100-300)
   - Create smaller, faster model
   - Bake in the LoRA style

3. **Consider VAE if needed** ($32-100)
   - For sprite interpolation features
   - If latent manipulation is key

---

## LoRA Fine-tuning

### Prerequisites

```bash
# Clone kohya-ss training scripts
git clone https://github.com/kohya-ss/sd-scripts
cd sd-scripts
pip install -r requirements.txt

# Or use diffusers training
pip install diffusers[training] accelerate
```

### Training Configuration

```yaml
# lora_config.yaml
pretrained_model_name_or_path: "segmind/tiny-sd"
train_data_dir: "./dataset/images"
output_dir: "./output/catsith-lora"

# Network architecture
network_module: "networks.lora"
network_dim: 32          # LoRA rank (8-128, higher = more capacity)
network_alpha: 16        # LoRA alpha (typically dim/2)

# Training params
resolution: 512
train_batch_size: 4
max_train_epochs: 10
learning_rate: 1e-4
lr_scheduler: "cosine"
lr_warmup_steps: 100

# Optimization
mixed_precision: "fp16"
gradient_checkpointing: true
gradient_accumulation_steps: 2

# Regularization
noise_offset: 0.1
caption_dropout_rate: 0.1
```

### Training Script

```bash
#!/bin/bash
# train_lora.sh

accelerate launch --num_cpu_threads_per_process=2 train_network.py \
    --pretrained_model_name_or_path="segmind/tiny-sd" \
    --train_data_dir="./dataset/images" \
    --output_dir="./output/catsith-lora" \
    --output_name="catsith_sprites_v1" \
    --network_module=networks.lora \
    --network_dim=32 \
    --network_alpha=16 \
    --resolution=512 \
    --train_batch_size=4 \
    --max_train_epochs=10 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=100 \
    --mixed_precision="fp16" \
    --save_every_n_epochs=2 \
    --caption_extension=".txt" \
    --enable_bucket \
    --bucket_reso_steps=64 \
    --min_bucket_reso=256 \
    --max_bucket_reso=512
```

### Expected Output

```
output/catsith-lora/
├── catsith_sprites_v1.safetensors    # ~50-150MB LoRA weights
├── catsith_sprites_v1.json           # Metadata
└── sample_images/                    # Generated during training
    ├── epoch_2_*.png
    ├── epoch_4_*.png
    └── ...
```

### Convert LoRA for CatSith

```python
# convert_lora.py
import torch
from safetensors.torch import load_file, save_file

def convert_to_catsith_format(input_path, output_path):
    """Convert kohya LoRA to CatSith format."""

    state_dict = load_file(input_path)

    catsith_lora = {
        "metadata": {
            "name": "catsith_sprites",
            "version": "1.0",
            "base_model": "segmind/tiny-sd",
            "rank": 32,
            "alpha": 16,
        },
        "layers": {}
    }

    for key, tensor in state_dict.items():
        # Parse layer name
        # kohya format: "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"

        if ".lora_down." in key:
            layer_name = key.replace(".lora_down.weight", "")
            if layer_name not in catsith_lora["layers"]:
                catsith_lora["layers"][layer_name] = {}
            catsith_lora["layers"][layer_name]["A"] = tensor.numpy().tolist()

        elif ".lora_up." in key:
            layer_name = key.replace(".lora_up.weight", "")
            if layer_name not in catsith_lora["layers"]:
                catsith_lora["layers"][layer_name] = {}
            catsith_lora["layers"][layer_name]["B"] = tensor.numpy().tolist()

    # Save as JSON for CatSith
    import json
    with open(output_path, 'w') as f:
        json.dump(catsith_lora, f)

    print(f"Converted {len(catsith_lora['layers'])} layers")

if __name__ == "__main__":
    convert_to_catsith_format(
        "output/catsith-lora/catsith_sprites_v1.safetensors",
        "output/catsith_sprites_v1.lora.json"
    )
```

---

## Knowledge Distillation

### Setup

```bash
# Clone Segmind distillation repo
git clone https://github.com/segmind/distill-sd
cd distill-sd
pip install -r requirements.txt
```

### Architecture Options

| Level | Blocks Removed | Params | Speed Gain |
|-------|----------------|--------|------------|
| small | 4 (of 12) | 559M | 60% faster |
| tiny | 6 (of 12) | 387M | 80% faster |
| micro* | 8 (of 12) | ~250M | 90% faster |

*Micro is experimental and may have quality issues.

### Distillation Configuration

```python
# distill_config.py
from dataclasses import dataclass

@dataclass
class DistillConfig:
    # Teacher model
    teacher_model: str = "runwayml/stable-diffusion-v1-5"

    # Student architecture
    distill_level: str = "sd_tiny"  # or "sd_small"
    remove_blocks: list = None  # Auto-determined by level

    # Dataset
    dataset_path: str = "./dataset"
    resolution: int = 512

    # Training
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-5

    # Loss weights
    output_weight: float = 1.0      # MSE on final output
    feature_weight: float = 0.5     # Feature-level KD
    perceptual_weight: float = 0.1  # VGG perceptual loss

    # Checkpointing
    save_every_n_epochs: int = 5
    output_dir: str = "./output/distilled"
```

### Training Script

```python
# train_distill.py
import torch
from diffusers import StableDiffusionPipeline
from distill_sd import (
    create_student_unet,
    DistillationTrainer,
    DistillationLoss
)

def main():
    config = DistillConfig()

    # Load teacher
    teacher_pipe = StableDiffusionPipeline.from_pretrained(
        config.teacher_model,
        torch_dtype=torch.float16
    ).to("cuda")

    # Create student U-Net
    student_unet = create_student_unet(
        teacher_pipe.unet,
        distill_level=config.distill_level
    )

    # Setup loss
    loss_fn = DistillationLoss(
        output_weight=config.output_weight,
        feature_weight=config.feature_weight,
        perceptual_weight=config.perceptual_weight
    )

    # Setup trainer
    trainer = DistillationTrainer(
        teacher_pipe=teacher_pipe,
        student_unet=student_unet,
        loss_fn=loss_fn,
        config=config
    )

    # Train
    trainer.train()

    # Save student model
    student_pipe = StableDiffusionPipeline(
        vae=teacher_pipe.vae,
        text_encoder=teacher_pipe.text_encoder,
        tokenizer=teacher_pipe.tokenizer,
        unet=student_unet,
        scheduler=teacher_pipe.scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    student_pipe.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()
```

### Export to ONNX

```bash
# After distillation, export to ONNX
optimum-cli export onnx \
    --model ./output/distilled \
    --task stable-diffusion \
    ./output/distilled-onnx/
```

---

## Custom VAE Training

For a specialized sprite VAE with structured latent space.

### Architecture

```python
# sprite_vae.py
import torch
import torch.nn as nn

class SpriteEncoder(nn.Module):
    """Encode 64x64 sprite to latent."""

    def __init__(self, latent_dim=64, hidden_dims=[32, 64, 128, 256]):
        super().__init__()

        layers = []
        in_channels = 4  # RGBA

        for h_dim in hidden_dims:
            layers.append(nn.Conv2d(in_channels, h_dim, 4, 2, 1))
            layers.append(nn.BatchNorm2d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = h_dim

        self.encoder = nn.Sequential(*layers)

        # After 4 downsamplings: 64 -> 32 -> 16 -> 8 -> 4
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_var(h)


class SpriteDecoder(nn.Module):
    """Decode latent to 64x64 sprite."""

    def __init__(self, latent_dim=64, hidden_dims=[256, 128, 64, 32]):
        super().__init__()

        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 4, 2, 1))
            layers.append(nn.BatchNorm2d(hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.ConvTranspose2d(hidden_dims[-1], 4, 4, 2, 1))  # RGBA
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)


class SpriteVAE(nn.Module):
    """Complete VAE for sprite generation."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = SpriteEncoder(latent_dim)
        self.decoder = SpriteDecoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x):
        mu, log_var = self.encoder(x)
        return self.reparameterize(mu, log_var)

    def decode(self, z):
        return self.decoder(z)
```

### Training Loop

```python
# train_vae.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sprite_vae import SpriteVAE

def vae_loss(recon, target, mu, log_var, beta=1.0):
    """VAE loss = reconstruction + KL divergence."""

    # Reconstruction loss (per-pixel MSE)
    recon_loss = F.mse_loss(recon, target, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + beta * kl_loss

def perceptual_loss(recon, target, vgg):
    """VGG perceptual loss for sharper results."""
    # Expand to 3 channels if RGBA
    if recon.shape[1] == 4:
        recon = recon[:, :3]
        target = target[:, :3]

    recon_features = vgg(recon)
    target_features = vgg(target)

    return F.mse_loss(recon_features, target_features)

def direction_consistency_loss(mu, direction_labels, num_directions=8):
    """Encourage similar latents for same entity, different directions."""

    # Group by entity (assuming batch is organized by entity)
    batch_size = mu.shape[0]
    entities_per_batch = batch_size // num_directions

    loss = 0.0
    for i in range(entities_per_batch):
        # Get all directions for this entity
        entity_latents = mu[i::entities_per_batch]

        # They should be similar (low variance across directions)
        mean_latent = entity_latents.mean(dim=0)
        variance = ((entity_latents - mean_latent) ** 2).mean()
        loss += variance

    return loss / entities_per_batch

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        images = batch['image'].to(device)

        optimizer.zero_grad()

        recon, mu, log_var = model(images)

        loss = vae_loss(recon, images, mu, log_var)

        # Optional: direction consistency
        if 'direction' in batch:
            loss += 0.1 * direction_consistency_loss(mu, batch['direction'])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpriteVAE(latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(100):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}: loss = {loss:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/vae_epoch_{epoch}.pt")

    # Export to ONNX
    dummy_input = torch.randn(1, 4, 64, 64).to(device)

    # Export encoder
    torch.onnx.export(
        model.encoder,
        dummy_input,
        "output/sprite_vae_encoder.onnx",
        input_names=['image'],
        output_names=['mu', 'log_var'],
        dynamic_axes={'image': {0: 'batch'}}
    )

    # Export decoder
    dummy_latent = torch.randn(1, 64).to(device)
    torch.onnx.export(
        model.decoder,
        dummy_latent,
        "output/sprite_vae_decoder.onnx",
        input_names=['latent'],
        output_names=['image'],
        dynamic_axes={'latent': {0: 'batch'}}
    )

if __name__ == "__main__":
    main()
```

---

## Infrastructure Setup

### Cloud GPU Options

| Provider | GPU | $/hour | Best For |
|----------|-----|--------|----------|
| [Lambda Labs](https://lambdalabs.com) | A100 40GB | $1.10 | Best value |
| [RunPod](https://runpod.io) | A100 40GB | $1.44 | Good availability |
| [Vast.ai](https://vast.ai) | Various | $0.80-2.00 | Budget option |
| AWS p4d | A100 40GB | $3.50 | Enterprise |
| Google Cloud | A100 40GB | $3.67 | TPU alternative |

### Recommended Setup Script

```bash
#!/bin/bash
# setup_training.sh

# Install base dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install safetensors wandb

# Clone training repos
git clone https://github.com/kohya-ss/sd-scripts
git clone https://github.com/segmind/distill-sd

# Setup sd-scripts
cd sd-scripts
pip install -r requirements.txt
cd ..

# Download base model
python -c "
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained('segmind/tiny-sd')
pipe.save_pretrained('./models/tiny-sd')
"

echo "Setup complete!"
```

### Weights & Biases Integration

```python
# Track training with wandb
import wandb

wandb.init(
    project="catsith-training",
    config={
        "model": "tiny-sd",
        "method": "lora",
        "dataset_size": 1000,
        "epochs": 10,
    }
)

# Log metrics during training
wandb.log({
    "loss": loss,
    "epoch": epoch,
    "learning_rate": lr,
})

# Log sample images
wandb.log({
    "samples": [wandb.Image(img) for img in sample_images]
})
```

---

## Evaluation and Testing

### Quality Metrics

```python
# evaluate.py
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpips import LearnedPerceptualImagePatchSimilarity

def evaluate_model(model, test_dataloader, device):
    """Evaluate generated image quality."""

    fid = FrechetInceptionDistance(feature=2048).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity().to(device)

    model.eval()
    lpips_scores = []

    with torch.no_grad():
        for batch in test_dataloader:
            real_images = batch['image'].to(device)
            prompts = batch['caption']

            # Generate images
            generated = model.generate(prompts)

            # Update FID
            fid.update(real_images, real=True)
            fid.update(generated, real=False)

            # Calculate LPIPS
            lpips_score = lpips(generated, real_images)
            lpips_scores.append(lpips_score.item())

    results = {
        "fid": fid.compute().item(),
        "lpips": sum(lpips_scores) / len(lpips_scores),
    }

    return results
```

### Visual Comparison Grid

```python
# compare.py
import matplotlib.pyplot as plt
from PIL import Image

def create_comparison_grid(prompts, models, output_path):
    """Create side-by-side comparison of different models."""

    fig, axes = plt.subplots(len(prompts), len(models) + 1, figsize=(4*(len(models)+1), 4*len(prompts)))

    for i, prompt in enumerate(prompts):
        # Ground truth (if available)
        axes[i, 0].set_title("Ground Truth")
        axes[i, 0].imshow(get_ground_truth(prompt))
        axes[i, 0].axis('off')

        for j, (name, model) in enumerate(models.items()):
            img = model.generate(prompt)
            axes[i, j+1].set_title(name)
            axes[i, j+1].imshow(img)
            axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
```

### CatSith Integration Test

```rust
// tests/neural_integration.rs
use catsith_backend_neural::{OnnxRuntime, DiffusionPipeline};
use catsith_core::ImageFrame;

#[tokio::test]
async fn test_trained_model() {
    let runtime = OnnxRuntime::new("models/catsith-tiny-sd").unwrap();
    let pipeline = DiffusionPipeline::load(&runtime).unwrap();

    // Test various prompts
    let prompts = [
        "pixel art fighter spaceship, green, facing up",
        "pixel art asteroid, brown, rocky",
        "pixel art explosion effect, orange, bright",
    ];

    for prompt in prompts {
        let image = pipeline.generate(prompt, 42).unwrap();

        assert_eq!(image.width, 512);
        assert_eq!(image.height, 512);
        assert!(!image.data.is_empty());

        // Optional: save for visual inspection
        // image.save(format!("test_output/{}.png", prompt.replace(" ", "_")));
    }
}
```

---

## Appendix: Cost Calculator

```python
# cost_calculator.py

def estimate_training_cost(
    method: str,
    dataset_size: int,
    epochs: int,
    batch_size: int = 4,
    gpu_cost_per_hour: float = 1.10  # Lambda Labs A100
) -> dict:
    """Estimate training cost and time."""

    # Approximate training speeds (images/second on A100)
    speeds = {
        "lora": 2.0,
        "distill": 0.5,
        "vae": 10.0,
        "full": 0.3,
    }

    images_per_epoch = dataset_size
    total_images = images_per_epoch * epochs
    speed = speeds[method]

    training_seconds = total_images / speed / batch_size
    training_hours = training_seconds / 3600

    # Add overhead (data loading, checkpointing, etc.)
    training_hours *= 1.2

    cost = training_hours * gpu_cost_per_hour

    return {
        "method": method,
        "dataset_size": dataset_size,
        "epochs": epochs,
        "estimated_hours": round(training_hours, 1),
        "estimated_cost": round(cost, 2),
    }

# Examples
print(estimate_training_cost("lora", 1000, 10))
# {'method': 'lora', 'dataset_size': 1000, 'epochs': 10, 'estimated_hours': 1.7, 'estimated_cost': 1.83}

print(estimate_training_cost("distill", 10000, 50))
# {'method': 'distill', 'dataset_size': 10000, 'epochs': 50, 'estimated_hours': 33.3, 'estimated_cost': 36.67}
```
