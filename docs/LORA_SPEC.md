# CatSith LoRA Specification

## Overview

LoRAs (Low-Rank Adaptations) allow customization of visual rendering without modifying base models. This specification defines the format, loading, and application of LoRAs in CatSith.

## LoRA Categories

| Category    | Affects                           | Example                    |
|-------------|-----------------------------------|----------------------------|
| Aesthetic   | Overall visual style              | "anime", "pixel art"       |
| Entity      | Specific entity rendering         | "custom ship design"       |
| Effects     | Visual effects                    | "retro explosions"         |
| Environment | Background/environment rendering  | "nebula style"             |
| Color       | Color grading/palette             | "neon colors"              |

## Directory Structure

```
my_lora/
├── manifest.json       # Required: LoRA metadata
├── weights.bin         # Required: Weight data
├── preview.png         # Optional: Preview image
└── preview_2.png       # Optional: Additional previews
```

## Manifest Format

```json
{
  "version": 1,
  "hash": "base64-encoded-32-bytes",
  "name": "Anime Style",
  "description": "Renders scenes in anime/cel-shaded style",
  "creator": {
    "name": "Artist Name",
    "contact": "artist@example.com",
    "website": "https://example.com"
  },
  "category": "Aesthetic",
  "compatible_with": [
    {
      "name": "sprite_vae_v1",
      "version": "1.0"
    }
  ],
  "weights": {
    "path": "weights.bin",
    "size_bytes": 16777216,
    "hash": "base64-encoded-32-bytes",
    "rank": 16,
    "alpha": 1.0
  },
  "previews": [
    {
      "path": "preview.png",
      "description": "Fighter ship in anime style"
    }
  ],
  "license": "CC-BY-4.0",
  "tags": ["anime", "cel-shaded", "stylized"],
  "created_at": 1700000000
}
```

## Weight File Format

### Simple Format (v1)

Binary file with layer weights:

```
Header (16 bytes):
  magic:    [u8; 4]  = "LORA"
  version:  u32      = 1
  rank:     u32      = LoRA rank
  layers:   u32      = Number of layers

For each layer:
  name_len: u32
  name:     [u8; name_len]  (UTF-8 string)
  in_dim:   u32
  out_dim:  u32
  a_data:   [f32; in_dim * rank]  (little-endian)
  b_data:   [f32; rank * out_dim] (little-endian)
```

## LoRA Application

LoRAs modify base model weights via low-rank decomposition:

```
W' = W + α/r * (A @ B)

Where:
  W  = Original weight matrix
  W' = Modified weight matrix
  α  = Alpha scaling factor
  r  = Rank
  A  = Down projection matrix (in_dim × rank)
  B  = Up projection matrix (rank × out_dim)
```

## Stacking Multiple LoRAs

LoRAs can be combined:

```rust
let mut stack = LoraStack::empty();
stack.push(anime_lora, 0.8);    // 80% strength
stack.push(neon_lora, 0.5);     // 50% strength
stack.push(custom_ship, 1.0);   // 100% strength

pipeline.set_loras(stack);
```

Weights are applied in order, so later LoRAs can override earlier ones.

## Validation

Before loading, LoRAs should be validated:

1. **Manifest Validation**
   - Required fields present
   - Valid category
   - Non-zero rank and size

2. **Weight Validation**
   - File size matches manifest
   - Hash matches manifest
   - Dimensions are consistent

3. **Compatibility Check**
   - Model architecture supported
   - Rank compatible with model

```rust
let validator = LoraValidator::new();
let result = validator.validate(&manifest, &weights);

if !result.valid {
    for error in result.errors {
        eprintln!("Error: {:?}", error);
    }
}
```

## Security Considerations

1. **Always verify hashes** before loading weights
2. **Sandboxed loading** - don't execute code from LoRA files
3. **Size limits** - reject unreasonably large files
4. **Path traversal** - validate all paths stay within LoRA directory

## Creating LoRAs

### Training (Conceptual)

1. Prepare training data (paired semantic/visual examples)
2. Fine-tune base model with LoRA adapters
3. Extract LoRA weights
4. Package with manifest

### Packaging

```rust
use catsith_lora::*;

let mut manifest = LoraManifest::new("My Style", LoraCategory::Aesthetic)
    .with_description("Custom style description")
    .with_creator(CreatorInfo::new("Your Name"))
    .with_weights(WeightInfo::new("weights.bin", file_size, rank))
    .with_license("CC-BY-4.0")
    .with_tag("custom");

manifest.compute_hash(&weight_data);

let loader = LoraLoader::default();
loader.save_to_dir(&LoadedLora { manifest, weights, source }, "my_lora/")?;
```

## Registry

LoRAs are discovered via search paths:

```rust
let mut registry = LoraRegistry::new();
registry.add_search_path("~/.catsith/loras");

// Scan for LoRAs
let found = registry.scan();

// Load and register
for path in found {
    let lora = loader.load_from_dir(&path)?;
    registry.register(RegistryEntry::new(lora.manifest, path));
}

// Search
let anime_loras = registry.search("anime");
let aesthetic_loras = registry.list_by_category(LoraCategory::Aesthetic);
```

## Best Practices

1. **Keep ranks reasonable** (8-64) for balance of quality/size
2. **Test with multiple scenes** before publishing
3. **Include preview images** for discoverability
4. **Use semantic tags** for searchability
5. **Document compatibility** clearly
6. **Version your LoRAs** when updating
