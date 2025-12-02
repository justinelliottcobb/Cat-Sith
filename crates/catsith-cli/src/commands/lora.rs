//! LoRA management commands

use catsith_lora::{LoraLoader, LoraRegistry, LoraValidator};

pub fn list() {
    let mut registry = LoraRegistry::new();

    // Scan for LoRAs
    let found = registry.scan();

    println!("LoRA Search Paths:");
    for path in registry.search_paths() {
        println!("  - {}", path.display());
    }

    println!();

    if found.is_empty() {
        println!("No LoRAs found.");
        return;
    }

    println!("Found {} LoRA(s):", found.len());
    for path in found {
        println!("  - {}", path.display());
    }
}

pub fn info(name: &str) {
    let registry = LoraRegistry::new();

    match registry.get_by_name(name) {
        Some(entry) => {
            let m = &entry.manifest;
            println!("LoRA: {}", m.name);
            println!("==============================");
            println!("Description: {}", m.description);
            println!("Category:    {:?}", m.category);
            println!("Creator:     {}", m.creator.name);
            println!("License:     {}", m.license);
            println!();
            println!("Weights:");
            println!("  Path:  {}", m.weights.path);
            println!("  Size:  {} bytes", m.weights.size_bytes);
            println!("  Rank:  {}", m.weights.rank);
            println!("  Alpha: {}", m.weights.alpha);
            println!();
            println!("Tags: {}", m.tags.join(", "));
            println!();
            println!("Compatible Models:");
            for arch in &m.compatible_with {
                println!("  - {} v{}", arch.name, arch.version);
            }
        }
        None => {
            println!("LoRA '{}' not found.", name);
            println!("Use 'catsith lora list' to see available LoRAs.");
        }
    }
}

pub fn validate(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Validating LoRA at: {}", path);

    let loader = LoraLoader::default();
    let validator = LoraValidator::new();

    // Load the LoRA
    let loaded = match loader.load_from_dir(path) {
        Ok(l) => l,
        Err(e) => {
            println!("Failed to load LoRA: {}", e);
            return Ok(());
        }
    };

    // Validate
    let result = validator.validate(&loaded.manifest, &loaded.weights);

    println!();
    if result.valid {
        println!("✓ LoRA is valid");
    } else {
        println!("✗ LoRA validation failed");
        println!();
        println!("Errors:");
        for error in &result.errors {
            println!("  - {:?}", error);
        }
    }

    if result.has_warnings() {
        println!();
        println!("Warnings:");
        for warning in &result.warnings {
            println!("  - {:?}", warning);
        }
    }

    println!();
    println!("Manifest:");
    println!("  Name:     {}", loaded.manifest.name);
    println!("  Category: {:?}", loaded.manifest.category);
    println!("  Size:     {} bytes", loaded.weights.len());

    Ok(())
}
