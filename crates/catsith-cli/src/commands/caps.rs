//! Capabilities command

use catsith_core::RenderCapabilities;

pub fn run() {
    let caps = RenderCapabilities::detect();

    println!("CatSith System Capabilities");
    println!("===========================\n");

    println!("CPU:");
    println!("  Cores:    {}", caps.cpu.cores);
    println!("  Threads:  {}", caps.cpu.threads);
    println!(
        "  AVX2:     {}",
        if caps.cpu.has_avx2 { "Yes" } else { "No" }
    );
    println!(
        "  AVX-512:  {}",
        if caps.cpu.has_avx512 { "Yes" } else { "No" }
    );

    println!();

    match &caps.gpu {
        Some(gpu) => {
            println!("GPU:");
            println!("  Name:     {}", gpu.name);
            println!("  Vendor:   {:?}", gpu.vendor);
            println!("  VRAM:     {} MB", gpu.vram_mb);
            println!(
                "  Tensor Cores: {}",
                if gpu.has_tensor_cores { "Yes" } else { "No" }
            );
            println!(
                "  Ray Tracing:  {}",
                if gpu.has_ray_tracing { "Yes" } else { "No" }
            );
        }
        None => {
            println!("GPU: Not detected");
        }
    }

    println!();
    println!("Available Backends:");
    for backend in &caps.backends {
        println!("  - {} ({})", backend.name(), backend.description());
    }

    println!();
    println!("Recommended Quality: {:?}", caps.recommended_tier);

    println!();
    println!("Supported Quality Tiers:");
    for tier in [
        catsith_core::QualityTier::Minimal,
        catsith_core::QualityTier::Low,
        catsith_core::QualityTier::Medium,
        catsith_core::QualityTier::High,
        catsith_core::QualityTier::Ultra,
        catsith_core::QualityTier::Cinematic,
    ] {
        let supported = caps.supports_tier(tier);
        let marker = if supported { "✓" } else { "✗" };
        println!("  {} {:?}", marker, tier);
    }
}
