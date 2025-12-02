//! Hardware capability detection
//!
//! Detects available rendering backends and hardware capabilities
//! to determine optimal quality settings.

use crate::intent::QualityTier;
use serde::{Deserialize, Serialize};

/// Detected rendering capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderCapabilities {
    /// CPU information
    pub cpu: CpuCapabilities,

    /// GPU information (if available)
    pub gpu: Option<GpuCapabilities>,

    /// Recommended quality tier based on hardware
    pub recommended_tier: QualityTier,

    /// Available rendering backends
    pub backends: Vec<BackendType>,
}

impl RenderCapabilities {
    /// Detect current system capabilities
    pub fn detect() -> Self {
        let cpu = CpuCapabilities::detect();
        let gpu = GpuCapabilities::detect();

        let recommended_tier = Self::calculate_recommended_tier(&cpu, &gpu);
        let backends = Self::available_backends(&gpu);

        Self {
            cpu,
            gpu,
            recommended_tier,
            backends,
        }
    }

    /// Create minimal capabilities (CPU only, terminal)
    pub fn minimal() -> Self {
        Self {
            cpu: CpuCapabilities {
                cores: 1,
                threads: 1,
                has_avx2: false,
                has_avx512: false,
            },
            gpu: None,
            recommended_tier: QualityTier::Minimal,
            backends: vec![BackendType::Terminal],
        }
    }

    /// Calculate recommended quality tier
    fn calculate_recommended_tier(
        cpu: &CpuCapabilities,
        gpu: &Option<GpuCapabilities>,
    ) -> QualityTier {
        match gpu {
            Some(g) if g.vram_mb >= 16000 && g.has_tensor_cores => QualityTier::Ultra,
            Some(g) if g.vram_mb >= 8000 => QualityTier::High,
            Some(g) if g.vram_mb >= 4000 => QualityTier::Medium,
            Some(_) => QualityTier::Low,
            None if cpu.threads >= 8 && cpu.has_avx2 => QualityTier::Low,
            None => QualityTier::Minimal,
        }
    }

    /// Determine available backends
    fn available_backends(gpu: &Option<GpuCapabilities>) -> Vec<BackendType> {
        let mut backends = vec![BackendType::Terminal, BackendType::Raster];

        backends.push(BackendType::NeuralCpu);

        if gpu.is_some() {
            backends.push(BackendType::NeuralGpu);
        }

        backends
    }

    /// Check if we can run the given quality tier
    pub fn supports_tier(&self, tier: QualityTier) -> bool {
        match tier {
            QualityTier::Minimal => true,
            QualityTier::Low => true,
            QualityTier::Medium => self.gpu.is_some(),
            QualityTier::High => self.gpu.as_ref().is_some_and(|g| g.vram_mb >= 4000),
            QualityTier::Ultra => self.gpu.as_ref().is_some_and(|g| g.vram_mb >= 8000),
            QualityTier::Cinematic => self.gpu.as_ref().is_some_and(|g| g.vram_mb >= 16000),
        }
    }

    /// Get the best supported quality tier up to the requested tier
    pub fn best_supported_tier(&self, requested: QualityTier) -> QualityTier {
        let tiers = [
            QualityTier::Cinematic,
            QualityTier::Ultra,
            QualityTier::High,
            QualityTier::Medium,
            QualityTier::Low,
            QualityTier::Minimal,
        ];

        for tier in tiers {
            if tier <= requested && self.supports_tier(tier) {
                return tier;
            }
        }

        QualityTier::Minimal
    }
}

/// CPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCapabilities {
    /// Number of physical cores
    pub cores: u32,
    /// Number of logical threads
    pub threads: u32,
    /// AVX2 support (useful for SIMD operations)
    pub has_avx2: bool,
    /// AVX-512 support
    pub has_avx512: bool,
}

impl CpuCapabilities {
    /// Detect CPU capabilities
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                cores: std::thread::available_parallelism()
                    .map(|p| p.get() as u32 / 2)
                    .unwrap_or(1)
                    .max(1),
                threads: std::thread::available_parallelism()
                    .map(|p| p.get() as u32)
                    .unwrap_or(1),
                has_avx2: std::arch::is_x86_feature_detected!("avx2"),
                has_avx512: std::arch::is_x86_feature_detected!("avx512f"),
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                cores: std::thread::available_parallelism()
                    .map(|p| p.get() as u32 / 2)
                    .unwrap_or(1)
                    .max(1),
                threads: std::thread::available_parallelism()
                    .map(|p| p.get() as u32)
                    .unwrap_or(1),
                has_avx2: false,
                has_avx512: false,
            }
        }
    }
}

/// GPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    /// GPU name
    pub name: String,
    /// Vendor
    pub vendor: GpuVendor,
    /// VRAM in megabytes
    pub vram_mb: u32,
    /// CUDA compute capability (for NVIDIA)
    pub compute_capability: Option<(u32, u32)>,
    /// Has tensor cores (for ML acceleration)
    pub has_tensor_cores: bool,
    /// Has hardware ray tracing
    pub has_ray_tracing: bool,
}

impl GpuCapabilities {
    /// Detect GPU capabilities
    pub fn detect() -> Option<Self> {
        // In a real implementation, this would query the system for GPU info
        // using vulkan, CUDA, or platform-specific APIs.
        //
        // For now, return None (no GPU detected)
        // TODO: Implement actual GPU detection
        None
    }

    /// Create mock GPU capabilities for testing
    #[cfg(test)]
    pub fn mock_nvidia_rtx() -> Self {
        Self {
            name: "NVIDIA RTX 4090".to_string(),
            vendor: GpuVendor::Nvidia,
            vram_mb: 24576,
            compute_capability: Some((8, 9)),
            has_tensor_cores: true,
            has_ray_tracing: true,
        }
    }
}

/// GPU vendor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

/// Available rendering backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendType {
    /// Terminal/ASCII rendering
    Terminal,
    /// Traditional 2D rasterization
    Raster,
    /// Neural inference on CPU
    NeuralCpu,
    /// Neural inference on GPU
    NeuralGpu,
}

impl BackendType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Terminal => "Terminal",
            Self::Raster => "Raster",
            Self::NeuralCpu => "Neural (CPU)",
            Self::NeuralGpu => "Neural (GPU)",
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Terminal => "ASCII/Unicode character-based rendering for terminals",
            Self::Raster => "Traditional sprite-based 2D rendering",
            Self::NeuralCpu => "Neural network inference on CPU",
            Self::NeuralGpu => "Neural network inference on GPU",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_capabilities() {
        let caps = RenderCapabilities::minimal();
        assert!(caps.supports_tier(QualityTier::Minimal));
        assert!(!caps.supports_tier(QualityTier::Medium));
        assert!(caps.backends.contains(&BackendType::Terminal));
    }

    #[test]
    fn test_gpu_capabilities() {
        let gpu = GpuCapabilities::mock_nvidia_rtx();
        assert_eq!(gpu.vendor, GpuVendor::Nvidia);
        assert!(gpu.has_tensor_cores);
        assert!(gpu.vram_mb >= 16000);
    }

    #[test]
    fn test_tier_support_with_gpu() {
        let mut caps = RenderCapabilities::minimal();
        caps.gpu = Some(GpuCapabilities::mock_nvidia_rtx());

        assert!(caps.supports_tier(QualityTier::Cinematic));
        assert!(caps.supports_tier(QualityTier::Ultra));
        assert!(caps.supports_tier(QualityTier::High));
    }

    #[test]
    fn test_best_supported_tier() {
        let caps = RenderCapabilities::minimal();

        // Without GPU, should fall back to Low even if Ultra requested
        assert_eq!(
            caps.best_supported_tier(QualityTier::Ultra),
            QualityTier::Low
        );
        assert_eq!(
            caps.best_supported_tier(QualityTier::Minimal),
            QualityTier::Minimal
        );
    }

    #[test]
    fn test_cpu_detection() {
        let cpu = CpuCapabilities::detect();
        assert!(cpu.threads >= 1);
        assert!(cpu.cores >= 1);
    }
}
