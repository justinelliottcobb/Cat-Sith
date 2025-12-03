//! Shared GPU context for KAN operations
//!
//! This module provides a centralized GPU context that can be shared across
//! multiple KAN components, avoiding redundant device initialization.

use std::sync::Arc;
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions,
};

use crate::{KanError, Result};

/// Shared GPU context for KAN operations
///
/// This context holds the WGPU device and queue, which are shared across
/// all KAN layers and networks to avoid redundant GPU initialization.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl GpuContext {
    /// Create a new GPU context
    ///
    /// This initializes WGPU and requests a high-performance GPU device.
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(KanError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("CatSith KAN Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| KanError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Create a GPU context with custom limits
    pub async fn with_limits(limits: Limits) -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(KanError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("CatSith KAN Device"),
                    required_features: Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await
            .map_err(|e| KanError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &"<wgpu::Device>")
            .field("queue", &"<wgpu::Queue>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_context_creation() {
        let result = GpuContext::new().await;
        assert!(result.is_ok(), "Should be able to create GPU context");
    }

    #[tokio::test]
    async fn test_gpu_context_clone() {
        let gpu1 = GpuContext::new().await.unwrap();
        let gpu2 = gpu1.clone();

        // Both should reference the same device (Arc)
        assert!(Arc::ptr_eq(&gpu1.device, &gpu2.device));
        assert!(Arc::ptr_eq(&gpu1.queue, &gpu2.queue));
    }

    #[tokio::test]
    async fn test_gpu_context_with_default_limits() {
        let result = GpuContext::with_limits(Limits::default()).await;
        assert!(
            result.is_ok(),
            "Should be able to create GPU context with default limits"
        );
    }

    #[tokio::test]
    async fn test_gpu_context_with_downlevel_limits() {
        let result = GpuContext::with_limits(Limits::downlevel_defaults()).await;
        assert!(
            result.is_ok(),
            "Should be able to create GPU context with downlevel limits"
        );
    }

    #[tokio::test]
    async fn test_gpu_context_debug() {
        let gpu = GpuContext::new().await.unwrap();
        let debug_str = format!("{:?}", gpu);

        assert!(debug_str.contains("GpuContext"));
        assert!(debug_str.contains("device"));
        assert!(debug_str.contains("queue"));
    }

    #[tokio::test]
    async fn test_gpu_context_can_create_buffer() {
        use wgpu::*;

        let gpu = GpuContext::new().await.unwrap();

        // Verify we can actually use the device
        let buffer = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("Test Buffer"),
            size: 64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Buffer creation should succeed (no panic)
        drop(buffer);
    }
}
