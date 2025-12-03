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
