//! Learnable univariate functions using B-spline basis
//!
//! Each univariate function is a weighted sum of B-spline basis functions.
//! The weights are learned during training.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

use crate::bspline::BSpline;
use crate::shaders::{UNIVARIATE, WEIGHT_UPDATE};
use crate::{GpuContext, Result};

/// GPU-compatible update parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct UpdateParams {
    pub gradient: f32,
    pub learning_rate: f32,
}

/// A learnable univariate function using B-spline basis
///
/// The function is represented as: f(x) = Σ wᵢ * Bᵢ(x)
/// where Bᵢ are B-spline basis functions and wᵢ are learnable weights.
pub struct UnivariateFunction {
    /// Underlying B-spline basis
    pub spline: BSpline,
    /// Learnable weights (one per basis function)
    pub weights: Vec<f32>,
    // GPU resources
    gpu: GpuContext,
    weights_buffer: Buffer,
    output_buffer: Buffer,
    eval_pipeline: ComputePipeline,
    eval_bind_group_layout: BindGroupLayout,
    update_pipeline: ComputePipeline,
    update_bind_group_layout: BindGroupLayout,
}

impl UnivariateFunction {
    /// Create a new univariate function with random weights
    pub fn new(gpu: GpuContext, range: Range<f32>, num_knots: usize, degree: usize) -> Result<Self> {
        let spline = BSpline::new(gpu.clone(), range, num_knots, degree)?;
        let num_weights = spline.num_basis;

        // Initialize weights with small random values (Xavier-like)
        let scale = (2.0 / num_weights as f32).sqrt() * 0.1;
        let weights: Vec<f32> = (0..num_weights)
            .map(|_| (rand::random::<f32>() - 0.5) * scale)
            .collect();

        // Create weights buffer
        let weights_buffer = gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Univariate Weights"),
            contents: bytemuck::cast_slice(&weights),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let output_buffer = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("Univariate Output"),
            size: std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Evaluation pipeline
        let eval_shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Univariate Eval Shader"),
            source: ShaderSource::Wgsl(UNIVARIATE.into()),
        });

        let eval_bind_group_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Univariate Eval Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let eval_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Univariate Eval Pipeline Layout"),
            bind_group_layouts: &[&eval_bind_group_layout],
            push_constant_ranges: &[],
        });

        let eval_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
            compilation_options: PipelineCompilationOptions {
                constants: &HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            label: Some("Univariate Eval Pipeline"),
            layout: Some(&eval_pipeline_layout),
            module: &eval_shader,
            entry_point: "main",
        });

        // Weight update pipeline
        let update_shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Weight Update Shader"),
            source: ShaderSource::Wgsl(WEIGHT_UPDATE.into()),
        });

        let update_bind_group_layout =
            gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Weight Update Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let update_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Weight Update Pipeline Layout"),
            bind_group_layouts: &[&update_bind_group_layout],
            push_constant_ranges: &[],
        });

        let update_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
            compilation_options: PipelineCompilationOptions {
                constants: &HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            label: Some("Weight Update Pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_shader,
            entry_point: "main",
        });

        Ok(Self {
            spline,
            weights,
            gpu,
            weights_buffer,
            output_buffer,
            eval_pipeline,
            eval_bind_group_layout,
            update_pipeline,
            update_bind_group_layout,
        })
    }

    /// Evaluate the function at point x (CPU)
    pub fn evaluate(&self, x: f32) -> f32 {
        let basis = self.spline.evaluate(x);
        basis
            .iter()
            .zip(self.weights.iter())
            .map(|(b, w)| b * w)
            .sum()
    }

    /// Evaluate the function at point x (GPU)
    pub async fn evaluate_gpu(&self, x: f32) -> f32 {
        // First evaluate basis functions
        let _basis = self.spline.evaluate_gpu(x).await;

        // Create bind group for evaluation
        let bind_group = self.gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Univariate Eval Bind Group"),
            layout: &self.eval_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.spline.output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.weights_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Univariate Eval Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Univariate Eval Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.eval_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Read back result
        let staging = self.gpu.device.create_buffer(&BufferDescriptor {
            label: Some("Univariate Staging"),
            size: std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &staging,
            0,
            std::mem::size_of::<f32>() as u64,
        );

        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.gpu.device.poll(Maintain::Wait);

        rx.recv_async().await.unwrap().unwrap();
        let data = slice.get_mapped_range();
        let result: f32 = *bytemuck::from_bytes(&data);

        drop(data);
        staging.unmap();

        result
    }

    /// Update weights using gradient descent (CPU)
    pub fn update(&mut self, x: f32, gradient: f32, learning_rate: f32) {
        let basis = self.spline.evaluate(x);
        for (i, b) in basis.iter().enumerate() {
            if i < self.weights.len() {
                self.weights[i] -= learning_rate * gradient * b;
            }
        }
    }

    /// Update weights using gradient descent (GPU)
    pub async fn update_gpu(&mut self, x: f32, gradient: f32, learning_rate: f32) {
        // Evaluate basis on GPU
        let basis = self.spline.evaluate_gpu(x).await;

        // Create basis buffer
        let basis_buffer = self.gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Update Basis"),
            contents: bytemuck::cast_slice(&basis),
            usage: BufferUsages::STORAGE,
        });

        let params = UpdateParams {
            gradient,
            learning_rate,
        };

        let params_buffer = self.gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Update Params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = self.gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Weight Update Bind Group"),
            layout: &self.update_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: basis_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.weights_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Weight Update Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Weight Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((self.weights.len() + 63) / 64) as u32, 1, 1);
        }

        // Read back updated weights
        let staging = self.gpu.device.create_buffer(&BufferDescriptor {
            label: Some("Weights Staging"),
            size: (self.weights.len() * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.weights_buffer,
            0,
            &staging,
            0,
            (self.weights.len() * std::mem::size_of::<f32>()) as u64,
        );

        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.gpu.device.poll(Maintain::Wait);

        rx.recv_async().await.unwrap().unwrap();
        let data = slice.get_mapped_range();
        self.weights = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();
    }

    /// Sync CPU weights to GPU buffer
    pub fn sync_to_gpu(&self) {
        self.gpu
            .queue
            .write_buffer(&self.weights_buffer, 0, bytemuck::cast_slice(&self.weights));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_univariate_creation() {
        let gpu = GpuContext::new().await.unwrap();
        let func = UnivariateFunction::new(gpu, -1.0..1.0, 8, 3).unwrap();

        // Should have correct number of weights
        assert_eq!(func.weights.len(), func.spline.num_basis);
        // Weights should be small (Xavier-like init)
        for w in &func.weights {
            assert!(w.abs() < 1.0, "Weight too large: {}", w);
        }
    }

    #[tokio::test]
    async fn test_univariate_evaluate_cpu() {
        let gpu = GpuContext::new().await.unwrap();
        let func = UnivariateFunction::new(gpu, -1.0..1.0, 8, 3).unwrap();

        // Evaluation should work at various points
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let result = func.evaluate(x);
            assert!(result.is_finite(), "Non-finite result at x={}", x);
        }
    }

    #[tokio::test]
    async fn test_univariate_cpu_gpu_match() {
        let gpu = GpuContext::new().await.unwrap();
        let func = UnivariateFunction::new(gpu, -1.0..1.0, 8, 3).unwrap();

        for x in [-0.8, -0.3, 0.0, 0.4, 0.9] {
            let cpu_result = func.evaluate(x);
            let gpu_result = func.evaluate_gpu(x).await;

            assert!(
                (cpu_result - gpu_result).abs() < 1e-4,
                "CPU/GPU mismatch at x={}: {} vs {}",
                x,
                cpu_result,
                gpu_result
            );
        }
    }

    #[tokio::test]
    async fn test_univariate_update_changes_weights() {
        let gpu = GpuContext::new().await.unwrap();
        let mut func = UnivariateFunction::new(gpu, -1.0..1.0, 8, 3).unwrap();

        let original_weights = func.weights.clone();

        // Apply an update
        func.update(0.5, 1.0, 0.1);

        // Weights should have changed
        let changed = func
            .weights
            .iter()
            .zip(original_weights.iter())
            .any(|(new, old)| (new - old).abs() > 1e-10);

        assert!(changed, "Weights should change after update");
    }

    #[tokio::test]
    async fn test_univariate_update_gpu_changes_weights() {
        let gpu = GpuContext::new().await.unwrap();
        let mut func = UnivariateFunction::new(gpu, -1.0..1.0, 8, 3).unwrap();

        let original_weights = func.weights.clone();

        // Apply a GPU update
        func.update_gpu(0.5, 1.0, 0.1).await;

        // Weights should have changed
        let changed = func
            .weights
            .iter()
            .zip(original_weights.iter())
            .any(|(new, old)| (new - old).abs() > 1e-10);

        assert!(changed, "Weights should change after GPU update");
    }

    #[tokio::test]
    async fn test_univariate_gradient_descent_direction() {
        let gpu = GpuContext::new().await.unwrap();
        let mut func = UnivariateFunction::new(gpu, -1.0..1.0, 8, 3).unwrap();

        // Set known weights
        for w in &mut func.weights {
            *w = 0.5;
        }
        func.sync_to_gpu();

        let x = 0.5;
        let before = func.evaluate(x);

        // Positive gradient should decrease output (gradient descent)
        func.update(x, 1.0, 0.1);

        let after = func.evaluate(x);

        // Output should decrease when gradient is positive
        assert!(
            after < before,
            "Gradient descent should decrease output: {} -> {}",
            before,
            after
        );
    }

    #[tokio::test]
    async fn test_univariate_sync_to_gpu() {
        let gpu = GpuContext::new().await.unwrap();
        let mut func = UnivariateFunction::new(gpu, -1.0..1.0, 8, 3).unwrap();

        // Modify CPU weights
        for w in &mut func.weights {
            *w = 1.0;
        }

        // Sync to GPU
        func.sync_to_gpu();

        // GPU evaluation should reflect new weights
        let result = func.evaluate_gpu(0.5).await;

        // With all weights = 1.0, output should equal sum of basis functions = 1.0
        assert!(
            (result - 1.0).abs() < 0.1,
            "After sync, GPU should use new weights: {}",
            result
        );
    }
}
