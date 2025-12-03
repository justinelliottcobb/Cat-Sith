//! KAN Layer implementation
//!
//! A single layer in a Kolmogorov-Arnold Network. Each layer consists of
//! multiple univariate functions and a projection matrix.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

use crate::shaders::KAN_LAYER;
use crate::univariate::UnivariateFunction;
use crate::{GpuContext, KanError, Result};

/// GPU-compatible layer parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LayerParams {
    pub input_dim: u32,
    pub output_dim: u32,
    pub num_functions: u32,
    pub _padding: u32,
}

/// A single layer in a KAN
///
/// The layer computes: output = Σ φᵢ(projection_i · input)
/// where φᵢ are learnable univariate functions.
pub struct KANLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    /// Univariate functions (learnable activations)
    pub functions: Vec<UnivariateFunction>,
    /// Projection matrix (input_dim × num_functions)
    pub projection: Vec<f32>,
    // GPU resources
    gpu: GpuContext,
    projection_buffer: Buffer,
    output_buffer: Buffer,
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl KANLayer {
    /// Create a new KAN layer
    ///
    /// # Arguments
    /// * `gpu` - Shared GPU context
    /// * `input_dim` - Input dimension
    /// * `output_dim` - Output dimension
    /// * `num_functions` - Number of univariate functions per output
    /// * `range` - Domain for B-splines
    /// * `num_knots` - Number of knots per B-spline
    /// * `degree` - B-spline degree
    pub fn new(
        gpu: GpuContext,
        input_dim: usize,
        output_dim: usize,
        num_functions: usize,
        range: Range<f32>,
        num_knots: usize,
        degree: usize,
    ) -> Result<Self> {
        // Create univariate functions
        let mut functions = Vec::with_capacity(num_functions);
        for _ in 0..num_functions {
            functions.push(UnivariateFunction::new(
                gpu.clone(),
                range.clone(),
                num_knots,
                degree,
            )?);
        }

        // Initialize projection matrix (Xavier initialization)
        let scale = (2.0 / (input_dim + num_functions) as f32).sqrt();
        let projection: Vec<f32> = (0..input_dim * num_functions)
            .map(|_| (rand::random::<f32>() - 0.5) * scale)
            .collect();

        // Create GPU buffers
        let projection_buffer = gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("KAN Layer Projection"),
            contents: bytemuck::cast_slice(&projection),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let output_buffer = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("KAN Layer Output"),
            size: (output_dim * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create shader and pipeline
        let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("KAN Layer Shader"),
            source: ShaderSource::Wgsl(KAN_LAYER.into()),
        });

        let bind_group_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("KAN Layer Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
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

        let pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("KAN Layer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
            compilation_options: PipelineCompilationOptions {
                constants: &HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            label: Some("KAN Layer Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            input_dim,
            output_dim,
            functions,
            projection,
            gpu,
            projection_buffer,
            output_buffer,
            compute_pipeline,
            bind_group_layout,
        })
    }

    /// Forward pass through the layer (CPU)
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim {
            return Err(KanError::DimensionMismatch {
                expected: self.input_dim,
                actual: input.len(),
            });
        }

        let num_functions = self.functions.len();
        let mut output = vec![0.0; self.output_dim];

        for o in 0..self.output_dim {
            for (f_idx, func) in self.functions.iter().enumerate() {
                // Compute projection
                let mut proj = 0.0;
                for i in 0..self.input_dim {
                    proj += input[i] * self.projection[i * num_functions + f_idx];
                }

                // Evaluate univariate function
                output[o] += func.evaluate(proj);
            }
        }

        Ok(output)
    }

    /// Forward pass through the layer (GPU)
    ///
    /// Note: This currently uses a simplified GPU kernel that doesn't fully
    /// integrate the univariate function evaluation. Full GPU integration
    /// is a TODO for better performance.
    pub async fn forward_gpu(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim {
            return Err(KanError::DimensionMismatch {
                expected: self.input_dim,
                actual: input.len(),
            });
        }

        let input_buffer = self.gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("KAN Layer Input"),
            contents: bytemuck::cast_slice(input),
            usage: BufferUsages::STORAGE,
        });

        let params = LayerParams {
            input_dim: self.input_dim as u32,
            output_dim: self.output_dim as u32,
            num_functions: self.functions.len() as u32,
            _padding: 0,
        };

        let params_buffer = self.gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("KAN Layer Params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = self.gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("KAN Layer Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.projection_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("KAN Layer Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("KAN Layer Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((self.output_dim + 63) / 64) as u32, 1, 1);
        }

        // Read back results
        let staging = self.gpu.device.create_buffer(&BufferDescriptor {
            label: Some("KAN Layer Staging"),
            size: (self.output_dim * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &staging,
            0,
            (self.output_dim * std::mem::size_of::<f32>()) as u64,
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
        let output: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();

        Ok(output)
    }

    /// Backward pass (CPU)
    pub fn backward(&mut self, input: &[f32], output_grad: &[f32], learning_rate: f32) {
        let num_functions = self.functions.len();

        for o in 0..self.output_dim {
            for (f_idx, func) in self.functions.iter_mut().enumerate() {
                // Compute projection
                let mut proj = 0.0;
                for i in 0..self.input_dim {
                    proj += input[i] * self.projection[i * num_functions + f_idx];
                }

                // Update univariate function weights
                func.update(proj, output_grad[o], learning_rate);

                // Update projection weights
                for i in 0..self.input_dim {
                    let idx = i * num_functions + f_idx;
                    self.projection[idx] -= learning_rate * output_grad[o] * input[i];
                }
            }
        }

        // Sync projection to GPU
        self.gpu.queue.write_buffer(
            &self.projection_buffer,
            0,
            bytemuck::cast_slice(&self.projection),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_layer_creation() {
        let gpu = GpuContext::new().await.unwrap();
        let layer = KANLayer::new(gpu, 4, 2, 3, -1.0..1.0, 8, 3).unwrap();

        assert_eq!(layer.input_dim, 4);
        assert_eq!(layer.output_dim, 2);
        assert_eq!(layer.functions.len(), 3);
        assert_eq!(layer.projection.len(), 4 * 3); // input_dim * num_functions
    }

    #[tokio::test]
    async fn test_layer_forward_dimensions() {
        let gpu = GpuContext::new().await.unwrap();
        let layer = KANLayer::new(gpu, 4, 2, 3, -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.len(), 2);
    }

    #[tokio::test]
    async fn test_layer_forward_wrong_input_dim() {
        let gpu = GpuContext::new().await.unwrap();
        let layer = KANLayer::new(gpu, 4, 2, 3, -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.1, 0.2]; // Wrong size
        let result = layer.forward(&input);

        assert!(result.is_err());
        match result {
            Err(KanError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[tokio::test]
    async fn test_layer_forward_produces_finite_values() {
        let gpu = GpuContext::new().await.unwrap();
        let layer = KANLayer::new(gpu, 4, 2, 3, -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.5, -0.5, 0.3, -0.3];
        let output = layer.forward(&input).unwrap();

        for (i, val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Output {} is not finite: {}", i, val);
        }
    }

    #[tokio::test]
    async fn test_layer_gpu_forward_dimensions() {
        let gpu = GpuContext::new().await.unwrap();
        let layer = KANLayer::new(gpu, 4, 2, 3, -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = layer.forward_gpu(&input).await.unwrap();

        assert_eq!(output.len(), 2);
    }

    #[tokio::test]
    async fn test_layer_backward_changes_weights() {
        let gpu = GpuContext::new().await.unwrap();
        let mut layer = KANLayer::new(gpu, 4, 2, 3, -1.0..1.0, 8, 3).unwrap();

        let original_projection = layer.projection.clone();

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output_grad = vec![1.0, -1.0];

        layer.backward(&input, &output_grad, 0.1);

        // Projection should have changed
        let changed = layer
            .projection
            .iter()
            .zip(original_projection.iter())
            .any(|(new, old)| (new - old).abs() > 1e-10);

        assert!(changed, "Projection weights should change after backward");
    }

    #[tokio::test]
    async fn test_layer_deterministic_forward() {
        let gpu = GpuContext::new().await.unwrap();
        let layer = KANLayer::new(gpu, 4, 2, 3, -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.1, 0.2, 0.3, 0.4];

        let output1 = layer.forward(&input).unwrap();
        let output2 = layer.forward(&input).unwrap();

        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Forward pass should be deterministic"
            );
        }
    }

    #[tokio::test]
    async fn test_layer_different_inputs_different_outputs() {
        let gpu = GpuContext::new().await.unwrap();
        let layer = KANLayer::new(gpu, 4, 2, 3, -1.0..1.0, 8, 3).unwrap();

        let input1 = vec![0.1, 0.2, 0.3, 0.4];
        let input2 = vec![0.9, 0.8, 0.7, 0.6];

        let output1 = layer.forward(&input1).unwrap();
        let output2 = layer.forward(&input2).unwrap();

        // Outputs should differ
        let different = output1
            .iter()
            .zip(output2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(
            different,
            "Different inputs should produce different outputs"
        );
    }
}
