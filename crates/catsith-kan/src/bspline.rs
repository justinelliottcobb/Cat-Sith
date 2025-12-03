//! B-spline basis functions for KAN
//!
//! B-splines provide smooth, locally-supported basis functions that form the
//! foundation of KAN's learnable univariate functions. This implementation
//! supports both CPU and GPU evaluation.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

use crate::shaders::BSPLINE;
use crate::{GpuContext, Result};

/// GPU-compatible B-spline parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BSplineParams {
    pub x: f32,
    pub degree: u32,
}

/// B-spline basis function evaluator
///
/// Uses Cox-de Boor recursion for stable evaluation of B-spline basis
/// functions. Supports both CPU and GPU computation.
pub struct BSpline {
    /// Knot vector (includes padding knots at boundaries)
    pub knots: Vec<f32>,
    /// Polynomial degree (typically 3 for cubic B-splines)
    pub degree: usize,
    /// Number of basis functions
    pub num_basis: usize,
    // GPU resources
    gpu: GpuContext,
    knots_buffer: Buffer,
    /// Output buffer containing evaluated basis functions (exposed for chaining)
    pub output_buffer: Buffer,
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl BSpline {
    /// Create a new B-spline with uniformly spaced knots
    ///
    /// # Arguments
    /// * `gpu` - Shared GPU context
    /// * `range` - Domain of the B-spline
    /// * `num_knots` - Number of interior knots
    /// * `degree` - Polynomial degree (typically 3)
    pub fn new(gpu: GpuContext, range: Range<f32>, num_knots: usize, degree: usize) -> Result<Self> {
        // Build knot vector with boundary padding
        let mut knots = Vec::with_capacity(num_knots + 2 * degree);

        // Pad start with repeated boundary knot
        for _ in 0..degree {
            knots.push(range.start);
        }

        // Interior knots uniformly spaced
        let step = (range.end - range.start) / (num_knots as f32 - 1.0);
        for i in 0..num_knots {
            knots.push(range.start + i as f32 * step);
        }

        // Pad end with repeated boundary knot
        for _ in 0..degree {
            knots.push(range.end);
        }

        let num_basis = knots.len() - degree - 1;

        // Create GPU resources
        let knots_buffer = gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("BSpline Knots"),
            contents: bytemuck::cast_slice(&knots),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let output_buffer = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("BSpline Output"),
            size: (num_basis * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("BSpline Shader"),
            source: ShaderSource::Wgsl(BSPLINE.into()),
        });

        let bind_group_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BSpline Bind Group Layout"),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("BSpline Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
            compilation_options: PipelineCompilationOptions {
                constants: &HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            label: Some("BSpline Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            knots,
            degree,
            num_basis,
            gpu,
            knots_buffer,
            output_buffer,
            compute_pipeline,
            bind_group_layout,
        })
    }

    /// Evaluate all basis functions at point x (GPU)
    pub async fn evaluate_gpu(&self, x: f32) -> Vec<f32> {
        let params = BSplineParams {
            x,
            degree: self.degree as u32,
        };

        let params_buffer = self.gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("BSpline Params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = self.gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("BSpline Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.knots_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("BSpline Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("BSpline Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per basis function
            pass.dispatch_workgroups(((self.num_basis + 63) / 64) as u32, 1, 1);
        }

        // Copy to staging buffer for readback
        let staging = self.gpu.device.create_buffer(&BufferDescriptor {
            label: Some("BSpline Staging"),
            size: (self.num_basis * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &staging,
            0,
            (self.num_basis * std::mem::size_of::<f32>()) as u64,
        );

        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let slice = staging.slice(..);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.gpu.device.poll(Maintain::Wait);

        rx.recv_async().await.unwrap().unwrap();
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();

        result
    }

    /// Evaluate all basis functions at point x (CPU)
    ///
    /// Uses Cox-de Boor recursion for numerically stable evaluation.
    pub fn evaluate(&self, x: f32) -> Vec<f32> {
        let mut basis = vec![0.0; self.num_basis];

        // Find the knot span containing x
        let mut span = self.degree;
        for i in self.degree..(self.knots.len() - 1) {
            if x < self.knots[i + 1] {
                span = i;
                break;
            }
        }

        // Handle right boundary
        if x >= self.knots[self.knots.len() - 1] {
            span = self.knots.len() - self.degree - 2;
        }

        // Cox-de Boor recursion
        let mut n = vec![0.0; self.degree + 1];
        n[0] = 1.0;

        for j in 1..=self.degree {
            let mut saved = 0.0;
            for r in 0..j {
                let left = self.knots[span + 1 + r - j];
                let right = self.knots[span + 1 + r];
                let alpha = if right != left {
                    (x - left) / (right - left)
                } else {
                    0.0
                };

                let temp = n[r];
                n[r] = saved + (1.0 - alpha) * temp;
                saved = alpha * temp;
            }
            n[j] = saved;
        }

        // Copy to output array
        for i in 0..=self.degree {
            let idx = span - self.degree + i;
            if idx < basis.len() {
                basis[idx] = n[i];
            }
        }

        basis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bspline_cpu_gpu_match() {
        let gpu = GpuContext::new().await.unwrap();
        let spline = BSpline::new(gpu, 0.0..1.0, 8, 3).unwrap();

        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let cpu = spline.evaluate(x);
            let gpu = spline.evaluate_gpu(x).await;

            for (c, g) in cpu.iter().zip(gpu.iter()) {
                assert!((c - g).abs() < 1e-5, "CPU/GPU mismatch at x={}: {} vs {}", x, c, g);
            }
        }
    }

    #[tokio::test]
    async fn test_bspline_partition_of_unity() {
        let gpu = GpuContext::new().await.unwrap();
        let spline = BSpline::new(gpu, 0.0..1.0, 8, 3).unwrap();

        // B-splines form a partition of unity: sum of all basis functions = 1
        for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let basis = spline.evaluate(x);
            let sum: f32 = basis.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "Not partition of unity at x={}: sum={}", x, sum);
        }
    }
}
