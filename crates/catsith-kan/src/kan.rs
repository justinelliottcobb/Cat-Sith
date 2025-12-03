//! Complete KAN network implementation
//!
//! Multi-layer Kolmogorov-Arnold Network with GPU acceleration.

use std::ops::Range;

use crate::kan_layer::KANLayer;
use crate::{GpuContext, KanError, Result};

/// A complete Kolmogorov-Arnold Network
///
/// Chains multiple KAN layers together for deep learning tasks.
pub struct KAN {
    /// Network layers
    pub layers: Vec<KANLayer>,
    /// Shared GPU context
    pub gpu: GpuContext,
}

impl KAN {
    /// Create a new KAN network
    ///
    /// # Arguments
    /// * `gpu` - Shared GPU context
    /// * `layer_dims` - Dimensions for each layer (input, hidden..., output)
    /// * `functions_per_layer` - Number of univariate functions per layer
    /// * `range` - Domain for B-splines (typically normalized input range)
    /// * `num_knots` - Number of knots per B-spline
    /// * `degree` - B-spline degree (typically 3 for cubic)
    ///
    /// # Example
    /// ```ignore
    /// // Create a network: 10 inputs -> 32 hidden -> 4 outputs
    /// let kan = KAN::new(
    ///     gpu,
    ///     &[10, 32, 4],
    ///     &[8, 8],  // 8 functions per layer
    ///     -1.0..1.0,
    ///     8,
    ///     3,
    /// )?;
    /// ```
    pub fn new(
        gpu: GpuContext,
        layer_dims: &[usize],
        functions_per_layer: &[usize],
        range: Range<f32>,
        num_knots: usize,
        degree: usize,
    ) -> Result<Self> {
        if layer_dims.len() < 2 {
            return Err(KanError::InvalidConfig(
                "Need at least input and output dimensions".into(),
            ));
        }

        if layer_dims.len() - 1 != functions_per_layer.len() {
            return Err(KanError::InvalidConfig(format!(
                "functions_per_layer length ({}) must equal number of layers ({})",
                functions_per_layer.len(),
                layer_dims.len() - 1
            )));
        }

        let mut layers = Vec::with_capacity(layer_dims.len() - 1);

        for i in 0..layer_dims.len() - 1 {
            layers.push(KANLayer::new(
                gpu.clone(),
                layer_dims[i],
                layer_dims[i + 1],
                functions_per_layer[i],
                range.clone(),
                num_knots,
                degree,
            )?);
        }

        Ok(Self { layers, gpu })
    }

    /// Forward pass through the network (CPU)
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut current = input.to_vec();

        for layer in &self.layers {
            current = layer.forward(&current)?;
        }

        Ok(current)
    }

    /// Forward pass through the network (GPU)
    pub async fn forward_gpu(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut current = input.to_vec();

        for layer in &self.layers {
            current = layer.forward_gpu(&current).await?;
        }

        Ok(current)
    }

    /// Train on a single example (CPU)
    ///
    /// Returns the MSE loss.
    pub fn train(&mut self, input: &[f32], target: &[f32], learning_rate: f32) -> Result<f32> {
        // Forward pass, storing activations
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());

        for layer in &self.layers {
            activations.push(layer.forward(activations.last().unwrap())?);
        }

        // Compute loss and output gradient
        let output = activations.last().unwrap();
        let mut loss = 0.0;
        let mut output_grad = Vec::with_capacity(output.len());

        for (o, t) in output.iter().zip(target.iter()) {
            let error = o - t;
            loss += 0.5 * error * error;
            output_grad.push(error);
        }

        // Backward pass
        for i in (0..self.layers.len()).rev() {
            let input_activation = &activations[i];
            self.layers[i].backward(input_activation, &output_grad, learning_rate);

            // Simplified gradient propagation (TODO: proper chain rule)
            if i > 0 {
                output_grad = vec![0.0; activations[i].len()];
            }
        }

        Ok(loss)
    }

    /// Train on a batch of examples (CPU)
    ///
    /// Returns the average MSE loss.
    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        learning_rate: f32,
    ) -> Result<f32> {
        if inputs.len() != targets.len() {
            return Err(KanError::InvalidConfig(
                "Inputs and targets must have same length".into(),
            ));
        }

        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            total_loss += self.train(input, target, learning_rate)?;
        }

        Ok(total_loss / inputs.len() as f32)
    }

    /// Get total parameter count
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;

        for layer in &self.layers {
            // Projection matrix
            count += layer.projection.len();

            // Univariate function weights
            for func in &layer.functions {
                count += func.weights.len();
            }
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kan_creation() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        assert_eq!(kan.layers.len(), 2); // 2 -> 4 -> 1 = 2 layers
    }

    #[tokio::test]
    async fn test_kan_creation_single_layer() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(gpu, &[3, 2], &[4], -1.0..1.0, 8, 3).unwrap();

        assert_eq!(kan.layers.len(), 1);
    }

    #[tokio::test]
    async fn test_kan_creation_invalid_dims() {
        let gpu = GpuContext::new().await.unwrap();

        // Only one dimension (no layers)
        let result = KAN::new(gpu, &[3], &[], -1.0..1.0, 8, 3);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_kan_creation_mismatched_functions() {
        let gpu = GpuContext::new().await.unwrap();

        // 2 layers but only 1 function count
        let result = KAN::new(gpu, &[2, 4, 1], &[4], -1.0..1.0, 8, 3);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_kan_forward() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.5, -0.3];
        let output = kan.forward(&input).unwrap();

        assert_eq!(output.len(), 1);
    }

    #[tokio::test]
    async fn test_kan_forward_gpu() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.5, -0.3];
        let output = kan.forward_gpu(&input).await.unwrap();

        assert_eq!(output.len(), 1);
    }

    #[tokio::test]
    async fn test_kan_forward_produces_finite() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(gpu, &[3, 8, 4, 2], &[4, 4, 4], -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.1, 0.2, 0.3];
        let output = kan.forward(&input).unwrap();

        for (i, val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Output {} is not finite: {}", i, val);
        }
    }

    #[tokio::test]
    async fn test_kan_forward_deterministic() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.5, -0.3];

        let output1 = kan.forward(&input).unwrap();
        let output2 = kan.forward(&input).unwrap();

        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a - b).abs() < 1e-10, "Forward should be deterministic");
        }
    }

    #[tokio::test]
    async fn test_kan_train() {
        let gpu = GpuContext::new().await.unwrap();
        let mut kan = KAN::new(gpu, &[1, 8, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        // Train on sin(x)
        let mut prev_loss = f32::MAX;
        for epoch in 0..100 {
            let mut epoch_loss = 0.0;

            for i in 0..20 {
                let x = (i as f32 / 20.0) * 2.0 - 1.0;
                let y = x.sin();

                epoch_loss += kan.train(&[x], &[y], 0.01).unwrap();
            }

            epoch_loss /= 20.0;

            // Loss should generally decrease
            if epoch > 10 {
                assert!(epoch_loss < prev_loss * 1.5, "Loss not decreasing");
            }
            prev_loss = epoch_loss;
        }
    }

    #[tokio::test]
    async fn test_kan_train_returns_loss() {
        let gpu = GpuContext::new().await.unwrap();
        let mut kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.5, -0.3];
        let target = vec![0.7];

        let loss = kan.train(&input, &target, 0.01).unwrap();

        assert!(loss >= 0.0, "Loss should be non-negative");
        assert!(loss.is_finite(), "Loss should be finite");
    }

    #[tokio::test]
    async fn test_kan_train_batch() {
        let gpu = GpuContext::new().await.unwrap();
        let mut kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        let inputs = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];

        let targets = vec![vec![0.5], vec![0.6], vec![0.7]];

        let loss = kan.train_batch(&inputs, &targets, 0.01).unwrap();

        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[tokio::test]
    async fn test_kan_train_batch_mismatched_lengths() {
        let gpu = GpuContext::new().await.unwrap();
        let mut kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        let inputs = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let targets = vec![vec![0.5]]; // Wrong length

        let result = kan.train_batch(&inputs, &targets, 0.01);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_kan_parameter_count() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        let count = kan.parameter_count();
        assert!(count > 0, "Should have parameters");
    }

    #[tokio::test]
    async fn test_kan_deeper_network() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(
            gpu,
            &[4, 8, 8, 4, 2],
            &[4, 4, 4, 4],
            -1.0..1.0,
            8,
            3,
        )
        .unwrap();

        assert_eq!(kan.layers.len(), 4);

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = kan.forward(&input).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[tokio::test]
    async fn test_kan_different_inputs_different_outputs() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        let input1 = vec![0.1, 0.2];
        let input2 = vec![0.9, 0.8];

        let output1 = kan.forward(&input1).unwrap();
        let output2 = kan.forward(&input2).unwrap();

        // Different inputs should produce different outputs
        let different = output1
            .iter()
            .zip(output2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(different, "Different inputs should produce different outputs");
    }
}
