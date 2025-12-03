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
    async fn test_kan_forward() {
        let gpu = GpuContext::new().await.unwrap();
        let kan = KAN::new(gpu, &[2, 4, 1], &[4, 4], -1.0..1.0, 8, 3).unwrap();

        let input = vec![0.5, -0.3];
        let output = kan.forward(&input).unwrap();

        assert_eq!(output.len(), 1);
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

                epoch_loss += kan.train(&[x], &[y], 0.01)?;
            }

            epoch_loss /= 20.0;

            // Loss should generally decrease
            if epoch > 10 {
                assert!(epoch_loss < prev_loss * 1.5, "Loss not decreasing");
            }
            prev_loss = epoch_loss;
        }

        Ok::<(), KanError>(())
    }
}
