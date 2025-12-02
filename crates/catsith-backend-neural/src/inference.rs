//! Neural inference engine
//!
//! Handles model loading and inference execution.

use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// Inference errors
#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Neural inference engine
pub struct InferenceEngine {
    /// Loaded models by name
    models: HashMap<String, LoadedModel>,
    /// Model search paths
    search_paths: Vec<PathBuf>,
    /// Device to use (CPU/GPU)
    device: Device,
}

/// Loaded model
pub struct LoadedModel {
    /// Model name
    pub name: String,
    /// Input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Model size in bytes
    pub size_bytes: usize,
}

impl LoadedModel {
    /// Create a dummy model for testing
    pub fn dummy(name: &str) -> Self {
        Self {
            name: name.to_string(),
            input_shapes: vec![vec![1, 3, 64, 64]],
            output_shapes: vec![vec![1, 3, 64, 64]],
            size_bytes: 1024 * 1024,
        }
    }
}

/// Inference device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(u32),
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            search_paths: vec![PathBuf::from("models"), PathBuf::from("~/.catsith/models")],
            device: Device::Cpu,
        }
    }

    /// Add a model search path
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }

    /// Set the device
    pub fn set_device(&mut self, device: Device) {
        self.device = device;
    }

    /// Get the current device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Load a model by name
    pub fn load_model(&mut self, name: &str) -> Result<(), InferenceError> {
        // TODO: Search for model file and load with tract

        // For now, create a dummy model
        let model = LoadedModel::dummy(name);
        self.models.insert(name.to_string(), model);
        Ok(())
    }

    /// Check if a model is loaded
    pub fn is_loaded(&self, name: &str) -> bool {
        self.models.contains_key(name)
    }

    /// Get loaded model info
    pub fn get_model(&self, name: &str) -> Option<&LoadedModel> {
        self.models.get(name)
    }

    /// Unload a model
    pub fn unload_model(&mut self, name: &str) -> bool {
        self.models.remove(name).is_some()
    }

    /// Run inference
    pub fn run(
        &self,
        model_name: &str,
        _inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, InferenceError> {
        let _model = self
            .models
            .get(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;

        // TODO: Run actual inference with tract

        // For now, return dummy output matching expected shape
        let output = Tensor::zeros(vec![1, 3, 64, 64]);
        Ok(vec![output])
    }

    /// Get total memory used by loaded models
    pub fn memory_usage(&self) -> usize {
        self.models.values().map(|m| m.size_bytes).sum()
    }

    /// List loaded models
    pub fn loaded_models(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Tensor data
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Shape
    pub shape: Vec<usize>,
    /// Data
    pub data: Vec<f32>,
}

impl Tensor {
    /// Create a new tensor
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data }
    }

    /// Create a zeros tensor
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    /// Create a random tensor (for testing)
    pub fn random(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1) % 1.0).collect();
        Self { shape, data }
    }

    /// Get total elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reshape (must have same total elements)
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), InferenceError> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err(InferenceError::ShapeMismatch {
                expected: new_shape,
                actual: self.shape.clone(),
            });
        }
        self.shape = new_shape;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_engine() {
        let mut engine = InferenceEngine::new();

        engine.load_model("test_model").unwrap();
        assert!(engine.is_loaded("test_model"));
        assert!(!engine.is_loaded("other_model"));

        let models = engine.loaded_models();
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_tensor() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.len(), 24);
        assert_eq!(t.shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_tensor_reshape() {
        let mut t = Tensor::zeros(vec![2, 3, 4]);
        t.reshape(vec![4, 6]).unwrap();
        assert_eq!(t.shape, vec![4, 6]);

        // Invalid reshape should fail
        let result = t.reshape(vec![5, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference() {
        let mut engine = InferenceEngine::new();
        engine.load_model("test").unwrap();

        let input = Tensor::random(vec![1, 3, 64, 64]);
        let outputs = engine.run("test", vec![input]).unwrap();

        assert_eq!(outputs.len(), 1);
    }
}
