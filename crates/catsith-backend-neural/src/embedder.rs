//! Text/description embedding
//!
//! Converts semantic descriptions to embedding vectors for neural rendering.

use thiserror::Error;

/// Embedding errors
#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("Model not loaded")]
    NotLoaded,

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Text embedder for semantic descriptions
pub struct TextEmbedder {
    /// Embedding dimension
    embedding_dim: usize,
    /// Model loaded flag
    loaded: bool,
}

impl TextEmbedder {
    /// Create a new text embedder
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            loaded: false,
        }
    }

    /// Load the model from a path
    pub fn load(&mut self, _model_path: &str) -> Result<(), EmbedderError> {
        // TODO: Load actual ONNX model
        self.loaded = true;
        Ok(())
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Embed a text description
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        if !self.loaded {
            // Return a simple hash-based embedding for testing
            return Ok(self.simple_embed(text));
        }

        // TODO: Run actual inference
        Ok(self.simple_embed(text))
    }

    /// Embed multiple descriptions
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Simple hash-based embedding for testing
    fn simple_embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.embedding_dim];

        // Create a deterministic embedding from the text
        for (i, byte) in text.bytes().enumerate() {
            let idx = i % self.embedding_dim;
            embedding[idx] += (byte as f32 - 128.0) / 128.0;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }
}

impl Default for TextEmbedder {
    fn default() -> Self {
        Self::new(512)
    }
}

/// Precomputed embeddings for common semantic descriptions
pub struct EmbeddingCache {
    cache: std::collections::HashMap<String, Vec<f32>>,
    max_size: usize,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size,
        }
    }

    /// Get cached embedding
    pub fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.cache.get(key)
    }

    /// Cache an embedding
    pub fn insert(&mut self, key: String, embedding: Vec<f32>) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: remove first entry
            if let Some(k) = self.cache.keys().next().cloned() {
                self.cache.remove(&k);
            }
        }
        self.cache.insert(key, embedding);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_embed() {
        let embedder = TextEmbedder::new(128);
        let embedding = embedder.embed("red fighter ship").unwrap();

        assert_eq!(embedding.len(), 128);

        // Should be normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_different_texts_different_embeddings() {
        let embedder = TextEmbedder::new(128);
        let e1 = embedder.embed("red fighter").unwrap();
        let e2 = embedder.embed("blue bomber").unwrap();

        // Should be different
        let diff: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.1);
    }

    #[test]
    fn test_embedding_cache() {
        let mut cache = EmbeddingCache::new(2);

        cache.insert("a".to_string(), vec![1.0, 2.0]);
        cache.insert("b".to_string(), vec![3.0, 4.0]);

        assert_eq!(cache.get("a"), Some(&vec![1.0, 2.0]));
        assert_eq!(cache.len(), 2);

        // Should evict when over capacity
        cache.insert("c".to_string(), vec![5.0, 6.0]);
        assert_eq!(cache.len(), 2);
    }
}
