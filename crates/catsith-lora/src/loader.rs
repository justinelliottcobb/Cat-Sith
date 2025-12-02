//! LoRA loading from disk/network

use crate::manifest::{LoraManifest, ManifestError};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Loader errors
#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("Manifest not found: {0}")]
    ManifestNotFound(PathBuf),

    #[error("Weights not found: {0}")]
    WeightsNotFound(PathBuf),

    #[error("Invalid manifest: {0}")]
    InvalidManifest(#[from] ManifestError),

    #[error("Hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Network error: {0}")]
    Network(String),
}

/// Loader configuration
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Verify hashes on load
    pub verify_hashes: bool,
    /// Maximum file size in bytes
    pub max_file_size: u64,
    /// Allow loading from network
    pub allow_network: bool,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            verify_hashes: true,
            max_file_size: 1024 * 1024 * 1024, // 1GB
            allow_network: false,
            cache_dir: None,
        }
    }
}

/// Loaded LoRA data
#[derive(Debug)]
pub struct LoadedLora {
    /// The manifest
    pub manifest: LoraManifest,
    /// Weight data
    pub weights: Vec<u8>,
    /// Source path
    pub source: PathBuf,
}

impl LoadedLora {
    /// Get the content hash
    pub fn hash(&self) -> [u8; 32] {
        self.manifest.hash
    }

    /// Get the name
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Verify the loaded weights match the manifest hash
    pub fn verify(&self) -> bool {
        let hash = *blake3::hash(&self.weights).as_bytes();
        hash == self.manifest.weights.hash
    }
}

/// LoRA loader
pub struct LoraLoader {
    config: LoaderConfig,
}

impl LoraLoader {
    /// Create a new loader
    pub fn new(config: LoaderConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    pub fn config(&self) -> &LoaderConfig {
        &self.config
    }

    /// Load a LoRA from a directory
    pub fn load_from_dir(&self, path: impl AsRef<Path>) -> Result<LoadedLora, LoaderError> {
        let path = path.as_ref();

        // Load manifest
        let manifest_path = path.join("manifest.json");
        let manifest = self.load_manifest(&manifest_path)?;

        // Load weights
        let weights_path = path.join(&manifest.weights.path);
        let weights = self.load_weights(&weights_path, &manifest)?;

        Ok(LoadedLora {
            manifest,
            weights,
            source: path.to_path_buf(),
        })
    }

    /// Load manifest from file
    pub fn load_manifest(&self, path: impl AsRef<Path>) -> Result<LoraManifest, LoaderError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(LoaderError::ManifestNotFound(path.to_path_buf()));
        }

        let content = std::fs::read_to_string(path)?;
        let manifest: LoraManifest =
            serde_json::from_str(&content).map_err(|e| LoaderError::Parse(e.to_string()))?;

        manifest.validate()?;

        Ok(manifest)
    }

    /// Load weights from file
    fn load_weights(
        &self,
        path: impl AsRef<Path>,
        manifest: &LoraManifest,
    ) -> Result<Vec<u8>, LoaderError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(LoaderError::WeightsNotFound(path.to_path_buf()));
        }

        // Check file size
        let metadata = std::fs::metadata(path)?;
        if metadata.len() > self.config.max_file_size {
            return Err(LoaderError::Io(std::io::Error::other(format!(
                "File too large: {} > {}",
                metadata.len(),
                self.config.max_file_size
            ))));
        }

        let weights = std::fs::read(path)?;

        // Verify hash if enabled
        if self.config.verify_hashes {
            let hash = *blake3::hash(&weights).as_bytes();
            if hash != manifest.weights.hash {
                return Err(LoaderError::HashMismatch {
                    expected: hex_encode(&manifest.weights.hash),
                    actual: hex_encode(&hash),
                });
            }
        }

        Ok(weights)
    }

    /// Save a LoRA to a directory
    pub fn save_to_dir(
        &self,
        lora: &LoadedLora,
        path: impl AsRef<Path>,
    ) -> Result<(), LoaderError> {
        let path = path.as_ref();

        // Create directory
        std::fs::create_dir_all(path)?;

        // Save manifest
        let manifest_json = serde_json::to_string_pretty(&lora.manifest)
            .map_err(|e| LoaderError::Parse(e.to_string()))?;
        std::fs::write(path.join("manifest.json"), manifest_json)?;

        // Save weights
        std::fs::write(path.join(&lora.manifest.weights.path), &lora.weights)?;

        Ok(())
    }
}

impl Default for LoraLoader {
    fn default() -> Self {
        Self::new(LoaderConfig::default())
    }
}

/// Encode bytes as hex string
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{LoraCategory, WeightInfo};
    use tempfile::TempDir;

    fn create_test_lora(dir: &Path) -> LoraManifest {
        let weights = vec![1u8, 2, 3, 4];
        let weight_hash = *blake3::hash(&weights).as_bytes();

        let manifest = LoraManifest::new("Test LoRA", LoraCategory::Aesthetic)
            .with_description("A test LoRA")
            .with_weights(
                WeightInfo::new("weights.bin", weights.len() as u64, 16).with_hash(weight_hash),
            );

        // Write manifest
        let manifest_json = serde_json::to_string_pretty(&manifest).unwrap();
        std::fs::write(dir.join("manifest.json"), manifest_json).unwrap();

        // Write weights
        std::fs::write(dir.join("weights.bin"), &weights).unwrap();

        manifest
    }

    #[test]
    fn test_load_from_dir() {
        let temp_dir = TempDir::new().unwrap();
        create_test_lora(temp_dir.path());

        let loader = LoraLoader::default();
        let loaded = loader.load_from_dir(temp_dir.path()).unwrap();

        assert_eq!(loaded.name(), "Test LoRA");
        assert_eq!(loaded.weights, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_hash_verification() {
        let temp_dir = TempDir::new().unwrap();
        create_test_lora(temp_dir.path());

        // Corrupt the weights
        std::fs::write(temp_dir.path().join("weights.bin"), [9, 9, 9, 9]).unwrap();

        let loader = LoraLoader::new(LoaderConfig {
            verify_hashes: true,
            ..Default::default()
        });

        let result = loader.load_from_dir(temp_dir.path());
        assert!(matches!(result, Err(LoaderError::HashMismatch { .. })));
    }

    #[test]
    fn test_skip_hash_verification() {
        let temp_dir = TempDir::new().unwrap();
        create_test_lora(temp_dir.path());

        // Corrupt the weights
        std::fs::write(temp_dir.path().join("weights.bin"), [9, 9, 9, 9]).unwrap();

        let loader = LoraLoader::new(LoaderConfig {
            verify_hashes: false,
            ..Default::default()
        });

        let result = loader.load_from_dir(temp_dir.path());
        assert!(result.is_ok());
    }
}
