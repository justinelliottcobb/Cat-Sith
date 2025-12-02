//! LoRA validation
//!
//! Validates LoRA integrity and compatibility.

use crate::manifest::LoraManifest;

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Is the LoRA valid?
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    /// Create a successful result
    pub fn success() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a failed result
    pub fn failure(errors: Vec<ValidationError>) -> Self {
        Self {
            valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, error: ValidationError) {
        self.valid = false;
        self.errors.push(error);
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Manifest is invalid
    InvalidManifest(String),
    /// Hash doesn't match
    HashMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// Unsupported format
    UnsupportedFormat(String),
    /// Weight dimensions don't match
    DimensionMismatch { expected: usize, actual: usize },
    /// Incompatible model
    IncompatibleModel(String),
}

/// Validation warning
#[derive(Debug, Clone)]
pub enum ValidationWarning {
    /// Missing preview images
    MissingPreviews,
    /// No tags defined
    NoTags,
    /// Large file size
    LargeFileSize(u64),
    /// Unknown model architecture
    UnknownArchitecture(String),
}

/// LoRA validator
pub struct LoraValidator {
    /// Known model architectures
    known_architectures: Vec<String>,
    /// Maximum recommended file size
    max_recommended_size: u64,
}

impl LoraValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            known_architectures: vec!["sprite_vae_v1".to_string(), "diffusion_v1".to_string()],
            max_recommended_size: 100 * 1024 * 1024, // 100MB
        }
    }

    /// Add a known architecture
    pub fn add_architecture(&mut self, name: impl Into<String>) {
        self.known_architectures.push(name.into());
    }

    /// Set maximum recommended size
    pub fn set_max_size(&mut self, size: u64) {
        self.max_recommended_size = size;
    }

    /// Validate a manifest
    pub fn validate_manifest(&self, manifest: &LoraManifest) -> ValidationResult {
        let mut result = ValidationResult::success();

        // Check required fields
        if manifest.name.is_empty() {
            result.add_error(ValidationError::InvalidManifest("Empty name".to_string()));
        }

        if manifest.weights.path.is_empty() {
            result.add_error(ValidationError::InvalidManifest(
                "Empty weights path".to_string(),
            ));
        }

        if manifest.weights.size_bytes == 0 {
            result.add_error(ValidationError::InvalidManifest(
                "Zero weight size".to_string(),
            ));
        }

        if manifest.weights.rank == 0 {
            result.add_error(ValidationError::InvalidManifest("Zero rank".to_string()));
        }

        // Check for warnings
        if manifest.previews.is_empty() {
            result.add_warning(ValidationWarning::MissingPreviews);
        }

        if manifest.tags.is_empty() {
            result.add_warning(ValidationWarning::NoTags);
        }

        if manifest.weights.size_bytes > self.max_recommended_size {
            result.add_warning(ValidationWarning::LargeFileSize(
                manifest.weights.size_bytes,
            ));
        }

        // Check architecture compatibility
        for arch in &manifest.compatible_with {
            if !self.known_architectures.contains(&arch.name) {
                result.add_warning(ValidationWarning::UnknownArchitecture(arch.name.clone()));
            }
        }

        result
    }

    /// Validate weight data
    pub fn validate_weights(&self, manifest: &LoraManifest, weights: &[u8]) -> ValidationResult {
        let mut result = ValidationResult::success();

        // Check size
        if weights.len() as u64 != manifest.weights.size_bytes {
            result.add_error(ValidationError::DimensionMismatch {
                expected: manifest.weights.size_bytes as usize,
                actual: weights.len(),
            });
        }

        // Check hash
        let hash = *blake3::hash(weights).as_bytes();
        if hash != manifest.weights.hash {
            result.add_error(ValidationError::HashMismatch {
                expected: manifest.weights.hash,
                actual: hash,
            });
        }

        result
    }

    /// Full validation
    pub fn validate(&self, manifest: &LoraManifest, weights: &[u8]) -> ValidationResult {
        let manifest_result = self.validate_manifest(manifest);
        if !manifest_result.valid {
            return manifest_result;
        }

        let mut weight_result = self.validate_weights(manifest, weights);

        // Combine warnings
        for warning in manifest_result.warnings {
            weight_result.add_warning(warning);
        }

        weight_result
    }
}

impl Default for LoraValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{LoraCategory, WeightInfo};

    #[test]
    fn test_validate_manifest_success() {
        let manifest = LoraManifest::new("Test", LoraCategory::Aesthetic)
            .with_weights(WeightInfo::new("weights.bin", 1024, 16))
            .with_tag("test");

        let validator = LoraValidator::new();
        let result = validator.validate_manifest(&manifest);

        assert!(result.valid);
    }

    #[test]
    fn test_validate_manifest_empty_name() {
        let manifest = LoraManifest::new("", LoraCategory::Aesthetic)
            .with_weights(WeightInfo::new("weights.bin", 1024, 16));

        let validator = LoraValidator::new();
        let result = validator.validate_manifest(&manifest);

        assert!(!result.valid);
    }

    #[test]
    fn test_validate_manifest_warnings() {
        let manifest = LoraManifest::new("Test", LoraCategory::Aesthetic)
            .with_weights(WeightInfo::new("weights.bin", 1024, 16));
        // No tags, no previews

        let validator = LoraValidator::new();
        let result = validator.validate_manifest(&manifest);

        assert!(result.valid);
        assert!(result.has_warnings());
    }

    #[test]
    fn test_validate_weights() {
        let weights = vec![1u8, 2, 3, 4];
        let hash = *blake3::hash(&weights).as_bytes();

        let manifest = LoraManifest::new("Test", LoraCategory::Aesthetic)
            .with_weights(WeightInfo::new("weights.bin", 4, 16).with_hash(hash));

        let validator = LoraValidator::new();
        let result = validator.validate_weights(&manifest, &weights);

        assert!(result.valid);
    }

    #[test]
    fn test_validate_weights_hash_mismatch() {
        let manifest = LoraManifest::new("Test", LoraCategory::Aesthetic)
            .with_weights(WeightInfo::new("weights.bin", 4, 16).with_hash([0; 32]));

        let validator = LoraValidator::new();
        let result = validator.validate_weights(&manifest, &[1, 2, 3, 4]);

        assert!(!result.valid);
    }
}
