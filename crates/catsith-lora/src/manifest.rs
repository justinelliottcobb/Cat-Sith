//! LoRA manifest format
//!
//! Defines the structure of LoRA packages and their metadata.

use serde::{Deserialize, Serialize};

/// Current manifest schema version
pub const MANIFEST_VERSION: u32 = 1;

/// LoRA manifest - describes a LoRA package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraManifest {
    /// Schema version
    pub version: u32,

    /// Unique content hash (blake3)
    pub hash: [u8; 32],

    /// Human-readable name
    pub name: String,

    /// Description
    pub description: String,

    /// Creator information
    pub creator: CreatorInfo,

    /// What this LoRA affects
    pub category: LoraCategory,

    /// Compatible model architectures
    pub compatible_with: Vec<ModelArchitecture>,

    /// Weight file information
    pub weights: WeightInfo,

    /// Preview images (optional)
    pub previews: Vec<PreviewImage>,

    /// License
    pub license: String,

    /// Tags for searchability
    pub tags: Vec<String>,

    /// Creation timestamp (Unix seconds)
    pub created_at: u64,
}

impl LoraManifest {
    /// Create a new manifest
    pub fn new(name: impl Into<String>, category: LoraCategory) -> Self {
        Self {
            version: MANIFEST_VERSION,
            hash: [0; 32],
            name: name.into(),
            description: String::new(),
            creator: CreatorInfo::default(),
            category,
            compatible_with: Vec::new(),
            weights: WeightInfo::default(),
            previews: Vec::new(),
            license: "All Rights Reserved".to_string(),
            tags: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Set the description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the creator
    pub fn with_creator(mut self, creator: CreatorInfo) -> Self {
        self.creator = creator;
        self
    }

    /// Add a compatible model
    pub fn with_compatible_model(mut self, model: ModelArchitecture) -> Self {
        self.compatible_with.push(model);
        self
    }

    /// Set weight info
    pub fn with_weights(mut self, weights: WeightInfo) -> Self {
        self.weights = weights;
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set license
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = license.into();
        self
    }

    /// Compute and set the content hash
    pub fn compute_hash(&mut self, weight_data: &[u8]) {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.name.as_bytes());
        hasher.update(&self.category.to_bytes());
        hasher.update(weight_data);
        self.hash = *hasher.finalize().as_bytes();
    }

    /// Validate the manifest
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.name.is_empty() {
            return Err(ManifestError::MissingField("name".to_string()));
        }
        if self.weights.path.is_empty() {
            return Err(ManifestError::MissingField("weights.path".to_string()));
        }
        if self.weights.size_bytes == 0 {
            return Err(ManifestError::InvalidField(
                "weights.size_bytes".to_string(),
            ));
        }
        Ok(())
    }
}

/// Manifest errors
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid field value: {0}")]
    InvalidField(String),

    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Creator information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreatorInfo {
    /// Creator name
    pub name: String,
    /// Contact information (optional)
    pub contact: Option<String>,
    /// Website (optional)
    pub website: Option<String>,
}

impl CreatorInfo {
    /// Create new creator info
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            contact: None,
            website: None,
        }
    }

    /// Set contact
    pub fn with_contact(mut self, contact: impl Into<String>) -> Self {
        self.contact = Some(contact.into());
        self
    }

    /// Set website
    pub fn with_website(mut self, website: impl Into<String>) -> Self {
        self.website = Some(website.into());
        self
    }
}

/// LoRA category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoraCategory {
    /// Affects overall visual style
    Aesthetic,
    /// Affects specific entity rendering
    Entity,
    /// Affects visual effects
    Effects,
    /// Affects environment rendering
    Environment,
    /// Affects color grading
    Color,
}

impl LoraCategory {
    /// Convert to bytes for hashing
    pub fn to_bytes(&self) -> [u8; 1] {
        match self {
            Self::Aesthetic => [0],
            Self::Entity => [1],
            Self::Effects => [2],
            Self::Environment => [3],
            Self::Color => [4],
        }
    }

    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Aesthetic => "Aesthetic",
            Self::Entity => "Entity",
            Self::Effects => "Effects",
            Self::Environment => "Environment",
            Self::Color => "Color",
        }
    }
}

/// Model architecture that a LoRA is compatible with
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
}

impl ModelArchitecture {
    /// Create new model architecture
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

/// Weight file information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WeightInfo {
    /// Relative path to weights file
    pub path: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Hash of weights file
    pub hash: [u8; 32],
    /// Rank/dimension of the LoRA
    pub rank: u32,
    /// Alpha scaling factor
    pub alpha: f32,
}

impl WeightInfo {
    /// Create new weight info
    pub fn new(path: impl Into<String>, size_bytes: u64, rank: u32) -> Self {
        Self {
            path: path.into(),
            size_bytes,
            hash: [0; 32],
            rank,
            alpha: 1.0,
        }
    }

    /// Set the hash
    pub fn with_hash(mut self, hash: [u8; 32]) -> Self {
        self.hash = hash;
        self
    }

    /// Set alpha
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
}

/// Preview image reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewImage {
    /// Relative path to preview image
    pub path: String,
    /// Description of what the preview shows
    pub description: String,
}

impl PreviewImage {
    /// Create new preview image reference
    pub fn new(path: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            description: description.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_creation() {
        let manifest = LoraManifest::new("Test LoRA", LoraCategory::Aesthetic)
            .with_description("A test LoRA")
            .with_creator(CreatorInfo::new("Test Creator"))
            .with_weights(WeightInfo::new("weights.bin", 1024, 16))
            .with_tag("test")
            .with_license("MIT");

        assert_eq!(manifest.name, "Test LoRA");
        assert_eq!(manifest.category, LoraCategory::Aesthetic);
        assert!(manifest.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_manifest_validation() {
        let mut manifest = LoraManifest::new("", LoraCategory::Aesthetic);
        assert!(manifest.validate().is_err());

        manifest.name = "Valid".to_string();
        assert!(manifest.validate().is_err()); // Missing weights path

        manifest.weights = WeightInfo::new("weights.bin", 1024, 16);
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_manifest_serialization() {
        let manifest = LoraManifest::new("Test", LoraCategory::Entity)
            .with_weights(WeightInfo::new("weights.bin", 1024, 16));

        let json = serde_json::to_string(&manifest).unwrap();
        let decoded: LoraManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.name, "Test");
        assert_eq!(decoded.category, LoraCategory::Entity);
    }

    #[test]
    fn test_compute_hash() {
        let mut manifest = LoraManifest::new("Test", LoraCategory::Aesthetic);
        manifest.compute_hash(&[1, 2, 3, 4]);

        assert_ne!(manifest.hash, [0; 32]);

        // Same input should produce same hash
        let mut manifest2 = LoraManifest::new("Test", LoraCategory::Aesthetic);
        manifest2.compute_hash(&[1, 2, 3, 4]);

        assert_eq!(manifest.hash, manifest2.hash);
    }
}
