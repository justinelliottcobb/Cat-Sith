//! LoRA registry
//!
//! Manages a collection of available LoRAs.

use crate::manifest::{LoraCategory, LoraManifest};
use std::collections::HashMap;
use std::path::PathBuf;

/// A registered LoRA entry
#[derive(Debug, Clone)]
pub struct RegistryEntry {
    /// The manifest
    pub manifest: LoraManifest,
    /// Path to the LoRA directory
    pub path: PathBuf,
    /// Whether the LoRA is loaded
    pub loaded: bool,
    /// Last used timestamp
    pub last_used: Option<u64>,
}

impl RegistryEntry {
    /// Create a new registry entry
    pub fn new(manifest: LoraManifest, path: PathBuf) -> Self {
        Self {
            manifest,
            path,
            loaded: false,
            last_used: None,
        }
    }

    /// Get the content hash
    pub fn hash(&self) -> [u8; 32] {
        self.manifest.hash
    }

    /// Get the name
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Get the category
    pub fn category(&self) -> LoraCategory {
        self.manifest.category
    }
}

/// LoRA registry
pub struct LoraRegistry {
    /// Registered LoRAs by hash
    entries: HashMap<[u8; 32], RegistryEntry>,
    /// Name to hash mapping for convenience
    name_index: HashMap<String, [u8; 32]>,
    /// Search paths for LoRAs
    search_paths: Vec<PathBuf>,
}

impl LoraRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            name_index: HashMap::new(),
            search_paths: vec![
                PathBuf::from("loras"),
                dirs::data_local_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("catsith")
                    .join("loras"),
            ],
        }
    }

    /// Add a search path
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }

    /// Get search paths
    pub fn search_paths(&self) -> &[PathBuf] {
        &self.search_paths
    }

    /// Register a LoRA
    pub fn register(&mut self, entry: RegistryEntry) {
        let hash = entry.hash();
        let name = entry.name().to_string();

        self.name_index.insert(name, hash);
        self.entries.insert(hash, entry);
    }

    /// Unregister a LoRA by hash
    pub fn unregister(&mut self, hash: &[u8; 32]) -> Option<RegistryEntry> {
        if let Some(entry) = self.entries.remove(hash) {
            self.name_index.remove(entry.name());
            Some(entry)
        } else {
            None
        }
    }

    /// Get a LoRA by hash
    pub fn get(&self, hash: &[u8; 32]) -> Option<&RegistryEntry> {
        self.entries.get(hash)
    }

    /// Get a LoRA by name
    pub fn get_by_name(&self, name: &str) -> Option<&RegistryEntry> {
        self.name_index
            .get(name)
            .and_then(|hash| self.entries.get(hash))
    }

    /// Get mutable LoRA by hash
    pub fn get_mut(&mut self, hash: &[u8; 32]) -> Option<&mut RegistryEntry> {
        self.entries.get_mut(hash)
    }

    /// Check if a LoRA is registered
    pub fn contains(&self, hash: &[u8; 32]) -> bool {
        self.entries.contains_key(hash)
    }

    /// Check if a name is registered
    pub fn contains_name(&self, name: &str) -> bool {
        self.name_index.contains_key(name)
    }

    /// List all registered LoRAs
    pub fn list(&self) -> Vec<&RegistryEntry> {
        self.entries.values().collect()
    }

    /// List LoRAs by category
    pub fn list_by_category(&self, category: LoraCategory) -> Vec<&RegistryEntry> {
        self.entries
            .values()
            .filter(|e| e.category() == category)
            .collect()
    }

    /// Search LoRAs by name (partial match)
    pub fn search(&self, query: &str) -> Vec<&RegistryEntry> {
        let query_lower = query.to_lowercase();
        self.entries
            .values()
            .filter(|e| e.name().to_lowercase().contains(&query_lower))
            .collect()
    }

    /// Search LoRAs by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<&RegistryEntry> {
        let tag_lower = tag.to_lowercase();
        self.entries
            .values()
            .filter(|e| {
                e.manifest
                    .tags
                    .iter()
                    .any(|t| t.to_lowercase() == tag_lower)
            })
            .collect()
    }

    /// Get number of registered LoRAs
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.name_index.clear();
    }

    /// Scan search paths for LoRAs
    pub fn scan(&mut self) -> Vec<PathBuf> {
        let mut found = Vec::new();

        for search_path in &self.search_paths.clone() {
            if let Ok(entries) = std::fs::read_dir(search_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let manifest_path = path.join("manifest.json");
                        if manifest_path.exists() {
                            found.push(path);
                        }
                    }
                }
            }
        }

        found
    }
}

impl Default for LoraRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::WeightInfo;

    fn test_entry(name: &str, category: LoraCategory) -> RegistryEntry {
        let mut manifest = LoraManifest::new(name, category).with_weights(WeightInfo::new(
            "weights.bin",
            1024,
            16,
        ));
        manifest.compute_hash(name.as_bytes());
        RegistryEntry::new(manifest, PathBuf::from("/test"))
    }

    #[test]
    fn test_registry_register() {
        let mut registry = LoraRegistry::new();

        let entry = test_entry("Test LoRA", LoraCategory::Aesthetic);
        let hash = entry.hash();

        registry.register(entry);

        assert!(registry.contains(&hash));
        assert!(registry.contains_name("Test LoRA"));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_search() {
        let mut registry = LoraRegistry::new();

        registry.register(test_entry("Anime Style", LoraCategory::Aesthetic));
        registry.register(test_entry("Pixel Art", LoraCategory::Aesthetic));
        registry.register(test_entry("Explosion FX", LoraCategory::Effects));

        let results = registry.search("style");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name(), "Anime Style");
    }

    #[test]
    fn test_registry_by_category() {
        let mut registry = LoraRegistry::new();

        registry.register(test_entry("Style 1", LoraCategory::Aesthetic));
        registry.register(test_entry("Style 2", LoraCategory::Aesthetic));
        registry.register(test_entry("Effect 1", LoraCategory::Effects));

        let aesthetic = registry.list_by_category(LoraCategory::Aesthetic);
        assert_eq!(aesthetic.len(), 2);

        let effects = registry.list_by_category(LoraCategory::Effects);
        assert_eq!(effects.len(), 1);
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = LoraRegistry::new();

        let entry = test_entry("Test", LoraCategory::Aesthetic);
        let hash = entry.hash();

        registry.register(entry);
        assert!(registry.contains(&hash));

        registry.unregister(&hash);
        assert!(!registry.contains(&hash));
        assert!(!registry.contains_name("Test"));
    }
}
