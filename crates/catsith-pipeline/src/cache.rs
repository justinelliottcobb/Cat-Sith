//! Entity embedding cache
//!
//! Caches entity embeddings and rendered sprites to avoid redundant computation.

use catsith_core::EntityId;
use std::collections::HashMap;
use std::time::Instant;

/// Cached entity data
#[derive(Debug, Clone)]
pub struct CachedEntity {
    /// Entity embedding vector
    pub embedding: Option<Vec<f32>>,
    /// Rendered sprite data (RGBA)
    pub sprite: Option<CachedSprite>,
    /// Cache entry creation time
    pub created_at: Instant,
    /// Last access time
    pub last_accessed: Instant,
    /// Access count
    pub access_count: u64,
    /// Content hash (for invalidation)
    pub content_hash: [u8; 32],
}

/// Cached sprite data
#[derive(Debug, Clone)]
pub struct CachedSprite {
    /// Pixel data (RGBA)
    pub pixels: Vec<u8>,
    /// Width
    pub width: u32,
    /// Height
    pub height: u32,
}

impl CachedSprite {
    /// Estimate memory usage in bytes
    pub fn memory_size(&self) -> usize {
        self.pixels.len() + std::mem::size_of::<Self>()
    }
}

impl CachedEntity {
    /// Create a new cached entity
    pub fn new(content_hash: [u8; 32]) -> Self {
        let now = Instant::now();
        Self {
            embedding: None,
            sprite: None,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            content_hash,
        }
    }

    /// Mark as accessed
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Estimate memory usage in bytes
    pub fn memory_size(&self) -> usize {
        let embedding_size = self
            .embedding
            .as_ref()
            .map(|e| e.len() * std::mem::size_of::<f32>())
            .unwrap_or(0);

        let sprite_size = self.sprite.as_ref().map(|s| s.memory_size()).unwrap_or(0);

        embedding_size + sprite_size + std::mem::size_of::<Self>()
    }

    /// Age in seconds
    pub fn age_seconds(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64()
    }

    /// Time since last access in seconds
    pub fn idle_seconds(&self) -> f64 {
        self.last_accessed.elapsed().as_secs_f64()
    }
}

/// Entity embedding and sprite cache
pub struct EntityCache {
    /// Cached entities by ID
    entries: HashMap<EntityId, CachedEntity>,
    /// Maximum cache size in bytes
    max_size_bytes: usize,
    /// Current estimated size
    current_size_bytes: usize,
    /// Cache statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total lookups
    pub lookups: u64,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Evictions
    pub evictions: u64,
    /// Insertions
    pub insertions: u64,
}

impl CacheStats {
    /// Hit rate (0.0 - 1.0)
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.lookups as f64
        }
    }
}

impl EntityCache {
    /// Create a new cache with the given maximum size
    pub fn new(max_size_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_size_bytes,
            current_size_bytes: 0,
            stats: CacheStats::default(),
        }
    }

    /// Get a cached entity
    pub fn get(&mut self, entity_id: &EntityId) -> Option<&CachedEntity> {
        self.stats.lookups += 1;

        if self.entries.contains_key(entity_id) {
            self.stats.hits += 1;
            // Touch to update last_accessed
            if let Some(entry) = self.entries.get_mut(entity_id) {
                entry.touch();
            }
            self.entries.get(entity_id)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Get mutable cached entity
    pub fn get_mut(&mut self, entity_id: &EntityId) -> Option<&mut CachedEntity> {
        self.stats.lookups += 1;

        if let Some(entry) = self.entries.get_mut(entity_id) {
            self.stats.hits += 1;
            entry.touch();
            Some(entry)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a cached entity
    pub fn insert(&mut self, entity_id: EntityId, entry: CachedEntity) {
        let entry_size = entry.memory_size();

        // Evict if necessary
        while self.current_size_bytes + entry_size > self.max_size_bytes && !self.entries.is_empty()
        {
            self.evict_one();
        }

        // Remove old entry if exists
        if let Some(old) = self.entries.remove(&entity_id) {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(old.memory_size());
        }

        self.current_size_bytes += entry_size;
        self.entries.insert(entity_id, entry);
        self.stats.insertions += 1;
    }

    /// Check if entity is cached
    pub fn contains(&self, entity_id: &EntityId) -> bool {
        self.entries.contains_key(entity_id)
    }

    /// Check if entity is cached with matching hash
    pub fn contains_with_hash(&self, entity_id: &EntityId, hash: &[u8; 32]) -> bool {
        self.entries
            .get(entity_id)
            .map(|e| &e.content_hash == hash)
            .unwrap_or(false)
    }

    /// Remove a cached entity
    pub fn remove(&mut self, entity_id: &EntityId) -> Option<CachedEntity> {
        if let Some(entry) = self.entries.remove(entity_id) {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(entry.memory_size());
            Some(entry)
        } else {
            None
        }
    }

    /// Clear all cached entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_size_bytes = 0;
    }

    /// Evict one entry (LRU policy)
    fn evict_one(&mut self) {
        // Find least recently used entry
        let lru_id = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(id, _)| *id);

        if let Some(id) = lru_id {
            if let Some(entry) = self.entries.remove(&id) {
                self.current_size_bytes =
                    self.current_size_bytes.saturating_sub(entry.memory_size());
                self.stats.evictions += 1;
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get current cache size in bytes
    pub fn size_bytes(&self) -> usize {
        self.current_size_bytes
    }

    /// Get maximum cache size in bytes
    pub fn max_size_bytes(&self) -> usize {
        self.max_size_bytes
    }

    /// Get number of cached entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get fill ratio (0.0 - 1.0)
    pub fn fill_ratio(&self) -> f64 {
        if self.max_size_bytes == 0 {
            0.0
        } else {
            self.current_size_bytes as f64 / self.max_size_bytes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let mut cache = EntityCache::new(1024 * 1024); // 1MB

        let entity_id = EntityId::new();
        let entry = CachedEntity::new([0; 32]);

        cache.insert(entity_id, entry);
        assert!(cache.contains(&entity_id));

        let retrieved = cache.get(&entity_id).unwrap();
        assert_eq!(retrieved.content_hash, [0; 32]);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = EntityCache::new(1024 * 1024);

        let entity_id = EntityId::new();
        cache.insert(entity_id, CachedEntity::new([0; 32]));

        // Hit
        cache.get(&entity_id);
        assert_eq!(cache.stats().hits, 1);

        // Miss
        cache.get(&EntityId::new());
        assert_eq!(cache.stats().misses, 1);

        assert!((cache.stats().hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cache_eviction() {
        // Small cache that can only hold 2 entries of 300 bytes each (600 bytes total)
        let mut cache = EntityCache::new(600);

        let id1 = EntityId::new();
        let id2 = EntityId::new();
        let id3 = EntityId::new();

        let mut entry1 = CachedEntity::new([1; 32]);
        entry1.embedding = Some(vec![0.0; 50]); // ~200 bytes + overhead

        let mut entry2 = CachedEntity::new([2; 32]);
        entry2.embedding = Some(vec![0.0; 50]);

        let mut entry3 = CachedEntity::new([3; 32]);
        entry3.embedding = Some(vec![0.0; 50]);

        cache.insert(id1, entry1);
        cache.insert(id2, entry2);

        // This should evict something to make room
        cache.insert(id3, entry3);

        // At least one entry should have been evicted
        assert!(cache.stats().evictions > 0);
        // Should have at most 2 entries
        assert!(cache.len() <= 2);
        // id3 should definitely be present (most recent)
        assert!(cache.contains(&id3));
    }

    #[test]
    fn test_cache_hash_check() {
        let mut cache = EntityCache::new(1024 * 1024);

        let entity_id = EntityId::new();
        let hash = [42; 32];
        cache.insert(entity_id, CachedEntity::new(hash));

        assert!(cache.contains_with_hash(&entity_id, &hash));
        assert!(!cache.contains_with_hash(&entity_id, &[0; 32]));
    }

    #[test]
    fn test_cached_entity_metrics() {
        let entry = CachedEntity::new([0; 32]);

        // Should have some base size
        assert!(entry.memory_size() > 0);

        // Age should be small
        assert!(entry.age_seconds() < 1.0);
    }
}
