//! CatSith client for receiving scenes from game servers

use crate::messages::{ClientCapabilities, ClientMessage, ClientMetrics, Message, MessagePayload};
use crate::serialization::Codec;
use catsith_core::{EntityId, EntityIdentity, PlayerStyle, RenderCapabilities, Scene};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{RwLock, mpsc};
use tracing::error;

/// Client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Server address
    pub server_address: String,
    /// Preferred codec
    pub codec: Codec,
    /// Reconnect on disconnect
    pub auto_reconnect: bool,
    /// Maximum reconnect attempts
    pub max_reconnect_attempts: u32,
    /// Scene buffer size
    pub scene_buffer_size: usize,
    /// Identity cache size
    pub identity_cache_size: usize,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            server_address: "127.0.0.1:7878".to_string(),
            codec: Codec::Json,
            auto_reconnect: true,
            max_reconnect_attempts: 5,
            scene_buffer_size: 16,
            identity_cache_size: 1024,
        }
    }
}

/// Client errors
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Disconnected from server")]
    Disconnected,

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Codec error: {0}")]
    Codec(#[from] crate::serialization::CodecError),

    #[error("Channel closed")]
    ChannelClosed,

    #[error("Authentication failed")]
    AuthFailed,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Scene receiver channel
pub type SceneReceiver = mpsc::Receiver<Scene>;

/// CatSith client
#[allow(dead_code)]
pub struct CatSithClient {
    config: ClientConfig,
    capabilities: RenderCapabilities,
    style: Arc<RwLock<PlayerStyle>>,
    identity_cache: Arc<RwLock<HashMap<EntityId, EntityIdentity>>>,
    sequence: Arc<std::sync::atomic::AtomicU64>,
    connected: Arc<std::sync::atomic::AtomicBool>,
}

impl CatSithClient {
    /// Create a new client
    pub fn new(config: ClientConfig, capabilities: RenderCapabilities) -> Self {
        Self {
            config,
            capabilities,
            style: Arc::new(RwLock::new(PlayerStyle::default())),
            identity_cache: Arc::new(RwLock::new(HashMap::new())),
            sequence: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get current style
    pub async fn style(&self) -> PlayerStyle {
        self.style.read().await.clone()
    }

    /// Update player style
    pub async fn set_style(&self, style: PlayerStyle) {
        *self.style.write().await = style;
    }

    /// Get cached entity identity
    pub async fn get_identity(&self, entity_id: &EntityId) -> Option<EntityIdentity> {
        self.identity_cache.read().await.get(entity_id).cloned()
    }

    /// Get all cached identities
    pub async fn identities(&self) -> HashMap<EntityId, EntityIdentity> {
        self.identity_cache.read().await.clone()
    }

    /// Cache an entity identity
    pub async fn cache_identity(&self, entity_id: EntityId, identity: EntityIdentity) {
        let mut cache = self.identity_cache.write().await;

        // Evict old entries if cache is full
        if cache.len() >= self.config.identity_cache_size {
            // Simple eviction: remove first entry (could be LRU)
            if let Some(key) = cache.keys().next().copied() {
                cache.remove(&key);
            }
        }

        cache.insert(entity_id, identity);
    }

    /// Get next sequence number
    fn next_sequence(&self) -> u64 {
        self.sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Create client capabilities message
    #[allow(dead_code)]
    fn client_capabilities(&self) -> ClientCapabilities {
        ClientCapabilities {
            version: env!("CARGO_PKG_VERSION").to_string(),
            max_quality: self.capabilities.recommended_tier,
            formats: vec![catsith_core::OutputFormat::Terminal {
                colors: catsith_core::intent::ColorDepth::TrueColor,
            }],
            backends: self.capabilities.backends.clone(),
            preferred_fps: 60,
            dimensions: (80, 24),
        }
    }

    /// Create a metrics report
    pub fn create_metrics(&self, frame_time_ms: f64, fps: f64, dropped_frames: u64) -> Message {
        let metrics = ClientMetrics {
            frame_time_ms,
            fps,
            cache_hit_rate: 0.0, // Would be calculated from actual cache stats
            memory_bytes: 0,     // Would be calculated from allocator stats
            dropped_frames,
        };

        Message::new(
            self.next_sequence(),
            MessagePayload::Client(ClientMessage::Metrics(metrics)),
        )
    }
}

/// Offline/mock client for testing
pub struct MockClient {
    scenes: Vec<Scene>,
    current_index: usize,
}

impl MockClient {
    /// Create a new mock client with scenes
    pub fn new(scenes: Vec<Scene>) -> Self {
        Self {
            scenes,
            current_index: 0,
        }
    }

    /// Get next scene (cycles through available scenes)
    pub fn next_scene(&mut self) -> Option<Scene> {
        if self.scenes.is_empty() {
            return None;
        }

        let scene = self.scenes[self.current_index].clone();
        self.current_index = (self.current_index + 1) % self.scenes.len();
        Some(scene)
    }

    /// Add a scene
    pub fn add_scene(&mut self, scene: Scene) {
        self.scenes.push(scene);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.codec, Codec::Json);
        assert!(config.auto_reconnect);
    }

    #[test]
    fn test_mock_client() {
        let scenes = vec![Scene::new(1), Scene::new(2), Scene::new(3)];

        let mut client = MockClient::new(scenes);

        assert_eq!(client.next_scene().unwrap().frame_id, 1);
        assert_eq!(client.next_scene().unwrap().frame_id, 2);
        assert_eq!(client.next_scene().unwrap().frame_id, 3);
        // Should cycle back
        assert_eq!(client.next_scene().unwrap().frame_id, 1);
    }

    #[tokio::test]
    async fn test_client_identity_cache() {
        let config = ClientConfig {
            identity_cache_size: 2,
            ..Default::default()
        };
        let client = CatSithClient::new(config, RenderCapabilities::minimal());

        let id1 = EntityId::new();
        let id2 = EntityId::new();
        let id3 = EntityId::new();

        let identity1 = EntityIdentity::new([0; 32]).with_name("Entity 1");
        let identity2 = EntityIdentity::new([1; 32]).with_name("Entity 2");
        let identity3 = EntityIdentity::new([2; 32]).with_name("Entity 3");

        client.cache_identity(id1, identity1).await;
        client.cache_identity(id2, identity2).await;

        assert!(client.get_identity(&id1).await.is_some());
        assert!(client.get_identity(&id2).await.is_some());

        // Adding third should evict one (cache size is 2)
        client.cache_identity(id3, identity3).await;

        let cache = client.identities().await;
        assert_eq!(cache.len(), 2);
        assert!(client.get_identity(&id3).await.is_some());
    }

    #[tokio::test]
    async fn test_client_style() {
        let client = CatSithClient::new(ClientConfig::default(), RenderCapabilities::minimal());

        let style = PlayerStyle::terminal();
        client.set_style(style.clone()).await;

        let retrieved = client.style().await;
        assert_eq!(
            retrieved.preferred_quality,
            catsith_core::QualityTier::Minimal
        );
    }
}
