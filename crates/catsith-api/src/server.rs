//! CatSith server for game integration

use crate::messages::{ClientCapabilities, IdentityMessage, Message, MessagePayload, SceneMessage};
use crate::serialization::Codec;
use catsith_core::{EntityId, EntityIdentity, Scene};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, error, info};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Listen address
    pub listen_address: String,
    /// Maximum connected clients
    pub max_clients: usize,
    /// Preferred codec
    pub codec: Codec,
    /// Scene broadcast buffer size
    pub broadcast_buffer_size: usize,
    /// Require authentication
    pub require_auth: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen_address: "0.0.0.0:7878".to_string(),
            max_clients: 64,
            codec: Codec::Json,
            broadcast_buffer_size: 16,
            require_auth: false,
        }
    }
}

/// Server errors
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("Server not started")]
    NotStarted,

    #[error("Maximum clients reached")]
    MaxClientsReached,

    #[error("Client not found: {0}")]
    ClientNotFound(String),

    #[error("Broadcast failed: {0}")]
    BroadcastFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Codec error: {0}")]
    Codec(#[from] crate::serialization::CodecError),
}

/// Connected client information
#[derive(Debug, Clone)]
pub struct ConnectedClient {
    /// Client identifier
    pub id: String,
    /// Connection time
    pub connected_at: std::time::Instant,
    /// Client capabilities
    pub capabilities: Option<ClientCapabilities>,
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
}

/// CatSith server for game integration
pub struct CatSithServer {
    config: ServerConfig,
    clients: Arc<RwLock<HashMap<String, ConnectedClient>>>,
    identity_store: Arc<RwLock<HashMap<EntityId, EntityIdentity>>>,
    scene_sender: broadcast::Sender<Scene>,
    sequence: Arc<std::sync::atomic::AtomicU64>,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl CatSithServer {
    /// Create a new server
    pub fn new(config: ServerConfig) -> Self {
        let (scene_sender, _) = broadcast::channel(config.broadcast_buffer_size);

        Self {
            config,
            clients: Arc::new(RwLock::new(HashMap::new())),
            identity_store: Arc::new(RwLock::new(HashMap::new())),
            scene_sender,
            sequence: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get connected client count
    pub async fn client_count(&self) -> usize {
        self.clients.read().await.len()
    }

    /// Get next sequence number
    fn next_sequence(&self) -> u64 {
        self.sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Broadcast a scene to all connected clients
    pub fn broadcast_scene(&self, scene: Scene) -> Result<usize, ServerError> {
        match self.scene_sender.send(scene) {
            Ok(count) => {
                debug!("Broadcast scene to {} clients", count);
                Ok(count)
            }
            Err(_) => {
                // No receivers is not an error
                Ok(0)
            }
        }
    }

    /// Subscribe to scene broadcasts
    pub fn subscribe_scenes(&self) -> broadcast::Receiver<Scene> {
        self.scene_sender.subscribe()
    }

    /// Store an entity identity
    pub async fn store_identity(&self, entity_id: EntityId, identity: EntityIdentity) {
        self.identity_store
            .write()
            .await
            .insert(entity_id, identity);
    }

    /// Get stored identity
    pub async fn get_identity(&self, entity_id: &EntityId) -> Option<EntityIdentity> {
        self.identity_store.read().await.get(entity_id).cloned()
    }

    /// Get all stored identities
    pub async fn get_all_identities(&self) -> HashMap<EntityId, EntityIdentity> {
        self.identity_store.read().await.clone()
    }

    /// Create a scene message
    pub fn create_scene_message(&self, scene: Scene) -> Message {
        Message::new(
            self.next_sequence(),
            MessagePayload::Scene(SceneMessage { scene }),
        )
    }

    /// Create an identity update message
    pub fn create_identity_message(
        &self,
        identities: HashMap<EntityId, EntityIdentity>,
    ) -> Message {
        Message::new(
            self.next_sequence(),
            MessagePayload::Identity(IdentityMessage { identities }),
        )
    }

    /// Register a client connection
    pub async fn register_client(&self, id: String) -> Result<(), ServerError> {
        let mut clients = self.clients.write().await;

        if clients.len() >= self.config.max_clients {
            return Err(ServerError::MaxClientsReached);
        }

        let client = ConnectedClient {
            id: id.clone(),
            connected_at: std::time::Instant::now(),
            capabilities: None,
            messages_sent: 0,
            messages_received: 0,
        };

        clients.insert(id.clone(), client);
        info!("Client registered: {}", id);
        Ok(())
    }

    /// Unregister a client
    pub async fn unregister_client(&self, id: &str) {
        let mut clients = self.clients.write().await;
        if clients.remove(id).is_some() {
            info!("Client unregistered: {}", id);
        }
    }

    /// Update client capabilities
    pub async fn update_client_capabilities(
        &self,
        id: &str,
        capabilities: ClientCapabilities,
    ) -> Result<(), ServerError> {
        let mut clients = self.clients.write().await;
        if let Some(client) = clients.get_mut(id) {
            client.capabilities = Some(capabilities);
            Ok(())
        } else {
            Err(ServerError::ClientNotFound(id.to_string()))
        }
    }

    /// Get list of connected clients
    pub async fn list_clients(&self) -> Vec<ConnectedClient> {
        self.clients.read().await.values().cloned().collect()
    }
}

/// Simple in-process scene producer for testing
pub struct SceneProducer {
    server: Arc<CatSithServer>,
    frame_counter: u64,
}

impl SceneProducer {
    /// Create a new scene producer
    pub fn new(server: Arc<CatSithServer>) -> Self {
        Self {
            server,
            frame_counter: 0,
        }
    }

    /// Produce and broadcast a scene
    pub fn produce(&mut self, scene: Scene) -> Result<usize, ServerError> {
        self.frame_counter += 1;
        self.server.broadcast_scene(scene)
    }

    /// Get current frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.max_clients, 64);
        assert!(!config.require_auth);
    }

    #[tokio::test]
    async fn test_server_client_registration() {
        let server = CatSithServer::new(ServerConfig::default());

        server.register_client("client1".to_string()).await.unwrap();
        server.register_client("client2".to_string()).await.unwrap();

        assert_eq!(server.client_count().await, 2);

        server.unregister_client("client1").await;
        assert_eq!(server.client_count().await, 1);
    }

    #[tokio::test]
    async fn test_server_max_clients() {
        let config = ServerConfig {
            max_clients: 2,
            ..Default::default()
        };
        let server = CatSithServer::new(config);

        server.register_client("client1".to_string()).await.unwrap();
        server.register_client("client2".to_string()).await.unwrap();

        let result = server.register_client("client3".to_string()).await;
        assert!(matches!(result, Err(ServerError::MaxClientsReached)));
    }

    #[tokio::test]
    async fn test_server_identity_store() {
        let server = CatSithServer::new(ServerConfig::default());

        let entity_id = EntityId::new();
        let identity = EntityIdentity::new([0; 32]).with_name("Test Entity");

        server.store_identity(entity_id, identity.clone()).await;

        let retrieved = server.get_identity(&entity_id).await.unwrap();
        assert_eq!(retrieved.name, identity.name);
    }

    #[tokio::test]
    async fn test_scene_broadcast() {
        let server = Arc::new(CatSithServer::new(ServerConfig::default()));

        let mut receiver = server.subscribe_scenes();

        let scene = Scene::new(42);
        server.broadcast_scene(scene).unwrap();

        let received = receiver.recv().await.unwrap();
        assert_eq!(received.frame_id, 42);
    }

    #[tokio::test]
    async fn test_scene_producer() {
        let server = Arc::new(CatSithServer::new(ServerConfig::default()));
        let mut producer = SceneProducer::new(server.clone());

        let mut receiver = server.subscribe_scenes();

        producer.produce(Scene::new(1)).unwrap();
        producer.produce(Scene::new(2)).unwrap();

        assert_eq!(producer.frame_count(), 2);

        let scene1 = receiver.recv().await.unwrap();
        let scene2 = receiver.recv().await.unwrap();

        assert_eq!(scene1.frame_id, 1);
        assert_eq!(scene2.frame_id, 2);
    }
}
