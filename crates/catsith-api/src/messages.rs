//! Protocol messages
//!
//! Defines all message types exchanged between game servers and CatSith clients.

use catsith_core::{EntityId, EntityIdentity, PlayerStyle, Scene};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Protocol version
pub const PROTOCOL_VERSION: u32 = 1;

/// Top-level message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Protocol version
    pub version: u32,
    /// Message sequence number
    pub sequence: u64,
    /// Timestamp (milliseconds since epoch)
    pub timestamp: u64,
    /// Message payload
    pub payload: MessagePayload,
}

impl Message {
    /// Create a new message with the given payload
    pub fn new(sequence: u64, payload: MessagePayload) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            sequence,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            payload,
        }
    }

    /// Create a scene message
    pub fn scene(sequence: u64, scene: Scene) -> Self {
        Self::new(sequence, MessagePayload::Scene(SceneMessage { scene }))
    }

    /// Create an identity update message
    pub fn identity(sequence: u64, identities: HashMap<EntityId, EntityIdentity>) -> Self {
        Self::new(
            sequence,
            MessagePayload::Identity(IdentityMessage { identities }),
        )
    }
}

/// Message payload types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Scene update from server
    Scene(SceneMessage),
    /// Entity identity update from server
    Identity(IdentityMessage),
    /// Client message to server
    Client(ClientMessage),
    /// Handshake/connection message
    Handshake(HandshakeMessage),
    /// Heartbeat/keep-alive
    Heartbeat,
    /// Acknowledgement
    Ack(AckMessage),
}

/// Scene update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMessage {
    /// The scene to render
    pub scene: Scene,
}

/// Entity identity update message
///
/// Sent separately from scenes to allow caching.
/// Full identity data is expensive, so we only send updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityMessage {
    /// Updated identities (EntityId → full identity)
    pub identities: HashMap<EntityId, EntityIdentity>,
}

/// Client → Server messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientMessage {
    /// Client capabilities report
    Capabilities(ClientCapabilities),

    /// Style preference update
    StyleUpdate(PlayerStyle),

    /// Request identity for entity
    RequestIdentity(Vec<EntityId>),

    /// Performance metrics
    Metrics(ClientMetrics),

    /// Input event (if client handles input)
    Input(InputEvent),
}

/// Client capabilities reported during handshake
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Client version
    pub version: String,
    /// Maximum supported quality tier
    pub max_quality: catsith_core::QualityTier,
    /// Supported output formats
    pub formats: Vec<catsith_core::OutputFormat>,
    /// Available backends
    pub backends: Vec<catsith_core::BackendType>,
    /// Preferred frame rate
    pub preferred_fps: u32,
    /// Output dimensions
    pub dimensions: (u32, u32),
}

/// Client performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientMetrics {
    /// Average frame time in milliseconds
    pub frame_time_ms: f64,
    /// Frames per second
    pub fps: f64,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Dropped frames count
    pub dropped_frames: u64,
}

/// Input event from client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputEvent {
    /// Key press
    KeyPress { key: String, modifiers: u32 },
    /// Key release
    KeyRelease { key: String, modifiers: u32 },
    /// Mouse/pointer movement
    PointerMove { x: f64, y: f64 },
    /// Mouse/pointer button
    PointerButton { button: u8, pressed: bool },
}

/// Handshake message for connection setup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeMessage {
    /// Handshake type
    pub handshake_type: HandshakeType,
    /// Client/server identifier
    pub identifier: String,
    /// Authentication token (if required)
    pub auth_token: Option<String>,
}

/// Handshake types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandshakeType {
    /// Client initiating connection
    ClientHello,
    /// Server accepting connection
    ServerHello,
    /// Authentication required
    AuthRequired,
    /// Authentication success
    AuthSuccess,
    /// Connection rejected
    Rejected,
}

/// Acknowledgement message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AckMessage {
    /// Sequence number being acknowledged
    pub sequence: u64,
    /// Optional error if message processing failed
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let scene = Scene::new(1);
        let msg = Message::scene(42, scene);

        assert_eq!(msg.version, PROTOCOL_VERSION);
        assert_eq!(msg.sequence, 42);
        assert!(matches!(msg.payload, MessagePayload::Scene(_)));
    }

    #[test]
    fn test_message_serialization() {
        let scene = Scene::new(1);
        let msg = Message::scene(1, scene);

        let json = serde_json::to_string(&msg).unwrap();
        let decoded: Message = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.sequence, 1);
    }

    #[test]
    fn test_client_message() {
        let metrics = ClientMetrics {
            frame_time_ms: 16.5,
            fps: 60.0,
            cache_hit_rate: 0.85,
            memory_bytes: 1024 * 1024 * 100,
            dropped_frames: 2,
        };

        let msg = ClientMessage::Metrics(metrics);
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: ClientMessage = serde_json::from_str(&json).unwrap();

        if let ClientMessage::Metrics(m) = decoded {
            assert_eq!(m.fps, 60.0);
        } else {
            panic!("Wrong message type");
        }
    }
}
