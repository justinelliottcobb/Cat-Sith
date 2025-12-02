//! Efficient message encoding/decoding
//!
//! Supports multiple codecs for different use cases:
//! - JSON: Human-readable, debugging

use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

/// Serialization codec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Codec {
    /// JSON encoding (human-readable)
    #[default]
    Json,
}

impl Codec {
    /// Encode a message
    pub fn encode<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, CodecError> {
        match self {
            Self::Json => serde_json::to_vec(value).map_err(|e| CodecError::Encode(e.to_string())),
        }
    }

    /// Decode a message
    pub fn decode<T: DeserializeOwned>(&self, data: &[u8]) -> Result<T, CodecError> {
        match self {
            Self::Json => {
                serde_json::from_slice(data).map_err(|e| CodecError::Decode(e.to_string()))
            }
        }
    }

    /// Get the content type for HTTP headers
    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Json => "application/json",
        }
    }
}

/// Codec errors
#[derive(Debug, Error)]
pub enum CodecError {
    #[error("Encoding failed: {0}")]
    Encode(String),

    #[error("Decoding failed: {0}")]
    Decode(String),
}

/// Message framing for streaming protocols
///
/// Format: [4 bytes length (big-endian)] [message data]
pub struct FramedCodec {
    codec: Codec,
}

impl FramedCodec {
    /// Create a new framed codec
    pub fn new(codec: Codec) -> Self {
        Self { codec }
    }

    /// Encode a message with framing
    pub fn encode_framed<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, CodecError> {
        let data = self.codec.encode(value)?;
        let len = data.len() as u32;

        let mut framed = Vec::with_capacity(4 + data.len());
        framed.extend_from_slice(&len.to_be_bytes());
        framed.extend(data);

        Ok(framed)
    }

    /// Decode a framed message
    ///
    /// Returns (decoded value, bytes consumed) or None if incomplete
    pub fn decode_framed<T: DeserializeOwned>(
        &self,
        data: &[u8],
    ) -> Result<Option<(T, usize)>, CodecError> {
        if data.len() < 4 {
            return Ok(None);
        }

        let len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if data.len() < 4 + len {
            return Ok(None);
        }

        let value = self.codec.decode(&data[4..4 + len])?;
        Ok(Some((value, 4 + len)))
    }
}

/// Content hash calculator for identity caching
pub fn content_hash<T: Serialize>(value: &T) -> [u8; 32] {
    let data = serde_json::to_vec(value).unwrap_or_default();
    *blake3::hash(&data).as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{Message, MessagePayload};
    use catsith_core::Scene;

    #[test]
    fn test_json_roundtrip() {
        let scene = Scene::new(42);
        let msg = Message::scene(1, scene);

        let codec = Codec::Json;
        let encoded = codec.encode(&msg).unwrap();
        let decoded: Message = codec.decode(&encoded).unwrap();

        assert_eq!(decoded.sequence, 1);
        if let MessagePayload::Scene(s) = decoded.payload {
            assert_eq!(s.scene.frame_id, 42);
        } else {
            panic!("Wrong payload type");
        }
    }

    #[test]
    fn test_framed_codec() {
        let scene = Scene::new(42);
        let msg = Message::scene(1, scene);

        let framed = FramedCodec::new(Codec::Json);
        let encoded = framed.encode_framed(&msg).unwrap();

        // Should have 4-byte length prefix
        assert!(encoded.len() > 4);

        // Check length prefix
        let len = u32::from_be_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        assert_eq!(len, encoded.len() - 4);

        // Decode
        let (decoded, consumed): (Message, usize) =
            framed.decode_framed(&encoded).unwrap().unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.sequence, 1);
    }

    #[test]
    fn test_framed_incomplete() {
        let framed = FramedCodec::new(Codec::Json);

        // Too short for length prefix
        let result: Result<Option<(Message, usize)>, _> = framed.decode_framed(&[0, 0, 0]);
        assert!(result.unwrap().is_none());

        // Has length but not enough data
        let result: Result<Option<(Message, usize)>, _> =
            framed.decode_framed(&[0, 0, 0, 100, 1, 2, 3]);
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_content_hash() {
        let scene1 = Scene::new(1);
        let scene2 = Scene::new(1);
        let scene3 = Scene::new(2);

        let hash1 = content_hash(&scene1);
        let hash2 = content_hash(&scene2);
        let hash3 = content_hash(&scene3);

        // Same content should produce same hash
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        assert_ne!(hash1, hash3);
    }
}
