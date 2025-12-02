//! CatSith API - Client/Server Protocol
//!
//! This crate provides the communication layer between game servers
//! and the CatSith rendering frontend.
//!
//! # Architecture
//!
//! ```text
//! Game Server                          CatSith Client
//! ┌─────────────┐                     ┌─────────────┐
//! │             │   SceneMessage      │             │
//! │  produces   │ ──────────────────► │  receives   │
//! │  semantic   │                     │  renders    │
//! │  scenes     │   IdentityMessage   │  scenes     │
//! │             │ ──────────────────► │             │
//! │             │                     │             │
//! │             │   ClientMessage     │             │
//! │             │ ◄────────────────── │  sends      │
//! │             │                     │  feedback   │
//! └─────────────┘                     └─────────────┘
//! ```

pub mod client;
pub mod messages;
pub mod serialization;
pub mod server;

// Re-export commonly used types
pub use client::{CatSithClient, ClientConfig, ClientError};
pub use messages::{ClientMessage, IdentityMessage, Message, SceneMessage};
pub use serialization::{Codec, CodecError};
pub use server::{CatSithServer, ServerConfig, ServerError};
