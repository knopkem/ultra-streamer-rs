//! Shared protocol types for the ultra-streamer pipeline.
//!
//! Defines the binary wire formats for:
//! - Encoded video frame packets (server → browser)
//! - Input events (browser → server)
//! - Quality/session control messages (bidirectional)

pub mod frame;
pub mod input;
pub mod quality;
