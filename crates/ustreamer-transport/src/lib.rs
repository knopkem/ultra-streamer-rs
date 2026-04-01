//! WebTransport server for streaming encoded video and receiving input.
//!
//! Uses `quinn` for QUIC/HTTP3 transport:
//! - **Unreliable datagrams** for video frames (newest-wins, no retransmission)
//! - **Unreliable datagrams** for continuous input (pointer move, scroll)
//! - **Reliable QUIC streams** for discrete input (key press, tool select)
//! - **Reliable QUIC stream** for session control (codec negotiation, quality params)
//!
//! Also includes a WebSocket fallback transport for browsers without WebTransport.

pub mod session;
pub mod websocket;
pub use session::{
    AcceptedSession, InputReliability, ReceivedInput, ServerIdentity, StreamSession,
    TransportConfig, WebTransportServer,
};
pub use websocket::{AcceptedWebSocketSession, WebSocketServer, WebSocketSession};

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("transport initialization failed: {0}")]
    InitFailed(String),
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("session closed")]
    SessionClosed,
    #[error("peer does not support QUIC datagrams")]
    DatagramsUnsupported,
    #[error("datagram too large: {size} bytes (max {max})")]
    DatagramTooLarge { size: usize, max: usize },
    #[error("invalid frame packet: {0}")]
    InvalidFramePacket(String),
    #[error("invalid input event: {0}")]
    InvalidInputEvent(String),
    #[error("stream I/O failed: {0}")]
    StreamIo(String),
}
