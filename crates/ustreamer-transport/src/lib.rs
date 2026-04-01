//! WebTransport server for streaming encoded video and receiving input.
//!
//! Uses `quinn` for QUIC/HTTP3 transport:
//! - **Unreliable datagrams** for video frames (newest-wins, no retransmission)
//! - **Unreliable datagrams** for continuous input (pointer move, scroll)
//! - **Reliable QUIC streams** for discrete input (key press, tool select)
//! - **Reliable QUIC stream** for session control (codec negotiation, quality params)

pub mod session;

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("session closed")]
    SessionClosed,
    #[error("datagram too large: {size} bytes (max {max})")]
    DatagramTooLarge { size: usize, max: usize },
}
