//! WebTransport session management primitives built on top of `wtransport`.

use std::net::SocketAddr;
use std::time::Duration;

use ustreamer_proto::frame::FramePacket;
use ustreamer_proto::input::InputEvent;
use wtransport::endpoint::endpoint_side::Server;
use wtransport::tls::Sha256Digest;
use wtransport::{Connection, Endpoint, Identity, ServerConfig};

use crate::TransportError;

/// Server-side TLS identity source.
pub enum ServerIdentity {
    /// Use an existing certificate chain and private key.
    Provided(Identity),
    /// Generate a two-week self-signed identity for a known set of hostnames/IPs.
    SelfSigned { subject_alt_names: Vec<String> },
}

impl ServerIdentity {
    fn into_identity_and_hash(self) -> Result<(Identity, Sha256Digest), TransportError> {
        let identity = match self {
            ServerIdentity::Provided(identity) => identity,
            ServerIdentity::SelfSigned { subject_alt_names } => {
                Identity::self_signed(subject_alt_names.iter().map(String::as_str))
                    .map_err(|err| TransportError::InitFailed(err.to_string()))?
            }
        };

        let certificate_hash = {
            let chain = identity.certificate_chain();
            let Some(certificate) = chain.as_slice().first() else {
                return Err(TransportError::InitFailed(
                    "identity did not contain a certificate".to_owned(),
                ));
            };

            certificate.hash()
        };

        Ok((identity, certificate_hash))
    }
}

/// Transport-layer configuration for the WebTransport endpoint.
pub struct TransportConfig {
    /// UDP socket bind address for the server endpoint.
    pub bind_address: SocketAddr,
    /// TLS identity used during the WebTransport handshake.
    pub identity: ServerIdentity,
    /// Keep-alive interval for preserving low-latency LAN sessions.
    pub keep_alive_interval: Option<Duration>,
    /// Maximum permitted idle time before the connection is timed out.
    pub max_idle_timeout: Option<Duration>,
}

impl TransportConfig {
    /// Convenience helper for local development with a self-signed identity.
    pub fn localhost_self_signed(bind_address: SocketAddr) -> Self {
        Self {
            bind_address,
            identity: ServerIdentity::SelfSigned {
                subject_alt_names: vec!["localhost".to_owned(), "127.0.0.1".to_owned()],
            },
            keep_alive_interval: Some(Duration::from_secs(3)),
            max_idle_timeout: Some(Duration::from_secs(10)),
        }
    }
}

/// An accepted WebTransport session request and the established session.
pub struct AcceptedSession {
    /// Host/authority requested by the client.
    pub authority: String,
    /// Path requested by the client (e.g. `/stream`).
    pub path: String,
    /// Session handle for video, control, and input traffic.
    pub session: StreamSession,
}

/// WebTransport server endpoint that accepts browser sessions.
pub struct WebTransportServer {
    endpoint: Endpoint<Server>,
    certificate_hash: Sha256Digest,
}

impl WebTransportServer {
    /// Bind a WebTransport server endpoint on the configured socket.
    pub fn bind(config: TransportConfig) -> Result<Self, TransportError> {
        let (identity, certificate_hash) = config.identity.into_identity_and_hash()?;

        let server_config = ServerConfig::builder()
            .with_bind_address(config.bind_address)
            .with_identity(identity)
            .keep_alive_interval(config.keep_alive_interval)
            .max_idle_timeout(config.max_idle_timeout)
            .map_err(|err| TransportError::InitFailed(err.to_string()))?
            .build();

        let endpoint = Endpoint::server(server_config)
            .map_err(|err| TransportError::InitFailed(err.to_string()))?;

        Ok(Self {
            endpoint,
            certificate_hash,
        })
    }

    /// Returns the local socket address of the bound UDP endpoint.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.endpoint.local_addr()
    }

    /// Returns the certificate digest browsers/clients can pin during setup.
    pub fn certificate_hash(&self) -> &Sha256Digest {
        &self.certificate_hash
    }

    /// Accept the next WebTransport session and complete the session handshake.
    pub async fn accept_session(&self) -> Result<AcceptedSession, TransportError> {
        let incoming = self.endpoint.accept().await;
        let request = incoming
            .await
            .map_err(|err| TransportError::ConnectionFailed(err.to_string()))?;

        let authority = request.authority().to_string();
        let path = request.path().to_string();
        let connection = request
            .accept()
            .await
            .map_err(|err| TransportError::ConnectionFailed(err.to_string()))?;

        Ok(AcceptedSession {
            authority,
            path,
            session: StreamSession { connection },
        })
    }
}

/// Whether an input event arrived unreliably (datagram) or reliably (stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputReliability {
    Unreliable,
    Reliable,
}

/// Input event received from the browser session.
#[derive(Debug, Clone, Copy)]
pub struct ReceivedInput {
    pub reliability: InputReliability,
    pub event: InputEvent,
}

/// Established WebTransport session used by the streaming loop.
#[derive(Clone)]
pub struct StreamSession {
    connection: Connection,
}

impl StreamSession {
    /// Current estimate of round-trip time for the session.
    pub fn rtt(&self) -> Duration {
        self.connection.rtt()
    }

    /// Current peer address.
    pub fn remote_address(&self) -> SocketAddr {
        self.connection.remote_address()
    }

    /// Maximum datagram payload permitted by the current path MTU estimate.
    pub fn max_datagram_size(&self) -> Option<usize> {
        self.connection.max_datagram_size()
    }

    /// Send a single packetized frame fragment over QUIC datagram transport.
    pub fn send_frame_packet(&self, packet: &FramePacket) -> Result<(), TransportError> {
        let bytes = packet.to_bytes();
        self.send_datagram(&bytes)
    }

    /// Send a batch of packetized frame fragments in order.
    pub fn send_frame_packets(&self, packets: &[FramePacket]) -> Result<(), TransportError> {
        for packet in packets {
            self.send_frame_packet(packet)?;
        }

        Ok(())
    }

    /// Receive the next unreliable input datagram from the browser.
    pub async fn recv_input_datagram(&self) -> Result<InputEvent, TransportError> {
        let datagram = self
            .connection
            .receive_datagram()
            .await
            .map_err(|err| TransportError::ConnectionFailed(err.to_string()))?;

        InputEvent::from_bytes(datagram.as_ref())
            .map_err(|err| TransportError::InvalidInputEvent(err.to_string()))
    }

    /// Receive the next reliable input message from a uni- or bidirectional stream.
    pub async fn recv_reliable_input(&self) -> Result<InputEvent, TransportError> {
        let message = self.recv_reliable_message().await?;
        InputEvent::from_bytes(&message)
            .map_err(|err| TransportError::InvalidInputEvent(err.to_string()))
    }

    /// Receive the next input event, regardless of reliability mode.
    pub async fn recv_input(&self) -> Result<ReceivedInput, TransportError> {
        let datagram_connection = self.connection.clone();
        let reliable_connection = self.connection.clone();

        tokio::select! {
            datagram = datagram_connection.receive_datagram() => {
                let datagram = datagram.map_err(|err| TransportError::ConnectionFailed(err.to_string()))?;
                let event = InputEvent::from_bytes(datagram.as_ref())
                    .map_err(|err| TransportError::InvalidInputEvent(err.to_string()))?;

                Ok(ReceivedInput {
                    reliability: InputReliability::Unreliable,
                    event,
                })
            }
            reliable = recv_reliable_message_from(reliable_connection) => {
                let bytes = reliable?;
                let event = InputEvent::from_bytes(&bytes)
                    .map_err(|err| TransportError::InvalidInputEvent(err.to_string()))?;

                Ok(ReceivedInput {
                    reliability: InputReliability::Reliable,
                    event,
                })
            }
        }
    }

    /// Send a reliable control message to the browser using a unidirectional stream.
    pub async fn send_control_message(&self, payload: &[u8]) -> Result<(), TransportError> {
        let mut stream = self
            .connection
            .open_uni()
            .await
            .map_err(|err| TransportError::ConnectionFailed(err.to_string()))?
            .await
            .map_err(|err| TransportError::ConnectionFailed(err.to_string()))?;

        stream
            .write_all(payload)
            .await
            .map_err(|err| TransportError::StreamIo(err.to_string()))
    }

    fn send_datagram(&self, payload: &[u8]) -> Result<(), TransportError> {
        let max = self
            .max_datagram_size()
            .ok_or(TransportError::DatagramsUnsupported)?;

        if payload.len() > max {
            return Err(TransportError::DatagramTooLarge {
                size: payload.len(),
                max,
            });
        }

        self.connection
            .send_datagram(payload)
            .map_err(|err| TransportError::ConnectionFailed(err.to_string()))
    }

    async fn recv_reliable_message(&self) -> Result<Vec<u8>, TransportError> {
        recv_reliable_message_from(self.connection.clone()).await
    }
}

async fn recv_reliable_message_from(connection: Connection) -> Result<Vec<u8>, TransportError> {
    let uni_connection = connection.clone();
    let bi_connection = connection;

    tokio::select! {
        uni = uni_connection.accept_uni() => {
            let mut stream = uni.map_err(|err| TransportError::ConnectionFailed(err.to_string()))?;
            read_all(&mut stream).await
        }
        bi = bi_connection.accept_bi() => {
            let (_, mut stream) = bi.map_err(|err| TransportError::ConnectionFailed(err.to_string()))?;
            read_all(&mut stream).await
        }
    }
}

async fn read_all(stream: &mut wtransport::RecvStream) -> Result<Vec<u8>, TransportError> {
    let mut output = Vec::new();
    let mut buffer = vec![0u8; 4096];

    loop {
        let bytes_read = stream
            .read(&mut buffer)
            .await
            .map_err(|err| TransportError::StreamIo(err.to_string()))?;

        match bytes_read {
            Some(0) => break,
            Some(bytes_read) => output.extend_from_slice(&buffer[..bytes_read]),
            None => break,
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
    use tokio::time::{Duration, timeout};
    use wtransport::endpoint::endpoint_side::Client;
    use wtransport::{ClientConfig, Endpoint};

    use super::*;

    struct LoopbackPair {
        _server: WebTransportServer,
        _client_endpoint: Endpoint<Client>,
        server_session: StreamSession,
        client_connection: Connection,
        path: String,
    }

    async fn loopback_pair() -> Result<LoopbackPair> {
        let bind_address = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0));
        let server =
            WebTransportServer::bind(TransportConfig::localhost_self_signed(bind_address))?;
        let cert_hash = server.certificate_hash().clone();
        let port = server.local_addr()?.port();

        let client_config = ClientConfig::builder()
            .with_bind_default()
            .with_server_certificate_hashes([cert_hash])
            .build();

        let client_endpoint = Endpoint::client(client_config)?;
        let url = format!("https://127.0.0.1:{port}/stream");

        let (accepted, client_connection) = tokio::join!(
            async {
                Ok::<_, anyhow::Error>(
                    timeout(Duration::from_secs(5), server.accept_session()).await??,
                )
            },
            async {
                Ok::<_, anyhow::Error>(
                    timeout(Duration::from_secs(5), client_endpoint.connect(url)).await??,
                )
            }
        );

        let accepted = accepted?;
        let client_connection = client_connection?;

        Ok(LoopbackPair {
            _server: server,
            _client_endpoint: client_endpoint,
            server_session: accepted.session,
            client_connection,
            path: accepted.path,
        })
    }

    async fn read_client_stream(stream: &mut wtransport::RecvStream) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        let mut buffer = vec![0u8; 4096];

        loop {
            let bytes_read = stream.read(&mut buffer).await?;
            match bytes_read {
                Some(0) => break,
                Some(bytes_read) => output.extend_from_slice(&buffer[..bytes_read]),
                None => break,
            }
        }

        Ok(output)
    }

    #[tokio::test]
    async fn accepts_session_and_receives_input_datagram() -> Result<()> {
        let pair = loopback_pair().await?;
        assert_eq!(pair.path, "/stream");

        let input = InputEvent::PointerMove {
            x: 0.25,
            y: 0.75,
            buttons: 1,
            timestamp_ms: 4242,
        };

        pair.client_connection.send_datagram(&input.to_bytes())?;

        let received = timeout(
            Duration::from_secs(5),
            pair.server_session.recv_input_datagram(),
        )
        .await??;

        match received {
            InputEvent::PointerMove {
                x,
                y,
                buttons,
                timestamp_ms,
            } => {
                assert!((x - 0.25).abs() < f32::EPSILON);
                assert!((y - 0.75).abs() < f32::EPSILON);
                assert_eq!(buttons, 1);
                assert_eq!(timestamp_ms, 4242);
            }
            _ => panic!("expected pointer move"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn sends_frame_packets_over_datagrams() -> Result<()> {
        let pair = loopback_pair().await?;

        let packet = FramePacket {
            frame_id: 7,
            fragment_idx: 0,
            fragment_count: 1,
            timestamp_us: 123_456,
            is_keyframe: true,
            is_lossless: false,
            payload: vec![1, 2, 3, 4, 5],
        };

        pair.server_session.send_frame_packet(&packet)?;

        let datagram = timeout(
            Duration::from_secs(5),
            pair.client_connection.receive_datagram(),
        )
        .await??;
        let decoded = FramePacket::from_bytes(datagram.as_ref())?;

        assert_eq!(decoded.frame_id, 7);
        assert_eq!(decoded.fragment_idx, 0);
        assert_eq!(decoded.fragment_count, 1);
        assert_eq!(decoded.timestamp_us, 123_456);
        assert!(decoded.is_keyframe);
        assert!(!decoded.is_lossless);
        assert_eq!(decoded.payload, vec![1, 2, 3, 4, 5]);

        Ok(())
    }

    #[tokio::test]
    async fn receives_reliable_input_and_sends_control_message() -> Result<()> {
        let pair = loopback_pair().await?;

        let mut send_stream = pair.client_connection.open_uni().await?.await?;
        send_stream
            .write_all(&InputEvent::KeyDown { code: 0x0041 }.to_bytes())
            .await?;
        drop(send_stream);

        let received = timeout(
            Duration::from_secs(5),
            pair.server_session.recv_reliable_input(),
        )
        .await??;

        match received {
            InputEvent::KeyDown { code } => assert_eq!(code, 0x0041),
            _ => panic!("expected key down"),
        }

        let control_message = b"codec=h265;mode=interactive";
        pair.server_session
            .send_control_message(control_message)
            .await?;

        let mut recv_stream =
            timeout(Duration::from_secs(5), pair.client_connection.accept_uni()).await??;
        let payload = read_client_stream(&mut recv_stream).await?;
        assert_eq!(payload, control_message);

        Ok(())
    }
}
