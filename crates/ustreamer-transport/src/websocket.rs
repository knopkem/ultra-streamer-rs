//! WebSocket transport fallback for browsers without WebTransport support.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex as StdMutex};

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{WebSocketStream, accept_hdr_async};
use ustreamer_proto::frame::FramePacket;
use ustreamer_proto::input::InputEvent;

use crate::{InputReliability, ReceivedInput, TransportError};

/// An accepted WebSocket session upgrade.
pub struct AcceptedWebSocketSession {
    /// Path requested by the client (for example `/stream`).
    pub path: String,
    /// Established WebSocket session.
    pub session: WebSocketSession,
}

/// A TCP listener that upgrades incoming requests to WebSocket sessions.
pub struct WebSocketServer {
    listener: TcpListener,
}

impl WebSocketServer {
    /// Bind a WebSocket fallback endpoint on the provided TCP address.
    pub async fn bind(bind_address: SocketAddr) -> Result<Self, TransportError> {
        let listener = TcpListener::bind(bind_address)
            .await
            .map_err(|err| TransportError::InitFailed(err.to_string()))?;
        Ok(Self { listener })
    }

    /// Returns the local socket address of the bound TCP listener.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept the next WebSocket session and complete the upgrade handshake.
    pub async fn accept_session(&self) -> Result<AcceptedWebSocketSession, TransportError> {
        let (stream, remote_address) = self
            .listener
            .accept()
            .await
            .map_err(|err| TransportError::ConnectionFailed(err.to_string()))?;

        let path = Arc::new(StdMutex::new(None::<String>));
        let path_capture = Arc::clone(&path);
        let websocket = accept_hdr_async(stream, move |request: &Request, response: Response| {
            if let Ok(mut slot) = path_capture.lock() {
                *slot = Some(request.uri().path().to_owned());
            }
            Ok(response)
        })
        .await
        .map_err(|err| TransportError::ConnectionFailed(err.to_string()))?;

        let path = path
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
            .unwrap_or_else(|| "/".to_owned());
        let (writer, reader) = websocket.split();

        Ok(AcceptedWebSocketSession {
            path,
            session: WebSocketSession {
                writer: Arc::new(Mutex::new(writer)),
                reader: Arc::new(Mutex::new(reader)),
                remote_address,
            },
        })
    }
}

/// Established WebSocket fallback session.
#[derive(Clone)]
pub struct WebSocketSession {
    writer: Arc<Mutex<SplitSink<WebSocketStream<TcpStream>, Message>>>,
    reader: Arc<Mutex<SplitStream<WebSocketStream<TcpStream>>>>,
    remote_address: SocketAddr,
}

impl WebSocketSession {
    /// Current peer address.
    pub fn remote_address(&self) -> SocketAddr {
        self.remote_address
    }

    /// Send a single packetized frame fragment over WebSocket binary transport.
    pub async fn send_frame_packet(&self, packet: &FramePacket) -> Result<(), TransportError> {
        self.send_message(Message::Binary(packet.to_bytes().into()))
            .await
    }

    /// Send a batch of frame fragments in order.
    pub async fn send_frame_packets(&self, packets: &[FramePacket]) -> Result<(), TransportError> {
        for packet in packets {
            self.send_frame_packet(packet).await?;
        }

        Ok(())
    }

    /// Send a reliable UTF-8 control message to the browser.
    pub async fn send_control_message(&self, payload: &[u8]) -> Result<(), TransportError> {
        let text = String::from_utf8(payload.to_vec())
            .map_err(|err| TransportError::StreamIo(format!("control payload was not utf-8: {err}")))?;
        self.send_message(Message::Text(text.into())).await
    }

    /// Receive the next reliable input event from the browser.
    pub async fn recv_reliable_input(&self) -> Result<InputEvent, TransportError> {
        let bytes = self.recv_binary_message().await?;
        InputEvent::from_bytes(&bytes)
            .map_err(|err| TransportError::InvalidInputEvent(err.to_string()))
    }

    /// Receive the next input event from the browser.
    pub async fn recv_input(&self) -> Result<ReceivedInput, TransportError> {
        Ok(ReceivedInput {
            reliability: InputReliability::Reliable,
            event: self.recv_reliable_input().await?,
        })
    }

    async fn send_message(&self, message: Message) -> Result<(), TransportError> {
        let mut writer = self.writer.lock().await;
        writer
            .send(message)
            .await
            .map_err(|err| TransportError::StreamIo(err.to_string()))
    }

    async fn recv_binary_message(&self) -> Result<Vec<u8>, TransportError> {
        loop {
            let next_message = {
                let mut reader = self.reader.lock().await;
                reader.next().await
            };

            match next_message {
                Some(Ok(Message::Binary(bytes))) => return Ok(bytes.to_vec()),
                Some(Ok(Message::Text(_))) => {
                    return Err(TransportError::InvalidInputEvent(
                        "expected binary input event over WebSocket".into(),
                    ))
                }
                Some(Ok(Message::Close(_))) | None => return Err(TransportError::SessionClosed),
                Some(Ok(Message::Ping(_)))
                | Some(Ok(Message::Pong(_)))
                | Some(Ok(Message::Frame(_))) => continue,
                Some(Err(err)) => return Err(TransportError::StreamIo(err.to_string())),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, anyhow};
    use futures_util::{SinkExt, StreamExt};
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
    use tokio::time::{Duration, timeout};
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message;

    use super::*;

    struct LoopbackPair {
        _server: WebSocketServer,
        server_session: WebSocketSession,
        client_socket: WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>,
        path: String,
    }

    async fn loopback_pair() -> Result<LoopbackPair> {
        let bind_address = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0));
        let server = WebSocketServer::bind(bind_address).await?;
        let port = server.local_addr()?.port();
        let url = format!("ws://127.0.0.1:{port}/stream");

        let (accepted, client) = tokio::join!(server.accept_session(), connect_async(url));
        let accepted = accepted?;
        let (client_socket, _) = client?;

        Ok(LoopbackPair {
            _server: server,
            server_session: accepted.session,
            client_socket,
            path: accepted.path,
        })
    }

    #[tokio::test]
    async fn accepts_websocket_session_and_receives_input() -> Result<()> {
        let mut pair = loopback_pair().await?;
        assert_eq!(pair.path, "/stream");

        let event = InputEvent::KeyDown { code: 0x0041 };
        pair.client_socket
            .send(Message::Binary(event.to_bytes().into()))
            .await?;

        let received = timeout(Duration::from_secs(2), pair.server_session.recv_reliable_input())
            .await??;
        match received {
            InputEvent::KeyDown { code } => assert_eq!(code, 0x0041),
            other => panic!("unexpected input event: {other:?}"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn sends_frame_packets_over_websocket_binary_messages() -> Result<()> {
        let mut pair = loopback_pair().await?;

        let packet = FramePacket {
            frame_id: 7,
            fragment_idx: 0,
            fragment_count: 1,
            timestamp_us: 999,
            is_keyframe: true,
            is_refine: true,
            is_lossless: true,
            payload: vec![1, 2, 3, 4],
        };

        pair.server_session.send_frame_packet(&packet).await?;

        let message = timeout(Duration::from_secs(2), pair.client_socket.next())
            .await?
            .transpose()?
            .ok_or_else(|| anyhow!("client websocket closed"))?;
        let Message::Binary(bytes) = message else {
            panic!("expected binary frame message");
        };
        let decoded = FramePacket::from_bytes(&bytes)?;
        assert_eq!(decoded.frame_id, 7);
        assert!(decoded.is_keyframe);
        assert!(decoded.is_refine);
        assert!(decoded.is_lossless);

        Ok(())
    }

    #[tokio::test]
    async fn sends_control_messages_as_text() -> Result<()> {
        let mut pair = loopback_pair().await?;

        pair.server_session
            .send_control_message(br#"{"type":"status","message":"ok"}"#)
            .await?;

        let message = timeout(Duration::from_secs(2), pair.client_socket.next())
            .await?
            .transpose()?
            .ok_or_else(|| anyhow!("client websocket closed"))?;
        let Message::Text(text) = message else {
            panic!("expected text control message");
        };
        assert_eq!(text, r#"{"type":"status","message":"ok"}"#);

        Ok(())
    }
}
