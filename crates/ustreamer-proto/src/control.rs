//! JSON control messages shared by the host and browser client.

use serde::{Deserialize, Serialize};

/// Reliable control messages sent over WebTransport streams or WebSocket text frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ControlMessage {
    DecoderConfig(DecoderConfigMessage),
    FrameChecksum(FrameChecksumMessage),
    Status(StatusMessage),
    SessionMetrics(SessionMetricsMessage),
}

impl ControlMessage {
    /// Serialize the control message to UTF-8 JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Parse a control message from UTF-8 JSON bytes.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

/// Browser decoder configuration pushed by the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderConfigMessage {
    pub codec: String,
    #[serde(rename = "hardwareAcceleration")]
    pub hardware_acceleration: String,
    #[serde(rename = "optimizeForLatency")]
    pub optimize_for_latency: bool,
    #[serde(rename = "codedWidth", skip_serializing_if = "Option::is_none")]
    pub coded_width: Option<u32>,
    #[serde(rename = "codedHeight", skip_serializing_if = "Option::is_none")]
    pub coded_height: Option<u32>,
    #[serde(
        rename = "descriptionBase64",
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_base64_bytes"
    )]
    pub description: Option<Vec<u8>>,
}

impl DecoderConfigMessage {
    /// Create a low-latency decoder configuration using hardware decode when possible.
    pub fn low_latency(codec: impl Into<String>) -> Self {
        Self {
            codec: codec.into(),
            hardware_acceleration: "prefer-hardware".to_owned(),
            optimize_for_latency: true,
            coded_width: None,
            coded_height: None,
            description: None,
        }
    }

    /// Attach coded dimensions to the decoder configuration.
    pub fn with_dimensions(mut self, coded_width: u32, coded_height: u32) -> Self {
        self.coded_width = Some(coded_width);
        self.coded_height = Some(coded_height);
        self
    }

    /// Attach codec-specific description bytes, serialized as base64 in JSON.
    pub fn with_description(mut self, description: Vec<u8>) -> Self {
        self.description = Some(description);
        self
    }
}

/// Diagnostic frame checksum sent alongside refine/lossless frames.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameChecksumMessage {
    #[serde(rename = "frameId")]
    pub frame_id: u32,
    pub algorithm: String,
    #[serde(rename = "hashHex")]
    pub hash_hex: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

impl FrameChecksumMessage {
    pub fn rgba8_fnv1a64(frame_id: u32, hash_hex: impl Into<String>) -> Self {
        Self {
            frame_id,
            algorithm: "fnv1a64-rgba8".to_owned(),
            hash_hex: hash_hex.into(),
            width: None,
            height: None,
        }
    }

    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }
}

/// Human-readable status text pushed to the browser.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusMessage {
    pub message: String,
}

impl StatusMessage {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Optional telemetry that can drive the browser-side performance dashboard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SessionMetricsMessage {
    #[serde(rename = "encodeTimeUs", skip_serializing_if = "Option::is_none")]
    pub encode_time_us: Option<u64>,
    #[serde(rename = "transportRttMs", skip_serializing_if = "Option::is_none")]
    pub transport_rtt_ms: Option<f32>,
}

impl SessionMetricsMessage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_encode_time_us(mut self, encode_time_us: u64) -> Self {
        self.encode_time_us = Some(encode_time_us);
        self
    }

    pub fn with_transport_rtt_ms(mut self, transport_rtt_ms: f32) -> Self {
        self.transport_rtt_ms = Some(transport_rtt_ms);
        self
    }
}

mod optional_base64_bytes {
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(bytes) => serializer.serialize_some(&BASE64.encode(bytes)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = Option::<String>::deserialize(deserializer)?;
        encoded
            .map(|value| BASE64.decode(value).map_err(serde::de::Error::custom))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ControlMessage, DecoderConfigMessage, FrameChecksumMessage, SessionMetricsMessage,
        StatusMessage,
    };

    #[test]
    fn roundtrips_decoder_config_message_with_base64_description() {
        let message = ControlMessage::DecoderConfig(
            DecoderConfigMessage::low_latency("hvc1.1.6.L153.B0")
                .with_dimensions(1920, 1080)
                .with_description(vec![0x01, 0x02, 0x03]),
        );

        let bytes = message.to_bytes().unwrap();
        let json = String::from_utf8(bytes.clone()).unwrap();
        assert!(json.contains("\"type\":\"decoder-config\""));
        assert!(json.contains("\"codec\":\"hvc1.1.6.L153.B0\""));
        assert!(json.contains("\"codedWidth\":1920"));
        assert!(json.contains("\"codedHeight\":1080"));
        assert!(json.contains("\"descriptionBase64\":\"AQID\""));

        let decoded = ControlMessage::from_slice(&bytes).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn serializes_session_metrics_message() {
        let message = ControlMessage::SessionMetrics(
            SessionMetricsMessage::new()
                .with_encode_time_us(1_750)
                .with_transport_rtt_ms(2.5),
        );

        let bytes = message.to_bytes().unwrap();
        let json = String::from_utf8(bytes).unwrap();
        assert!(json.contains("\"type\":\"session-metrics\""));
        assert!(json.contains("\"encodeTimeUs\":1750"));
        assert!(json.contains("\"transportRttMs\":2.5"));
    }

    #[test]
    fn serializes_frame_checksum_message() {
        let message = ControlMessage::FrameChecksum(
            FrameChecksumMessage::rgba8_fnv1a64(7, "0123456789abcdef").with_dimensions(1920, 1080),
        );

        let bytes = message.to_bytes().unwrap();
        let json = String::from_utf8(bytes.clone()).unwrap();
        assert!(json.contains("\"type\":\"frame-checksum\""));
        assert!(json.contains("\"frameId\":7"));
        assert!(json.contains("\"algorithm\":\"fnv1a64-rgba8\""));
        assert!(json.contains("\"hashHex\":\"0123456789abcdef\""));
        assert!(json.contains("\"width\":1920"));
        assert!(json.contains("\"height\":1080"));

        let decoded = ControlMessage::from_slice(&bytes).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn parses_status_message_from_json() {
        let decoded =
            ControlMessage::from_slice(br#"{"type":"status","message":"ready"}"#).unwrap();

        assert_eq!(decoded, ControlMessage::Status(StatusMessage::new("ready")));
    }
}
