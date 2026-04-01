//! Hardware video encoding backends.
//!
//! Provides a trait [`FrameEncoder`] with platform-specific implementations:
//! - **VideoToolbox** (macOS): H.265 Main10 via `VTCompressionSession`
//! - **NVENC** (NVIDIA): H.265/AV1 via NVIDIA Video Codec SDK
//! - **GStreamer** (fallback): cross-platform via GStreamer pipeline

#[cfg(target_os = "macos")]
pub mod videotoolbox;

use ustreamer_capture::CapturedFrame;
use ustreamer_proto::{
    control::{ControlMessage, DecoderConfigMessage},
    quality::EncodeParams,
};

/// Encoded output from a single frame.
#[derive(Debug)]
pub struct EncodedFrame {
    /// Raw NALUs (H.265) or OBUs (AV1).
    pub data: Vec<u8>,
    /// Whether this is a keyframe.
    pub is_keyframe: bool,
    /// Whether this frame was encoded losslessly.
    pub is_lossless: bool,
    /// Encode duration in microseconds.
    pub encode_time_us: u64,
}

/// Browser decoder configuration for the current encoded stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderConfig {
    /// RFC 6381 codec string, for example `hvc1.1.6.L153.B0`.
    pub codec: String,
    /// Optional codec-specific description payload (for example `hvcC`).
    pub description: Option<Vec<u8>>,
    /// Encoded frame width.
    pub coded_width: u32,
    /// Encoded frame height.
    pub coded_height: u32,
}

impl DecoderConfig {
    /// Build the typed control message expected by the browser client.
    pub fn to_control_message(&self) -> ControlMessage {
        let mut message = DecoderConfigMessage::low_latency(self.codec.clone())
            .with_dimensions(self.coded_width, self.coded_height);
        if let Some(description) = &self.description {
            message = message.with_description(description.clone());
        }

        ControlMessage::DecoderConfig(message)
    }

    /// Build the JSON control message expected by the browser client.
    pub fn to_control_message_bytes(&self) -> Vec<u8> {
        self.to_control_message()
            .to_bytes()
            .expect("decoder control message should serialize")
    }
}

/// Trait for hardware video encoder implementations.
pub trait FrameEncoder: Send {
    /// Encode a captured frame with the given parameters.
    fn encode(
        &mut self,
        frame: &CapturedFrame,
        params: &EncodeParams,
    ) -> Result<EncodedFrame, EncodeError>;

    /// Flush any buffered frames.
    fn flush(&mut self) -> Result<Vec<EncodedFrame>, EncodeError>;

    /// Return the browser decoder configuration for the current stream, if known.
    fn decoder_config(&self) -> Option<DecoderConfig> {
        None
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("encoder initialization failed: {0}")]
    InitFailed(String),
    #[error("encoding failed: {0}")]
    EncodeFailed(String),
    #[error("unsupported configuration: {0}")]
    UnsupportedConfig(String),
}

#[cfg(test)]
mod tests {
    use super::DecoderConfig;
    use ustreamer_proto::control::ControlMessage;

    #[test]
    fn decoder_config_control_message_includes_base64_description() {
        let config = DecoderConfig {
            codec: "hvc1.1.6.L153.B0".into(),
            description: Some(vec![0x01, 0x02, 0x03]),
            coded_width: 1920,
            coded_height: 1080,
        };

        let message = config.to_control_message();
        let bytes = config.to_control_message_bytes();
        assert_eq!(ControlMessage::from_slice(&bytes).unwrap(), message);

        let json = String::from_utf8(bytes).unwrap();
        assert!(json.contains("\"type\":\"decoder-config\""));
        assert!(json.contains("\"codec\":\"hvc1.1.6.L153.B0\""));
        assert!(json.contains("\"codedWidth\":1920"));
        assert!(json.contains("\"codedHeight\":1080"));
        assert!(json.contains("\"descriptionBase64\":\"AQID\""));
    }

    #[test]
    fn decoder_config_control_message_omits_description_when_absent() {
        let config = DecoderConfig {
            codec: "av01.0.08M.08".into(),
            description: None,
            coded_width: 1280,
            coded_height: 720,
        };

        let bytes = config.to_control_message_bytes();
        let message = String::from_utf8(bytes.clone()).unwrap();
        assert!(message.contains("\"codec\":\"av01.0.08M.08\""));
        assert!(!message.contains("descriptionBase64"));
        assert_eq!(
            ControlMessage::from_slice(&bytes).unwrap(),
            config.to_control_message()
        );
    }
}
