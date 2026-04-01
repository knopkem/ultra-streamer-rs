//! Hardware video encoding backends.
//!
//! Provides a trait [`FrameEncoder`] with platform-specific implementations:
//! - **VideoToolbox** (macOS): H.265 Main10 via `VTCompressionSession`
//! - **NVENC** (NVIDIA): H.265/AV1 via NVIDIA Video Codec SDK
//! - **GStreamer** (fallback): cross-platform via GStreamer pipeline

use ustreamer_capture::CapturedFrame;
use ustreamer_proto::quality::EncodeParams;

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
