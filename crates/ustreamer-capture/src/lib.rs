//! GPU frame capture from wgpu render targets.
//!
//! Provides a trait [`FrameCapture`] with platform-specific implementations:
//! - **Metal/IOSurface** (macOS): zero-copy via `wgpu-hal` Metal interop
//! - **Vulkan/CUDA** (NVIDIA): zero-copy via external memory export
//! - **Staging buffer** (fallback): triple-buffered `copy_texture_to_buffer`

pub mod staging;

/// Handle to a captured frame, ready for the encoder.
pub enum CapturedFrame {
    /// Raw pixel data in CPU memory (from staging buffer path).
    CpuBuffer {
        data: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
    },
    // Future: platform-specific zero-copy handles
    // MetalIOSurface { surface: ... },
    // CudaMappedResource { ptr: ... },
}

/// Trait for frame capture implementations.
pub trait FrameCapture: Send + Sync {
    /// Capture the current render target contents.
    fn capture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<CapturedFrame, CaptureError>;
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("buffer mapping failed: {0}")]
    MapFailed(String),
    #[error("texture format unsupported: {0:?}")]
    UnsupportedFormat(wgpu::TextureFormat),
}
