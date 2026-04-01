//! GPU frame capture from wgpu render targets.
//!
//! Provides a trait [`FrameCapture`] with platform-specific implementations:
//! - **Metal/IOSurface** (macOS): zero-copy via `wgpu-hal` Metal interop
//! - **Vulkan/CUDA** (NVIDIA): zero-copy via external memory export
//! - **Staging buffer** (fallback): triple-buffered `copy_texture_to_buffer`

#[cfg(target_os = "macos")]
pub mod metal;
pub mod staging;

#[cfg(target_os = "macos")]
use objc2_core_foundation::CFRetained;
#[cfg(target_os = "macos")]
use objc2_core_video::CVPixelBuffer;
#[cfg(target_os = "macos")]
use objc2_io_surface::IOSurfaceRef;

/// Handle to a captured frame, ready for the encoder.
pub enum CapturedFrame {
    /// Raw pixel data in CPU memory (from staging buffer path).
    CpuBuffer {
        data: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
        format: wgpu::TextureFormat,
    },
    /// macOS zero-copy capture via IOSurface → CVPixelBuffer.
    #[cfg(target_os = "macos")]
    MetalPixelBuffer {
        /// Retained IOSurface backing the rendered Metal texture.
        surface: CFRetained<IOSurfaceRef>,
        /// CoreVideo wrapper around the IOSurface, ready for VideoToolbox input.
        pixel_buffer: CFRetained<CVPixelBuffer>,
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
        /// Row stride in bytes.
        stride: u32,
        /// CoreVideo / IOSurface pixel format fourcc.
        pixel_format: u32,
    },
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
    #[error("capture backend unsupported: {0}")]
    UnsupportedBackend(&'static str),
    #[error("texture is not backed by IOSurface")]
    NotIosurfaceBacked,
    #[error("failed to wrap IOSurface in CVPixelBuffer (status {0})")]
    PixelBufferCreateFailed(i32),
    #[error("invalid IOSurface metadata: {0}")]
    InvalidSurface(String),
}
