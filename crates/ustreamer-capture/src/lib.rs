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

/// Diagnostic checksum over canonical RGBA8 pixel bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameChecksum {
    pub width: u32,
    pub height: u32,
    pub rgba8_fnv1a64: u64,
}

impl FrameChecksum {
    pub fn hex_string(&self) -> String {
        format!("{:016x}", self.rgba8_fnv1a64)
    }
}

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

impl CapturedFrame {
    /// Compute a diagnostic checksum over canonical RGBA8 bytes when CPU pixel data is available.
    pub fn diagnostic_checksum(&self) -> Result<Option<FrameChecksum>, CaptureError> {
        match self {
            Self::CpuBuffer {
                data,
                width,
                height,
                stride,
                format,
            } => checksum_cpu_buffer(data, *width, *height, *stride, *format).map(Some),
            #[cfg(target_os = "macos")]
            Self::MetalPixelBuffer { .. } => Ok(None),
        }
    }
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
    #[error("invalid CPU buffer metadata: {0}")]
    InvalidCpuBuffer(String),
}

fn checksum_cpu_buffer(
    data: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    format: wgpu::TextureFormat,
) -> Result<FrameChecksum, CaptureError> {
    let row_bytes = match format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            width
                .checked_mul(4)
                .ok_or_else(|| CaptureError::InvalidCpuBuffer("row byte size overflow".into()))?
                as usize
        }
        other => return Err(CaptureError::UnsupportedFormat(other)),
    };

    let stride = stride as usize;
    if stride < row_bytes {
        return Err(CaptureError::InvalidCpuBuffer(format!(
            "stride {stride} is smaller than required row width {row_bytes}"
        )));
    }

    let required_len = stride
        .checked_mul(height as usize)
        .ok_or_else(|| CaptureError::InvalidCpuBuffer("buffer size overflow".into()))?;
    if data.len() < required_len {
        return Err(CaptureError::InvalidCpuBuffer(format!(
            "buffer length {} is smaller than expected {required_len}",
            data.len()
        )));
    }

    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for row in 0..height as usize {
        let row_start = row * stride;
        let row_data = &data[row_start..row_start + row_bytes];
        for pixel in row_data.chunks_exact(4) {
            hash = fnv1a64_byte(hash, pixel[2]);
            hash = fnv1a64_byte(hash, pixel[1]);
            hash = fnv1a64_byte(hash, pixel[0]);
            hash = fnv1a64_byte(hash, pixel[3]);
        }
    }

    Ok(FrameChecksum {
        width,
        height,
        rgba8_fnv1a64: hash,
    })
}

#[inline]
fn fnv1a64_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
}

#[cfg(test)]
mod tests {
    use super::{CapturedFrame, FrameChecksum};

    #[test]
    fn checksum_normalizes_bgra_to_rgba() {
        let frame = CapturedFrame::CpuBuffer {
            data: vec![1, 2, 3, 4],
            width: 1,
            height: 1,
            stride: 4,
            format: wgpu::TextureFormat::Bgra8Unorm,
        };

        let checksum = frame.diagnostic_checksum().unwrap().unwrap();
        assert_eq!(
            checksum,
            FrameChecksum {
                width: 1,
                height: 1,
                rgba8_fnv1a64: 0xdbd0_9687_ea36_bd25,
            }
        );
        assert_eq!(checksum.hex_string(), "dbd09687ea36bd25");
    }

    #[test]
    fn checksum_ignores_padding_bytes() {
        let frame = CapturedFrame::CpuBuffer {
            data: vec![1, 2, 3, 4, 7, 6, 5, 8, 255, 255, 255, 255],
            width: 2,
            height: 1,
            stride: 12,
            format: wgpu::TextureFormat::Bgra8Unorm,
        };

        let checksum = frame.diagnostic_checksum().unwrap().unwrap();
        assert_eq!(checksum.hex_string(), "608d68d9e0fd3305");
    }
}
