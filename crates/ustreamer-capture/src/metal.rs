#![allow(unexpected_cfgs)]

//! macOS zero-copy capture using Metal texture interop and IOSurface wrapping.

use std::ptr::{self, NonNull};

use metal::TextureRef;
use objc::{msg_send, sel, sel_impl};
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVPixelBufferCreateWithIOSurface, kCVPixelFormatType_32BGRA, kCVPixelFormatType_64RGBAHalf,
    kCVReturnSuccess,
};
use objc2_io_surface::IOSurfaceRef;

use crate::{CaptureError, CapturedFrame, FrameCapture};

/// Zero-copy capture for Metal-backed `wgpu::Texture`s on macOS.
///
/// This path only succeeds when the input texture is IOSurface-backed.
/// That is the fastest route into VideoToolbox because the GPU texture can be
/// wrapped directly as a `CVPixelBuffer` with no CPU copy.
#[derive(Debug, Default)]
pub struct MetalCapture;

impl MetalCapture {
    /// Create a new zero-copy Metal capture instance.
    pub fn new() -> Self {
        Self
    }
}

impl FrameCapture for MetalCapture {
    #[allow(unexpected_cfgs)]
    fn capture(
        &mut self,
        _instance: &wgpu::Instance,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<CapturedFrame, CaptureError> {
        let expected_pixel_format = cv_pixel_format_for_texture(texture.format())
            .ok_or(CaptureError::UnsupportedFormat(texture.format()))?;

        let hal_texture =
            unsafe {
                texture.as_hal::<wgpu::hal::api::Metal>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu texture is not using Metal"),
                )?
            };

        let metal_texture = unsafe { hal_texture.raw_handle() };
        let texture_ref: &TextureRef = metal_texture;

        let surface_ptr = unsafe {
            #[allow(unexpected_cfgs)]
            let surface: *mut std::ffi::c_void = msg_send![texture_ref, iosurface];
            surface
        };
        let surface_ptr = NonNull::new(surface_ptr.cast::<IOSurfaceRef>())
            .ok_or(CaptureError::NotIosurfaceBacked)?;
        let surface = unsafe { CFRetained::retain(surface_ptr) };

        let width = surface.width();
        let height = surface.height();
        let stride = surface.bytes_per_row();
        let pixel_format = surface.pixel_format();

        if width == 0 || height == 0 || stride == 0 {
            return Err(CaptureError::InvalidSurface(format!(
                "width={width}, height={height}, stride={stride}"
            )));
        }

        if pixel_format != expected_pixel_format {
            return Err(CaptureError::InvalidSurface(format!(
                "IOSurface pixel format 0x{pixel_format:08x} does not match expected 0x{expected_pixel_format:08x}"
            )));
        }

        let mut pixel_buffer = ptr::null_mut();
        let status = unsafe {
            CVPixelBufferCreateWithIOSurface(None, &surface, None, NonNull::from(&mut pixel_buffer))
        };

        if status != kCVReturnSuccess {
            return Err(CaptureError::PixelBufferCreateFailed(status));
        }

        let pixel_buffer_ptr = NonNull::new(pixel_buffer).ok_or_else(|| {
            CaptureError::InvalidSurface("CVPixelBufferCreateWithIOSurface returned null".into())
        })?;
        let pixel_buffer = unsafe { CFRetained::from_raw(pixel_buffer_ptr) };

        Ok(CapturedFrame::MetalPixelBuffer {
            surface,
            pixel_buffer,
            width: width as u32,
            height: height as u32,
            stride: stride as u32,
            pixel_format,
        })
    }
}

fn cv_pixel_format_for_texture(format: wgpu::TextureFormat) -> Option<u32> {
    match format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            Some(kCVPixelFormatType_32BGRA)
        }
        wgpu::TextureFormat::Rgba16Float => Some(kCVPixelFormatType_64RGBAHalf),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::cv_pixel_format_for_texture;

    #[test]
    fn maps_bgra_to_corevideo_fourcc() {
        assert_eq!(
            cv_pixel_format_for_texture(wgpu::TextureFormat::Bgra8Unorm),
            Some(0x4247_5241)
        );
        assert_eq!(
            cv_pixel_format_for_texture(wgpu::TextureFormat::Bgra8UnormSrgb),
            Some(0x4247_5241)
        );
    }

    #[test]
    fn maps_rgba16f_to_half_float_fourcc() {
        assert_eq!(
            cv_pixel_format_for_texture(wgpu::TextureFormat::Rgba16Float),
            Some(0x5247_6841)
        );
    }

    #[test]
    fn rejects_unsupported_formats() {
        assert_eq!(
            cv_pixel_format_for_texture(wgpu::TextureFormat::Rgba8Unorm),
            None
        );
    }
}
