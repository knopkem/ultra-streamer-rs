//! Direct NVENC encoder groundwork for Vulkan external-memory frames.
//!
//! This module does not yet perform CUDA import or invoke the NVIDIA Video Codec
//! SDK. Instead, it validates exported Vulkan frames and translates them into the
//! input/resource descriptor shape that the next FFI slice will consume.

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

use ustreamer_capture::{CapturedFrame, VulkanExternalImage, VulkanExternalMemoryHandle};
use ustreamer_proto::quality::{EncodeMode, EncodeParams};

use crate::{DecoderConfig, EncodeError, EncodedFrame, FrameEncoder};

const DEFAULT_HEVC_CODEC: &str = "hvc1.1.6.L153.B0";
const DEFAULT_AV1_CODEC: &str = "av01.0.08M.08";

/// Preferred direct NVENC codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencCodec {
    Hevc,
    Av1,
}

/// Exported Vulkan texture format as seen by the future CUDA/NVENC import step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencInputFormat {
    Bgra8,
    Rgba8,
}

/// Exported OS handle representation to hand off to CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencExternalMemoryHandleDescriptor {
    #[cfg(target_os = "linux")]
    OpaqueFd(i32),
    #[cfg(target_os = "windows")]
    OpaqueWin32Handle(usize),
}

/// Encode-side view of an exported Vulkan image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvencExternalImageDescriptor {
    pub resource_id: u64,
    pub width: u32,
    pub height: u32,
    pub allocation_size: u64,
    pub format: NvencInputFormat,
    pub memory_handle: NvencExternalMemoryHandleDescriptor,
}

/// Encode settings derived from the adaptive quality controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvencRateControl {
    pub target_fps: u32,
    pub average_bitrate_bps: u64,
    pub max_bitrate_bps: u64,
    pub mode: EncodeMode,
    pub force_keyframe: bool,
    pub request_lossless: bool,
}

/// Fully prepared frame description for the future CUDA/NVENC FFI layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvencPreparedFrame {
    pub codec: NvencCodec,
    pub input: NvencExternalImageDescriptor,
    pub rate_control: NvencRateControl,
}

/// Static configuration for the direct NVENC backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvencEncoderConfig {
    pub codec: NvencCodec,
    pub hevc_codec_string: String,
    pub av1_codec_string: String,
}

impl Default for NvencEncoderConfig {
    fn default() -> Self {
        Self {
            codec: NvencCodec::Hevc,
            hevc_codec_string: DEFAULT_HEVC_CODEC.into(),
            av1_codec_string: DEFAULT_AV1_CODEC.into(),
        }
    }
}

/// Direct NVENC backend entry point.
#[derive(Debug, Default)]
pub struct NvencEncoder {
    config: NvencEncoderConfig,
}

impl NvencEncoder {
    pub fn new() -> Self {
        Self::with_config(NvencEncoderConfig::default())
    }

    pub fn with_config(config: NvencEncoderConfig) -> Self {
        Self { config }
    }

    /// Translate an exported Vulkan frame into the descriptor shape the CUDA/NVENC
    /// interop layer will consume next.
    pub fn prepare_frame(
        &self,
        frame: &CapturedFrame,
        params: &EncodeParams,
    ) -> Result<NvencPreparedFrame, EncodeError> {
        let image = match frame {
            CapturedFrame::VulkanExternalImage(image) => image,
            other => {
                return Err(EncodeError::UnsupportedFrame(format!(
                    "NVENC direct backend requires CapturedFrame::VulkanExternalImage, got {other_kind}",
                    other_kind = captured_frame_kind(other)
                )));
            }
        };

        validate_dimensions(image, params)?;
        let input = prepare_external_image_descriptor(image)?;
        let rate_control = NvencRateControl {
            target_fps: params.target_fps.max(1),
            average_bitrate_bps: params.bitrate_bps,
            max_bitrate_bps: params.max_bitrate_bps.max(params.bitrate_bps),
            mode: params.mode,
            force_keyframe: params.force_keyframe,
            request_lossless: matches!(params.mode, EncodeMode::LosslessRefine),
        };

        Ok(NvencPreparedFrame {
            codec: self.config.codec,
            input,
            rate_control,
        })
    }

    pub fn codec(&self) -> NvencCodec {
        self.config.codec
    }

    fn codec_string(&self) -> &str {
        match self.config.codec {
            NvencCodec::Hevc => &self.config.hevc_codec_string,
            NvencCodec::Av1 => &self.config.av1_codec_string,
        }
    }
}

impl FrameEncoder for NvencEncoder {
    fn encode(
        &mut self,
        frame: &CapturedFrame,
        params: &EncodeParams,
    ) -> Result<EncodedFrame, EncodeError> {
        let prepared = self.prepare_frame(frame, params)?;
        Err(EncodeError::InitFailed(format!(
            "direct NVENC encode path is not wired yet; prepared {:?} {}x{} frame with codec {} but CUDA import/NVENC session creation remain pending",
            prepared.input.format,
            prepared.input.width,
            prepared.input.height,
            self.codec_string()
        )))
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, EncodeError> {
        Ok(Vec::new())
    }

    fn decoder_config(&self) -> Option<DecoderConfig> {
        None
    }
}

fn validate_dimensions(
    image: &VulkanExternalImage,
    params: &EncodeParams,
) -> Result<(), EncodeError> {
    if image.width() != params.width || image.height() != params.height {
        return Err(EncodeError::UnsupportedConfig(format!(
            "NVENCEncoder does not scale frames yet; exported Vulkan image is {}x{} but EncodeParams requested {}x{}",
            image.width(),
            image.height(),
            params.width,
            params.height
        )));
    }
    Ok(())
}

fn prepare_external_image_descriptor(
    image: &VulkanExternalImage,
) -> Result<NvencExternalImageDescriptor, EncodeError> {
    Ok(NvencExternalImageDescriptor {
        resource_id: image.resource_id(),
        width: image.width(),
        height: image.height(),
        allocation_size: image.allocation_size(),
        format: map_input_format(image.format())?,
        memory_handle: map_memory_handle(image.memory_handle())?,
    })
}

fn map_input_format(format: wgpu::TextureFormat) -> Result<NvencInputFormat, EncodeError> {
    match format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            Ok(NvencInputFormat::Bgra8)
        }
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
            Ok(NvencInputFormat::Rgba8)
        }
        other => Err(EncodeError::UnsupportedConfig(format!(
            "unsupported NVENC Vulkan input texture format {other:?}"
        ))),
    }
}

fn map_memory_handle(
    handle: &VulkanExternalMemoryHandle,
) -> Result<NvencExternalMemoryHandleDescriptor, EncodeError> {
    match handle {
        #[cfg(target_os = "linux")]
        VulkanExternalMemoryHandle::OpaqueFd(fd) => Ok(
            NvencExternalMemoryHandleDescriptor::OpaqueFd(fd.as_raw_fd()),
        ),
        #[cfg(target_os = "windows")]
        _ => Err(EncodeError::UnsupportedConfig(
            "Windows Vulkan external-memory import is not implemented yet".into(),
        )),
    }
}

fn captured_frame_kind(frame: &CapturedFrame) -> &'static str {
    match frame {
        CapturedFrame::CpuBuffer { .. } => "CpuBuffer",
        #[cfg(target_os = "macos")]
        CapturedFrame::MetalPixelBuffer { .. } => "MetalPixelBuffer",
        #[cfg(all(
            feature = "nvenc-direct",
            any(target_os = "linux", target_os = "windows")
        ))]
        CapturedFrame::VulkanExternalImage(..) => "VulkanExternalImage",
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use ustreamer_capture::VulkanExternalMemoryHandle;
    use ustreamer_proto::quality::{EncodeMode, EncodeParams};

    use super::{NvencEncoder, NvencExternalMemoryHandleDescriptor, NvencInputFormat};
    use ustreamer_capture::{CapturedFrame, VulkanExternalImage};

    #[test]
    fn rejects_non_vulkan_frames() {
        let frame = CapturedFrame::CpuBuffer {
            data: vec![0; 4],
            width: 1,
            height: 1,
            stride: 4,
            format: wgpu::TextureFormat::Bgra8Unorm,
        };

        let error = NvencEncoder::new()
            .prepare_frame(&frame, &EncodeParams::default())
            .unwrap_err();
        assert!(
            matches!(error, crate::EncodeError::UnsupportedFrame(message) if message.contains("CpuBuffer"))
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn prepares_linux_exported_vulkan_frame() {
        let fd = File::open("/dev/null").unwrap();
        let image = unsafe {
            VulkanExternalImage::from_raw_export_for_test(
                7,
                0x1234,
                0x5678,
                16 * 1024 * 1024,
                1920,
                1080,
                wgpu::TextureFormat::Bgra8Unorm,
                VulkanExternalMemoryHandle::OpaqueFd(fd.into()),
            )
        };
        let frame = CapturedFrame::VulkanExternalImage(image);
        let params = EncodeParams {
            width: 1920,
            height: 1080,
            target_fps: 60,
            bitrate_bps: 40_000_000,
            max_bitrate_bps: 90_000_000,
            mode: EncodeMode::LosslessRefine,
            force_keyframe: true,
        };

        let prepared = NvencEncoder::new().prepare_frame(&frame, &params).unwrap();
        assert_eq!(prepared.input.resource_id, 7);
        assert_eq!(prepared.input.width, 1920);
        assert_eq!(prepared.input.height, 1080);
        assert_eq!(prepared.input.allocation_size, 16 * 1024 * 1024);
        assert_eq!(prepared.input.format, NvencInputFormat::Bgra8);
        match prepared.input.memory_handle {
            NvencExternalMemoryHandleDescriptor::OpaqueFd(fd) => assert!(fd >= 0),
        }
        assert_eq!(prepared.rate_control.target_fps, 60);
        assert_eq!(prepared.rate_control.average_bitrate_bps, 40_000_000);
        assert_eq!(prepared.rate_control.max_bitrate_bps, 90_000_000);
        assert!(prepared.rate_control.force_keyframe);
        assert!(prepared.rate_control.request_lossless);
    }
}
