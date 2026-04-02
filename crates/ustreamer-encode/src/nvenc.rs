//! Direct NVENC encoder path for Vulkan external-memory frames.
//!
//! This module validates exported Vulkan frames, imports exported memory into
//! CUDA, and now wires a first real NVENC session slice that registers the
//! imported CUDA pointer per frame and reads back encoded HEVC/AV1 bitstreams.
//! The conservative `HostSynchronized` handoff remains the default runtime path;
//! decoder-config extraction, true GPU-driven semaphore handoff, and
//! backend-specific true-lossless refinement are still pending.

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(target_os = "windows")]
use std::os::windows::io::AsRawHandle;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::{
    ffi::{CStr, c_void},
    fs::File,
    mem::ManuallyDrop,
    ptr,
    sync::Arc,
    time::Instant,
};

#[cfg(any(target_os = "linux", target_os = "windows"))]
use cudarc::driver::{
    result::{self, DriverError as CudaDriverError},
    safe::CudaContext,
    sys,
};
#[cfg(any(target_os = "linux", target_os = "windows"))]
use nvidia_video_codec_sdk::{
    ENCODE_API,
    sys::nvEncodeAPI::{
        GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_AV1_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CONFIG,
        NV_ENC_CONFIG_VER, NV_ENC_CREATE_BITSTREAM_BUFFER, NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
        NV_ENC_DEVICE_TYPE, NV_ENC_INITIALIZE_PARAMS, NV_ENC_INITIALIZE_PARAMS_VER,
        NV_ENC_INPUT_RESOURCE_TYPE, NV_ENC_LOCK_BITSTREAM, NV_ENC_LOCK_BITSTREAM_VER,
        NV_ENC_MAP_INPUT_RESOURCE, NV_ENC_MAP_INPUT_RESOURCE_VER,
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
        NV_ENC_OUTPUT_PTR, NV_ENC_PARAMS_RC_MODE, NV_ENC_PIC_FLAGS, NV_ENC_PIC_PARAMS,
        NV_ENC_PIC_PARAMS_VER, NV_ENC_PIC_STRUCT, NV_ENC_PIC_TYPE, NV_ENC_PRESET_CONFIG,
        NV_ENC_PRESET_CONFIG_VER, NV_ENC_PRESET_P1_GUID, NV_ENC_REGISTER_RESOURCE,
        NV_ENC_TUNING_INFO, NVENCAPI_VERSION, NVENCSTATUS,
    },
};
use ustreamer_capture::{
    CapturedFrame, VulkanExternalImage, VulkanExternalMemoryHandle, VulkanExternalSync,
    VulkanExternalSyncHandle,
};
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

/// Exported OS synchronization handle representation to hand off to CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencExternalSyncHandleDescriptor {
    #[cfg(target_os = "linux")]
    OpaqueFd(i32),
    #[cfg(target_os = "windows")]
    OpaqueWin32Handle(usize),
}

/// Encode-side synchronization contract for an exported Vulkan image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NvencExternalSyncDescriptor {
    HostSynchronized,
    ExternalSemaphore {
        handle: NvencExternalSyncHandleDescriptor,
        value: u64,
    },
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
    pub sync: NvencExternalSyncDescriptor,
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
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    cuda_importer: Option<NvencCudaImporter>,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    nvenc_session: Option<NvencSessionState>,
}

impl NvencEncoder {
    pub fn new() -> Self {
        Self::with_config(NvencEncoderConfig::default())
    }

    pub fn with_config(config: NvencEncoderConfig) -> Self {
        Self {
            config,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            cuda_importer: None,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            nvenc_session: None,
        }
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn with_cuda_device(device_ordinal: usize) -> Result<Self, EncodeError> {
        Self::with_config_and_cuda_device(NvencEncoderConfig::default(), device_ordinal)
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn with_config_and_cuda_device(
        config: NvencEncoderConfig,
        device_ordinal: usize,
    ) -> Result<Self, EncodeError> {
        Ok(Self {
            config,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            cuda_importer: Some(NvencCudaImporter::new(device_ordinal)?),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            nvenc_session: None,
        })
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
        let sync = prepare_external_sync_descriptor(image.sync())?;
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
            sync,
            rate_control,
        })
    }

    pub fn codec(&self) -> NvencCodec {
        self.config.codec
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn import_to_cuda(
        &self,
        frame: &CapturedFrame,
        params: &EncodeParams,
    ) -> Result<NvencCudaImportedFrame, EncodeError> {
        let prepared = self.prepare_frame(frame, params)?;
        let importer = self.cuda_importer.as_ref().ok_or_else(|| {
            EncodeError::InitFailed(
                "CUDA importer is not configured; create NvencEncoder with with_cuda_device(...)"
                    .into(),
            )
        })?;
        let image = match frame {
            CapturedFrame::VulkanExternalImage(image) => image,
            _ => unreachable!("prepare_frame already validated the frame kind"),
        };

        importer.import_external_image(image, &prepared)
    }

    fn codec_string(&self) -> &str {
        match self.config.codec {
            NvencCodec::Hevc => &self.config.hevc_codec_string,
            NvencCodec::Av1 => &self.config.av1_codec_string,
        }
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn ensure_nvenc_session(&mut self, prepared: &NvencPreparedFrame) -> Result<(), EncodeError> {
        let descriptor = NvencSessionDescriptor::from_prepared(prepared);
        let needs_recreate = self
            .nvenc_session
            .as_ref()
            .map(|session| session.runtime.descriptor() != &descriptor)
            .unwrap_or(true);
        if !needs_recreate {
            return Ok(());
        }

        let ctx = self
            .cuda_importer
            .as_ref()
            .ok_or_else(|| {
                EncodeError::InitFailed(
                    "CUDA importer is not configured; create NvencEncoder with with_cuda_device(...)"
                        .into(),
                )
            })?
            .context()
            .clone();
        let runtime = NvencRuntimeSession::create(ctx, prepared)?;
        self.nvenc_session = Some(NvencSessionState {
            runtime,
            next_input_timestamp: 0,
            emit_parameter_sets: true,
        });
        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn encode_via_nvenc(
        &mut self,
        frame: &CapturedFrame,
        params: &EncodeParams,
        prepared: &NvencPreparedFrame,
    ) -> Result<EncodedFrame, EncodeError> {
        self.ensure_nvenc_session(prepared)?;
        let imported = self.import_to_cuda(frame, params)?;
        let session = self
            .nvenc_session
            .as_mut()
            .expect("NVENC session should exist after ensure_nvenc_session");
        let encoded = session.runtime.encode_imported_frame(
            imported,
            params,
            session.next_input_timestamp,
            session.emit_parameter_sets,
        )?;
        session.next_input_timestamp = session.next_input_timestamp.saturating_add(1);
        session.emit_parameter_sets = false;
        Ok(encoded)
    }
}

impl FrameEncoder for NvencEncoder {
    fn encode(
        &mut self,
        frame: &CapturedFrame,
        params: &EncodeParams,
    ) -> Result<EncodedFrame, EncodeError> {
        let prepared = self.prepare_frame(frame, params)?;
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if self.cuda_importer.is_some() {
            return self.encode_via_nvenc(frame, params, &prepared);
        }

        Err(EncodeError::InitFailed(format!(
            "direct NVENC encode path is not wired yet; prepared {:?} {}x{} frame with codec {} but CUDA import/NVENC session creation remain pending",
            prepared.input.format,
            prepared.input.width,
            prepared.input.height,
            self.codec_string()
        )))
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, EncodeError> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(mut session) = self.nvenc_session.take() {
            return session.runtime.flush();
        }
        Ok(Vec::new())
    }

    fn decoder_config(&self) -> Option<DecoderConfig> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(session) = &self.nvenc_session {
            return Some(DecoderConfig {
                codec: self.codec_string().to_owned(),
                description: None,
                coded_width: session.runtime.descriptor().width,
                coded_height: session.runtime.descriptor().height,
            });
        }
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

fn prepare_external_sync_descriptor(
    sync: &VulkanExternalSync,
) -> Result<NvencExternalSyncDescriptor, EncodeError> {
    match sync {
        VulkanExternalSync::HostSynchronized => Ok(NvencExternalSyncDescriptor::HostSynchronized),
        VulkanExternalSync::ExternalSemaphore { handle, value } => {
            Ok(NvencExternalSyncDescriptor::ExternalSemaphore {
                handle: map_sync_handle(handle)?,
                value: *value,
            })
        }
    }
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
        VulkanExternalMemoryHandle::OpaqueWin32Handle(handle) => Ok(
            NvencExternalMemoryHandleDescriptor::OpaqueWin32Handle(handle.as_raw_handle() as usize),
        ),
    }
}

fn map_sync_handle(
    handle: &VulkanExternalSyncHandle,
) -> Result<NvencExternalSyncHandleDescriptor, EncodeError> {
    match handle {
        #[cfg(target_os = "linux")]
        VulkanExternalSyncHandle::OpaqueFd(fd) => {
            Ok(NvencExternalSyncHandleDescriptor::OpaqueFd(fd.as_raw_fd()))
        }
        #[cfg(target_os = "windows")]
        VulkanExternalSyncHandle::OpaqueWin32Handle(handle) => Ok(
            NvencExternalSyncHandleDescriptor::OpaqueWin32Handle(handle.as_raw_handle() as usize),
        ),
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

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct NvencSessionDescriptor {
    codec: NvencCodec,
    width: u32,
    height: u32,
    format: NvencInputFormat,
    target_fps: u32,
    average_bitrate_bps: u64,
    max_bitrate_bps: u64,
    request_lossless: bool,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl NvencSessionDescriptor {
    fn from_prepared(prepared: &NvencPreparedFrame) -> Self {
        Self {
            codec: prepared.codec,
            width: prepared.input.width,
            height: prepared.input.height,
            format: prepared.input.format,
            target_fps: prepared.rate_control.target_fps,
            average_bitrate_bps: prepared.rate_control.average_bitrate_bps,
            max_bitrate_bps: prepared.rate_control.max_bitrate_bps,
            request_lossless: prepared.rate_control.request_lossless,
        }
    }

    fn codec_guid(&self) -> GUID {
        match self.codec {
            NvencCodec::Hevc => NV_ENC_CODEC_HEVC_GUID,
            NvencCodec::Av1 => NV_ENC_CODEC_AV1_GUID,
        }
    }

    fn buffer_format(&self) -> NV_ENC_BUFFER_FORMAT {
        match self.format {
            // NVENC's ARGB/ABGR enums describe little-endian 32-bit words.
            // BGRA8 bytes in memory correspond to the ARGB enum, and RGBA8 bytes
            // correspond to ABGR.
            NvencInputFormat::Bgra8 => NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
            NvencInputFormat::Rgba8 => NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR,
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug)]
struct NvencSessionState {
    runtime: NvencRuntimeSession,
    next_input_timestamp: u64,
    emit_parameter_sets: bool,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug)]
struct NvencRuntimeSession {
    ctx: Arc<CudaContext>,
    encoder: *mut c_void,
    output_bitstream: NV_ENC_OUTPUT_PTR,
    descriptor: NvencSessionDescriptor,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
unsafe impl Send for NvencRuntimeSession {}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl NvencRuntimeSession {
    fn create(ctx: Arc<CudaContext>, prepared: &NvencPreparedFrame) -> Result<Self, EncodeError> {
        let descriptor = NvencSessionDescriptor::from_prepared(prepared);
        ctx.bind_to_thread().map_err(|error| {
            EncodeError::InitFailed(format!(
                "failed to bind CUDA context before opening NVENC session: {error}"
            ))
        })?;

        let mut session = Self {
            ctx,
            encoder: ptr::null_mut(),
            output_bitstream: ptr::null_mut(),
            descriptor,
        };
        session.open_encode_session()?;
        session.initialize_encoder()?;
        Ok(session)
    }

    fn descriptor(&self) -> &NvencSessionDescriptor {
        &self.descriptor
    }

    fn open_encode_session(&mut self) -> Result<(), EncodeError> {
        let mut params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
            version: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
            deviceType: NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_CUDA,
            apiVersion: NVENCAPI_VERSION,
            device: self.ctx.cu_ctx().cast::<c_void>(),
            ..Default::default()
        };
        let mut encoder = ptr::null_mut();
        let status = unsafe { (ENCODE_API.open_encode_session_ex)(&mut params, &mut encoder) };
        self.encoder = encoder;
        nvenc_init_result(status, self.encoder, "failed to open NVENC encode session")
    }

    fn initialize_encoder(&mut self) -> Result<(), EncodeError> {
        let codec_guid = self.descriptor.codec_guid();
        let supported_codecs = query_nvenc_codecs(self.encoder)?;
        if !supported_codecs.contains(&codec_guid) {
            return Err(EncodeError::UnsupportedConfig(format!(
                "NVENC device does not support codec {:?}",
                self.descriptor.codec
            )));
        }

        let supported_inputs = query_nvenc_input_formats(self.encoder, codec_guid)?;
        if !supported_inputs.contains(&self.descriptor.buffer_format()) {
            return Err(EncodeError::UnsupportedConfig(format!(
                "NVENC device does not support {:?} input for codec {:?}",
                self.descriptor.format, self.descriptor.codec
            )));
        }

        let mut preset_config = query_nvenc_preset_config(
            self.encoder,
            codec_guid,
            NV_ENC_PRESET_P1_GUID,
            NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
        )?;
        preset_config.presetCfg.gopLength = self.descriptor.target_fps.max(1);
        preset_config.presetCfg.frameIntervalP = 1;
        preset_config.presetCfg.rcParams.rateControlMode =
            NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_VBR;
        preset_config.presetCfg.rcParams.averageBitRate =
            clamp_u64_to_u32(self.descriptor.average_bitrate_bps);
        preset_config.presetCfg.rcParams.maxBitRate =
            clamp_u64_to_u32(self.descriptor.max_bitrate_bps);

        let mut initialize_params = NV_ENC_INITIALIZE_PARAMS {
            version: NV_ENC_INITIALIZE_PARAMS_VER,
            encodeGUID: codec_guid,
            presetGUID: NV_ENC_PRESET_P1_GUID,
            encodeWidth: self.descriptor.width,
            encodeHeight: self.descriptor.height,
            darWidth: self.descriptor.width,
            darHeight: self.descriptor.height,
            frameRateNum: self.descriptor.target_fps.max(1),
            frameRateDen: 1,
            enableEncodeAsync: 0,
            enablePTD: 1,
            encodeConfig: &mut preset_config.presetCfg,
            maxEncodeWidth: self.descriptor.width,
            maxEncodeHeight: self.descriptor.height,
            tuningInfo: NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
            ..Default::default()
        };
        let status =
            unsafe { (ENCODE_API.initialize_encoder)(self.encoder, &mut initialize_params) };
        nvenc_init_result(status, self.encoder, "failed to initialize NVENC encoder")?;

        let mut create_bitstream = NV_ENC_CREATE_BITSTREAM_BUFFER {
            version: NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
            bitstreamBuffer: ptr::null_mut(),
            ..Default::default()
        };
        let status =
            unsafe { (ENCODE_API.create_bitstream_buffer)(self.encoder, &mut create_bitstream) };
        nvenc_init_result(
            status,
            self.encoder,
            "failed to create NVENC output bitstream",
        )?;
        self.output_bitstream = create_bitstream.bitstreamBuffer;
        Ok(())
    }

    fn encode_imported_frame(
        &mut self,
        imported: NvencCudaImportedFrame,
        params: &EncodeParams,
        input_timestamp: u64,
        emit_parameter_sets: bool,
    ) -> Result<EncodedFrame, EncodeError> {
        self.ctx.bind_to_thread().map_err(|error| {
            EncodeError::EncodeFailed(format!(
                "failed to bind CUDA context before NVENC encode: {error}"
            ))
        })?;
        let started_at = Instant::now();
        let mut registered = NvencRegisteredResource::register(
            self.encoder,
            &self.descriptor,
            imported.pitch_bytes,
            imported,
        )?;
        let force_keyframe = params.force_keyframe || emit_parameter_sets;
        let mut encode_pic_flags = 0u32;
        if force_keyframe {
            encode_pic_flags |= NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32;
            encode_pic_flags |= NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_OUTPUT_SPSPPS as u32;
        }
        let mut encode_pic_params = NV_ENC_PIC_PARAMS {
            version: NV_ENC_PIC_PARAMS_VER,
            inputWidth: self.descriptor.width,
            inputHeight: self.descriptor.height,
            inputPitch: registered.pitch(),
            inputBuffer: registered.mapped_resource(),
            outputBitstream: self.output_bitstream,
            bufferFmt: self.descriptor.buffer_format(),
            pictureStruct: NV_ENC_PIC_STRUCT::NV_ENC_PIC_STRUCT_FRAME,
            inputTimeStamp: input_timestamp,
            pictureType: if force_keyframe {
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR
            } else {
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_UNKNOWN
            },
            encodePicFlags: encode_pic_flags,
            ..Default::default()
        };
        let status = unsafe { (ENCODE_API.encode_picture)(self.encoder, &mut encode_pic_params) };
        nvenc_encode_result(status, self.encoder, "failed to encode NVENC picture")?;

        let output = self.read_output_bitstream()?;
        if output.data.is_empty() {
            return Err(EncodeError::EncodeFailed(
                "NVENC encode completed but produced an empty output bitstream".into(),
            ));
        }

        Ok(EncodedFrame {
            data: output.data,
            is_keyframe: matches!(
                output.picture_type,
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR | NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_I
            ),
            is_refine: matches!(params.mode, EncodeMode::LosslessRefine),
            is_lossless: false,
            encode_time_us: started_at.elapsed().as_micros().min(u64::MAX as u128) as u64,
        })
    }

    fn read_output_bitstream(&mut self) -> Result<NvencBitstreamData, EncodeError> {
        let mut lock_params = NV_ENC_LOCK_BITSTREAM {
            version: NV_ENC_LOCK_BITSTREAM_VER,
            outputBitstream: self.output_bitstream,
            ..Default::default()
        };
        let status = unsafe { (ENCODE_API.lock_bitstream)(self.encoder, &mut lock_params) };
        nvenc_encode_result(
            status,
            self.encoder,
            "failed to lock NVENC output bitstream",
        )?;

        let bitstream = unsafe {
            std::slice::from_raw_parts(
                lock_params.bitstreamBufferPtr.cast::<u8>(),
                lock_params.bitstreamSizeInBytes as usize,
            )
        }
        .to_vec();
        let picture_type = lock_params.pictureType;
        let unlock_status =
            unsafe { (ENCODE_API.unlock_bitstream)(self.encoder, self.output_bitstream) };
        nvenc_encode_result(
            unlock_status,
            self.encoder,
            "failed to unlock NVENC output bitstream",
        )?;
        Ok(NvencBitstreamData {
            data: bitstream,
            picture_type,
        })
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, EncodeError> {
        self.ctx.bind_to_thread().map_err(|error| {
            EncodeError::EncodeFailed(format!(
                "failed to bind CUDA context before flushing NVENC encoder: {error}"
            ))
        })?;
        let mut eos = NV_ENC_PIC_PARAMS::end_of_stream();
        let status = unsafe { (ENCODE_API.encode_picture)(self.encoder, &mut eos) };
        nvenc_encode_result(status, self.encoder, "failed to flush NVENC encoder")?;
        Ok(Vec::new())
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl Drop for NvencRuntimeSession {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if !self.output_bitstream.is_null() {
            let _ = unsafe {
                (ENCODE_API.destroy_bitstream_buffer)(self.encoder, self.output_bitstream)
            }
            .result_without_string();
        }
        if !self.encoder.is_null() {
            let _ = unsafe { (ENCODE_API.destroy_encoder)(self.encoder) }.result_without_string();
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug)]
struct NvencBitstreamData {
    data: Vec<u8>,
    picture_type: NV_ENC_PIC_TYPE,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug)]
struct NvencRegisteredResource<T> {
    encoder: *mut c_void,
    registered_resource: *mut c_void,
    mapped_resource: *mut c_void,
    pitch: u32,
    _marker: T,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl<T> NvencRegisteredResource<T> {
    fn register(
        encoder: *mut c_void,
        descriptor: &NvencSessionDescriptor,
        pitch: u32,
        marker: T,
    ) -> Result<Self, EncodeError>
    where
        T: AsRef<NvencCudaImportedFrame>,
    {
        let imported = marker.as_ref();
        let mut register_resource = NV_ENC_REGISTER_RESOURCE::new(
            NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
            descriptor.width,
            descriptor.height,
            imported.device_ptr as *mut c_void,
            descriptor.buffer_format(),
        )
        .pitch(pitch);
        let status = unsafe { (ENCODE_API.register_resource)(encoder, &mut register_resource) };
        nvenc_encode_result(
            status,
            encoder,
            "failed to register imported CUDA pointer with NVENC",
        )?;

        let mut map_input = NV_ENC_MAP_INPUT_RESOURCE {
            version: NV_ENC_MAP_INPUT_RESOURCE_VER,
            registeredResource: register_resource.registeredResource,
            mappedResource: ptr::null_mut(),
            mappedBufferFmt: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED,
            ..Default::default()
        };
        let status = unsafe { (ENCODE_API.map_input_resource)(encoder, &mut map_input) };
        nvenc_encode_result(status, encoder, "failed to map NVENC input resource")?;

        Ok(Self {
            encoder,
            registered_resource: register_resource.registeredResource,
            mapped_resource: map_input.mappedResource,
            pitch,
            _marker: marker,
        })
    }

    fn mapped_resource(&mut self) -> *mut c_void {
        self.mapped_resource
    }

    fn pitch(&self) -> u32 {
        self.pitch
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl<T> Drop for NvencRegisteredResource<T> {
    fn drop(&mut self) {
        if !self.mapped_resource.is_null() {
            let _ =
                unsafe { (ENCODE_API.unmap_input_resource)(self.encoder, self.mapped_resource) }
                    .result_without_string();
        }
        if !self.registered_resource.is_null() {
            let _ =
                unsafe { (ENCODE_API.unregister_resource)(self.encoder, self.registered_resource) }
                    .result_without_string();
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug)]
pub struct NvencCudaImporter {
    ctx: Arc<CudaContext>,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl NvencCudaImporter {
    pub fn new(device_ordinal: usize) -> Result<Self, EncodeError> {
        let ctx = CudaContext::new(device_ordinal).map_err(|error| {
            EncodeError::InitFailed(format!("failed to create CUDA context: {error}"))
        })?;
        Ok(Self { ctx })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn import_external_image(
        &self,
        image: &VulkanExternalImage,
        prepared: &NvencPreparedFrame,
    ) -> Result<NvencCudaImportedFrame, EncodeError> {
        let file = file_from_import_handle(image)?;
        let sync = import_cuda_sync(image.sync(), &self.ctx)?;
        let mapped_buffer =
            CudaExternalMemoryBuffer::import_dedicated(&self.ctx, file, image.allocation_size())
                .map_err(|error| {
                    EncodeError::EncodeFailed(format!(
                        "CUDA external-memory import failed for resource {}: {error}",
                        prepared.input.resource_id
                    ))
                })?;
        let stream = self.ctx.default_stream();
        sync.wait(&stream).map_err(|error| {
            EncodeError::EncodeFailed(format!(
                "CUDA synchronization handoff failed for resource {}: {error}",
                prepared.input.resource_id
            ))
        })?;

        Ok(NvencCudaImportedFrame {
            device_ptr: mapped_buffer.device_ptr,
            mapped_len: mapped_buffer.len(),
            pitch_bytes: row_bytes_for_input(prepared.input.format, prepared.input.width)?,
            width: prepared.input.width,
            height: prepared.input.height,
            format: prepared.input.format,
            sync: prepared.sync.clone(),
            _cuda_sync: sync,
            _mapped_buffer: mapped_buffer,
        })
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug)]
pub struct NvencCudaImportedFrame {
    pub device_ptr: u64,
    pub mapped_len: usize,
    pub pitch_bytes: u32,
    pub width: u32,
    pub height: u32,
    pub format: NvencInputFormat,
    pub sync: NvencExternalSyncDescriptor,
    _cuda_sync: NvencCudaSync,
    _mapped_buffer: CudaExternalMemoryBuffer,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl AsRef<NvencCudaImportedFrame> for NvencCudaImportedFrame {
    fn as_ref(&self) -> &NvencCudaImportedFrame {
        self
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug)]
struct CudaExternalMemoryBuffer {
    device_ptr: u64,
    len: usize,
    external_memory: sys::CUexternalMemory,
    ctx: Arc<CudaContext>,
    _file: ManuallyDrop<File>,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl CudaExternalMemoryBuffer {
    fn import_dedicated(
        ctx: &Arc<CudaContext>,
        file: File,
        size: u64,
    ) -> Result<Self, CudaDriverError> {
        ctx.bind_to_thread()?;
        let external_memory = unsafe { import_external_memory_dedicated(&file, size) }?;
        let device_ptr =
            unsafe { result::external_memory::get_mapped_buffer(external_memory, 0, size) }?;
        Ok(Self {
            device_ptr,
            len: size as usize,
            external_memory,
            ctx: ctx.clone(),
            _file: ManuallyDrop::new(file),
        })
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl Drop for CudaExternalMemoryBuffer {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = result::memory_free(self.device_ptr);
            let _ = result::external_memory::destroy_external_memory(self.external_memory);
        }
        #[cfg(target_os = "windows")]
        unsafe {
            ManuallyDrop::drop(&mut self._file);
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug)]
enum NvencCudaSync {
    HostSynchronized,
    ExternalSemaphore(CudaExternalSemaphore),
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl NvencCudaSync {
    fn wait(&self, stream: &cudarc::driver::safe::CudaStream) -> Result<(), CudaDriverError> {
        match self {
            Self::HostSynchronized => Ok(()),
            Self::ExternalSemaphore(semaphore) => semaphore.wait(stream),
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug)]
struct CudaExternalSemaphore {
    semaphore: sys::CUexternalSemaphore,
    wait_value: u64,
    ctx: Arc<CudaContext>,
    _file: ManuallyDrop<File>,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl CudaExternalSemaphore {
    fn import(
        ctx: &Arc<CudaContext>,
        handle: &VulkanExternalSyncHandle,
        wait_value: u64,
    ) -> Result<Self, EncodeError> {
        ctx.bind_to_thread().map_err(|error| {
            EncodeError::InitFailed(format!(
                "failed to bind CUDA context before external semaphore import: {error}"
            ))
        })?;
        let file = file_from_sync_handle(handle)?;
        let semaphore = import_external_semaphore_from_file(&file).map_err(|error| {
            EncodeError::EncodeFailed(format!(
                "failed to import external Vulkan semaphore into CUDA: {error}"
            ))
        })?;

        Ok(Self {
            semaphore,
            wait_value,
            ctx: ctx.clone(),
            _file: ManuallyDrop::new(file),
        })
    }

    fn wait(&self, stream: &cudarc::driver::safe::CudaStream) -> Result<(), CudaDriverError> {
        if self.ctx.cu_ctx() != stream.context().cu_ctx() {
            return Err(CudaDriverError(
                sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        self.ctx.bind_to_thread()?;
        let mut wait_params: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS =
            unsafe { std::mem::zeroed() };
        wait_params.params.fence.value = self.wait_value;
        unsafe {
            sys::cuWaitExternalSemaphoresAsync(&self.semaphore, &wait_params, 1, stream.cu_stream())
        }
        .result()
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl Drop for CudaExternalSemaphore {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = sys::cuDestroyExternalSemaphore(self.semaphore);
        }
        #[cfg(target_os = "windows")]
        unsafe {
            ManuallyDrop::drop(&mut self._file);
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn import_cuda_sync(
    sync: &VulkanExternalSync,
    ctx: &Arc<CudaContext>,
) -> Result<NvencCudaSync, EncodeError> {
    match sync {
        VulkanExternalSync::HostSynchronized => Ok(NvencCudaSync::HostSynchronized),
        VulkanExternalSync::ExternalSemaphore { handle, value } => Ok(
            NvencCudaSync::ExternalSemaphore(CudaExternalSemaphore::import(ctx, handle, *value)?),
        ),
    }
}

#[cfg(target_os = "linux")]
unsafe fn import_external_memory_dedicated(
    file: &File,
    size: u64,
) -> Result<sys::CUexternalMemory, CudaDriverError> {
    use std::mem::MaybeUninit;

    let mut external_memory = MaybeUninit::uninit();
    let handle_description = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
        handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
            fd: file.as_raw_fd(),
        },
        size,
        flags: sys::CUDA_EXTERNAL_MEMORY_DEDICATED,
        reserved: [0; 16],
    };
    unsafe { sys::cuImportExternalMemory(external_memory.as_mut_ptr(), &handle_description) }
        .result()?;
    Ok(unsafe { external_memory.assume_init() })
}

#[cfg(target_os = "windows")]
unsafe fn import_external_memory_dedicated(
    file: &File,
    size: u64,
) -> Result<sys::CUexternalMemory, CudaDriverError> {
    use std::mem::MaybeUninit;

    let mut external_memory = MaybeUninit::uninit();
    let handle_description = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
        handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: file.as_raw_handle(),
                name: std::ptr::null(),
            },
        },
        size,
        flags: sys::CUDA_EXTERNAL_MEMORY_DEDICATED,
        reserved: [0; 16],
    };
    unsafe { sys::cuImportExternalMemory(external_memory.as_mut_ptr(), &handle_description) }
        .result()?;
    Ok(unsafe { external_memory.assume_init() })
}

#[cfg(target_os = "linux")]
fn file_from_import_handle(image: &VulkanExternalImage) -> Result<File, EncodeError> {
    let cloned_fd = image.try_clone_opaque_fd().map_err(|error| {
        EncodeError::EncodeFailed(format!(
            "failed to clone exported Vulkan opaque FD for CUDA import: {error}"
        ))
    })?;
    Ok(File::from(cloned_fd))
}

#[cfg(target_os = "windows")]
fn file_from_import_handle(image: &VulkanExternalImage) -> Result<File, EncodeError> {
    let cloned_handle = image.try_clone_opaque_win32_handle().map_err(|error| {
        EncodeError::EncodeFailed(format!(
            "failed to clone exported Vulkan Win32 handle for CUDA import: {error}"
        ))
    })?;
    Ok(File::from(cloned_handle))
}

#[cfg(target_os = "linux")]
fn file_from_sync_handle(handle: &VulkanExternalSyncHandle) -> Result<File, EncodeError> {
    let cloned_fd = handle.try_clone_opaque_fd().map_err(|error| {
        EncodeError::EncodeFailed(format!(
            "failed to clone exported Vulkan sync FD for CUDA import: {error}"
        ))
    })?;
    Ok(File::from(cloned_fd))
}

#[cfg(target_os = "windows")]
fn file_from_sync_handle(handle: &VulkanExternalSyncHandle) -> Result<File, EncodeError> {
    let cloned_handle = handle.try_clone_opaque_win32_handle().map_err(|error| {
        EncodeError::EncodeFailed(format!(
            "failed to clone exported Vulkan sync handle for CUDA import: {error}"
        ))
    })?;
    Ok(File::from(cloned_handle))
}

#[cfg(target_os = "linux")]
fn import_external_semaphore_from_file(
    file: &File,
) -> Result<sys::CUexternalSemaphore, CudaDriverError> {
    let mut desc: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = unsafe { std::mem::zeroed() };
    desc.type_ =
        sys::CUexternalSemaphoreHandleType_enum::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
    desc.handle.fd = file.as_raw_fd();
    let mut semaphore: sys::CUexternalSemaphore = std::ptr::null_mut();
    unsafe { sys::cuImportExternalSemaphore(&mut semaphore, &desc) }.result()?;
    Ok(semaphore)
}

#[cfg(target_os = "windows")]
fn import_external_semaphore_from_file(
    file: &File,
) -> Result<sys::CUexternalSemaphore, CudaDriverError> {
    let mut desc: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = unsafe { std::mem::zeroed() };
    desc.type_ =
        sys::CUexternalSemaphoreHandleType_enum::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
    desc.handle.win32 = sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
        handle: file.as_raw_handle(),
        name: std::ptr::null(),
    };
    let mut semaphore: sys::CUexternalSemaphore = std::ptr::null_mut();
    unsafe { sys::cuImportExternalSemaphore(&mut semaphore, &desc) }.result()?;
    Ok(semaphore)
}

fn row_bytes_for_input(format: NvencInputFormat, width: u32) -> Result<u32, EncodeError> {
    let bytes_per_pixel = match format {
        NvencInputFormat::Bgra8 | NvencInputFormat::Rgba8 => 4u32,
    };
    width.checked_mul(bytes_per_pixel).ok_or_else(|| {
        EncodeError::UnsupportedConfig(format!(
            "NVENC input row-bytes overflow for width {width} and format {format:?}"
        ))
    })
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn query_nvenc_codecs(encoder: *mut c_void) -> Result<Vec<GUID>, EncodeError> {
    let mut supported_count = 0;
    let status = unsafe { (ENCODE_API.get_encode_guid_count)(encoder, &mut supported_count) };
    nvenc_init_result(status, encoder, "failed to query NVENC codec count")?;

    let mut actual_count = 0;
    let mut encode_guids = vec![GUID::default(); supported_count as usize];
    let status = unsafe {
        (ENCODE_API.get_encode_guids)(
            encoder,
            encode_guids.as_mut_ptr(),
            supported_count,
            &mut actual_count,
        )
    };
    nvenc_init_result(status, encoder, "failed to query NVENC codec GUIDs")?;
    encode_guids.truncate(actual_count as usize);
    Ok(encode_guids)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn query_nvenc_input_formats(
    encoder: *mut c_void,
    codec_guid: GUID,
) -> Result<Vec<NV_ENC_BUFFER_FORMAT>, EncodeError> {
    let mut format_count = 0;
    let status =
        unsafe { (ENCODE_API.get_input_format_count)(encoder, codec_guid, &mut format_count) };
    nvenc_init_result(status, encoder, "failed to query NVENC input-format count")?;

    let mut actual_count = 0;
    let mut formats =
        vec![NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED; format_count as usize];
    let status = unsafe {
        (ENCODE_API.get_input_formats)(
            encoder,
            codec_guid,
            formats.as_mut_ptr(),
            format_count,
            &mut actual_count,
        )
    };
    nvenc_init_result(status, encoder, "failed to query NVENC input formats")?;
    formats.truncate(actual_count as usize);
    Ok(formats)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn query_nvenc_preset_config(
    encoder: *mut c_void,
    codec_guid: GUID,
    preset_guid: GUID,
    tuning_info: NV_ENC_TUNING_INFO,
) -> Result<NV_ENC_PRESET_CONFIG, EncodeError> {
    let mut preset_config = NV_ENC_PRESET_CONFIG {
        version: NV_ENC_PRESET_CONFIG_VER,
        presetCfg: NV_ENC_CONFIG {
            version: NV_ENC_CONFIG_VER,
            ..Default::default()
        },
        ..Default::default()
    };
    let status = unsafe {
        (ENCODE_API.get_encode_preset_config_ex)(
            encoder,
            codec_guid,
            preset_guid,
            tuning_info,
            &mut preset_config,
        )
    };
    nvenc_init_result(
        status,
        encoder,
        "failed to query NVENC preset configuration",
    )?;
    Ok(preset_config)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn clamp_u64_to_u32(value: u64) -> u32 {
    value.min(u32::MAX as u64) as u32
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvenc_init_result(
    status: NVENCSTATUS,
    encoder: *mut c_void,
    action: &str,
) -> Result<(), EncodeError> {
    status
        .result_without_string()
        .map_err(|_| EncodeError::InitFailed(nvenc_status_message(status, encoder, action)))
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvenc_encode_result(
    status: NVENCSTATUS,
    encoder: *mut c_void,
    action: &str,
) -> Result<(), EncodeError> {
    status
        .result_without_string()
        .map_err(|_| EncodeError::EncodeFailed(nvenc_status_message(status, encoder, action)))
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvenc_status_message(status: NVENCSTATUS, encoder: *mut c_void, action: &str) -> String {
    let base = match status.result_without_string() {
        Ok(()) => return action.to_owned(),
        Err(error) => error.to_string(),
    };
    match nvenc_last_error_string(encoder) {
        Some(detail) => format!("{action}: {base}: {detail}"),
        None => format!("{action}: {base}"),
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvenc_last_error_string(encoder: *mut c_void) -> Option<String> {
    if encoder.is_null() {
        return None;
    }
    let ptr = unsafe { (ENCODE_API.get_last_error_string)(encoder) };
    if ptr.is_null() {
        return None;
    }
    let text = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .trim()
        .to_owned();
    if text.is_empty() { None } else { Some(text) }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use ustreamer_capture::VulkanExternalMemoryHandle;
    use ustreamer_proto::quality::{EncodeMode, EncodeParams};

    use super::{
        NvencEncoder, NvencExternalMemoryHandleDescriptor, NvencExternalSyncDescriptor,
        NvencInputFormat,
    };
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

    #[test]
    fn computes_row_bytes_for_rgba_inputs() {
        assert_eq!(
            super::row_bytes_for_input(NvencInputFormat::Bgra8, 1920).unwrap(),
            7680
        );
        assert_eq!(
            super::row_bytes_for_input(NvencInputFormat::Rgba8, 1).unwrap(),
            4
        );
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn maps_wgpu_rgba_layouts_to_nvenc_buffer_formats() {
        use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_BUFFER_FORMAT;

        let bgra = super::NvencSessionDescriptor {
            codec: super::NvencCodec::Hevc,
            width: 1,
            height: 1,
            format: NvencInputFormat::Bgra8,
            target_fps: 60,
            average_bitrate_bps: 1,
            max_bitrate_bps: 1,
            request_lossless: false,
        };
        let rgba = super::NvencSessionDescriptor {
            format: NvencInputFormat::Rgba8,
            ..bgra.clone()
        };

        assert_eq!(
            bgra.buffer_format(),
            NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB
        );
        assert_eq!(
            rgba.buffer_format(),
            NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR
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
        assert!(matches!(
            prepared.sync,
            NvencExternalSyncDescriptor::HostSynchronized
        ));
        assert_eq!(prepared.rate_control.target_fps, 60);
        assert_eq!(prepared.rate_control.average_bitrate_bps, 40_000_000);
        assert_eq!(prepared.rate_control.max_bitrate_bps, 90_000_000);
        assert!(prepared.rate_control.force_keyframe);
        assert!(prepared.rate_control.request_lossless);
    }
}
