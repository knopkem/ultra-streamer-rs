//! Direct NVENC encoder path for Vulkan external-memory frames.
//!
//! This module validates exported Vulkan frames, imports exported memory into
//! CUDA, and now wires a first real NVENC session slice that registers the
//! imported CUDA pointer per frame and reads back encoded HEVC/AV1 bitstreams.
//! The conservative `HostSynchronized` handoff remains the default runtime path;
//! HEVC output is normalized into length-prefixed access units with cached
//! `hvcC` decoder config for browser-side WebCodecs consumption, while true
//! GPU-driven semaphore handoff, AV1 decoder-config extraction, and
//! backend-specific true-lossless refinement remain pending.

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
    sync::{Arc, OnceLock},
    time::Instant,
};

#[cfg(any(target_os = "linux", target_os = "windows"))]
use cudarc::driver::{
    result::{self, DriverError as CudaDriverError},
    safe::CudaContext,
    sys,
};
#[cfg(any(target_os = "linux", target_os = "windows"))]
use libloading::Library;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use nvidia_video_codec_sdk::sys::nvEncodeAPI::{
    GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_AV1_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CONFIG,
    NV_ENC_CONFIG_VER, NV_ENC_CREATE_BITSTREAM_BUFFER, NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
    NV_ENC_DEVICE_TYPE, NV_ENC_INITIALIZE_PARAMS, NV_ENC_INITIALIZE_PARAMS_VER,
    NV_ENC_INPUT_RESOURCE_TYPE, NV_ENC_LOCK_BITSTREAM, NV_ENC_LOCK_BITSTREAM_VER,
    NV_ENC_MAP_INPUT_RESOURCE, NV_ENC_MAP_INPUT_RESOURCE_VER, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER, NV_ENC_OUTPUT_PTR, NV_ENC_PARAMS_RC_MODE,
    NV_ENC_PIC_FLAGS, NV_ENC_PIC_PARAMS, NV_ENC_PIC_PARAMS_VER, NV_ENC_PIC_STRUCT, NV_ENC_PIC_TYPE,
    NV_ENC_PRESET_CONFIG, NV_ENC_PRESET_CONFIG_VER, NV_ENC_PRESET_P1_GUID,
    NV_ENC_REGISTER_RESOURCE, NV_ENC_REGISTER_RESOURCE_VER, NV_ENC_SEQUENCE_PARAM_PAYLOAD,
    NV_ENC_TUNING_INFO, NV_ENCODE_API_FUNCTION_LIST, NV_ENCODE_API_FUNCTION_LIST_VER,
    NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION, NVENCAPI_VERSION, NVENCSTATUS,
};
use ustreamer_capture::{
    CapturedFrame, VulkanExternalImage, VulkanExternalMemoryHandle, VulkanExternalSync,
    VulkanExternalSyncHandle,
};
use ustreamer_proto::quality::{EncodeMode, EncodeParams};

use crate::{DecoderConfig, EncodeError, EncodedFrame, FrameEncoder};

const DEFAULT_HEVC_CODEC: &str = "hvc1.1.6.L153.B0";
const DEFAULT_AV1_CODEC: &str = "av01.0.08M.08";
const NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER: u32 = NVENCAPI_VERSION | (1 << 16) | (0x7 << 28);
const HEVC_NAL_TYPE_VPS: u8 = 32;
const HEVC_NAL_TYPE_SPS: u8 = 33;
const HEVC_NAL_TYPE_PPS: u8 = 34;
const HEVC_ACCESS_UNIT_LENGTH_BYTES: usize = 4;
const HEVC_HVCC_LENGTH_SIZE_MINUS_ONE: u8 = (HEVC_ACCESS_UNIT_LENGTH_BYTES - 1) as u8;
const NVENC_SEQUENCE_PAYLOAD_BUFFER_SIZE: usize = 4096;

#[cfg(target_os = "windows")]
const NVENC_RUNTIME_LIBRARY_CANDIDATES: &[&str] = &["nvEncodeAPI64.dll", "nvEncodeAPI.dll"];
#[cfg(target_os = "linux")]
const NVENC_RUNTIME_LIBRARY_CANDIDATES: &[&str] = &["libnvidia-encode.so.1", "libnvidia-encode.so"];

#[cfg(any(target_os = "linux", target_os = "windows"))]
const NVENC_API_MSG: &str =
    "The NVENC runtime should populate the required function table entries.";

#[cfg(any(target_os = "linux", target_os = "windows"))]
type NvEncodeApiGetMaxSupportedVersionFn = unsafe extern "C" fn(*mut u32) -> NVENCSTATUS;
#[cfg(any(target_os = "linux", target_os = "windows"))]
type NvEncodeApiCreateInstanceFn =
    unsafe extern "C" fn(*mut NV_ENCODE_API_FUNCTION_LIST) -> NVENCSTATUS;

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug, Clone)]
struct NvencApi {
    function_list: NV_ENCODE_API_FUNCTION_LIST,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
unsafe impl Send for NvencApi {}
#[cfg(any(target_os = "linux", target_os = "windows"))]
unsafe impl Sync for NvencApi {}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[allow(unsafe_op_in_unsafe_fn)]
impl NvencApi {
    unsafe fn open_encode_session_ex(
        &self,
        params: *mut NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
        encoder: *mut *mut c_void,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncOpenEncodeSessionEx
            .expect(NVENC_API_MSG))(params, encoder)
    }

    unsafe fn initialize_encoder(
        &self,
        encoder: *mut c_void,
        params: *mut NV_ENC_INITIALIZE_PARAMS,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncInitializeEncoder
            .expect(NVENC_API_MSG))(encoder, params)
    }

    unsafe fn create_bitstream_buffer(
        &self,
        encoder: *mut c_void,
        params: *mut NV_ENC_CREATE_BITSTREAM_BUFFER,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncCreateBitstreamBuffer
            .expect(NVENC_API_MSG))(encoder, params)
    }

    unsafe fn destroy_bitstream_buffer(
        &self,
        encoder: *mut c_void,
        bitstream: NV_ENC_OUTPUT_PTR,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncDestroyBitstreamBuffer
            .expect(NVENC_API_MSG))(encoder, bitstream)
    }

    unsafe fn encode_picture(
        &self,
        encoder: *mut c_void,
        params: *mut NV_ENC_PIC_PARAMS,
    ) -> NVENCSTATUS {
        (self.function_list.nvEncEncodePicture.expect(NVENC_API_MSG))(encoder, params)
    }

    unsafe fn lock_bitstream(
        &self,
        encoder: *mut c_void,
        params: *mut NV_ENC_LOCK_BITSTREAM,
    ) -> NVENCSTATUS {
        (self.function_list.nvEncLockBitstream.expect(NVENC_API_MSG))(encoder, params)
    }

    unsafe fn unlock_bitstream(
        &self,
        encoder: *mut c_void,
        bitstream: NV_ENC_OUTPUT_PTR,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncUnlockBitstream
            .expect(NVENC_API_MSG))(encoder, bitstream)
    }

    unsafe fn register_resource(
        &self,
        encoder: *mut c_void,
        params: *mut NV_ENC_REGISTER_RESOURCE,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncRegisterResource
            .expect(NVENC_API_MSG))(encoder, params)
    }

    unsafe fn map_input_resource(
        &self,
        encoder: *mut c_void,
        params: *mut NV_ENC_MAP_INPUT_RESOURCE,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncMapInputResource
            .expect(NVENC_API_MSG))(encoder, params)
    }

    unsafe fn unmap_input_resource(
        &self,
        encoder: *mut c_void,
        mapped_resource: *mut c_void,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncUnmapInputResource
            .expect(NVENC_API_MSG))(encoder, mapped_resource)
    }

    unsafe fn unregister_resource(
        &self,
        encoder: *mut c_void,
        registered_resource: *mut c_void,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncUnregisterResource
            .expect(NVENC_API_MSG))(encoder, registered_resource)
    }

    unsafe fn destroy_encoder(&self, encoder: *mut c_void) -> NVENCSTATUS {
        (self.function_list.nvEncDestroyEncoder.expect(NVENC_API_MSG))(encoder)
    }

    unsafe fn get_encode_guid_count(
        &self,
        encoder: *mut c_void,
        supported_count: *mut u32,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncGetEncodeGUIDCount
            .expect(NVENC_API_MSG))(encoder, supported_count)
    }

    unsafe fn get_encode_guids(
        &self,
        encoder: *mut c_void,
        guid_buffer: *mut GUID,
        guid_array_size: u32,
        actual_count: *mut u32,
    ) -> NVENCSTATUS {
        (self.function_list.nvEncGetEncodeGUIDs.expect(NVENC_API_MSG))(
            encoder,
            guid_buffer,
            guid_array_size,
            actual_count,
        )
    }

    unsafe fn get_input_format_count(
        &self,
        encoder: *mut c_void,
        codec_guid: GUID,
        format_count: *mut u32,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncGetInputFormatCount
            .expect(NVENC_API_MSG))(encoder, codec_guid, format_count)
    }

    unsafe fn get_input_formats(
        &self,
        encoder: *mut c_void,
        codec_guid: GUID,
        formats: *mut NV_ENC_BUFFER_FORMAT,
        format_array_size: u32,
        actual_count: *mut u32,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncGetInputFormats
            .expect(NVENC_API_MSG))(
            encoder,
            codec_guid,
            formats,
            format_array_size,
            actual_count,
        )
    }

    unsafe fn get_encode_preset_config_ex(
        &self,
        encoder: *mut c_void,
        codec_guid: GUID,
        preset_guid: GUID,
        tuning_info: NV_ENC_TUNING_INFO,
        preset_config: *mut NV_ENC_PRESET_CONFIG,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncGetEncodePresetConfigEx
            .expect(NVENC_API_MSG))(
            encoder, codec_guid, preset_guid, tuning_info, preset_config
        )
    }

    unsafe fn get_sequence_params(
        &self,
        encoder: *mut c_void,
        params: *mut NV_ENC_SEQUENCE_PARAM_PAYLOAD,
    ) -> NVENCSTATUS {
        (self
            .function_list
            .nvEncGetSequenceParams
            .expect(NVENC_API_MSG))(encoder, params)
    }

    unsafe fn get_last_error_string(&self, encoder: *mut c_void) -> *const i8 {
        (self
            .function_list
            .nvEncGetLastErrorString
            .expect(NVENC_API_MSG))(encoder)
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvenc_api() -> Result<&'static NvencApi, EncodeError> {
    static API: OnceLock<Result<NvencApi, String>> = OnceLock::new();
    match API.get_or_init(load_nvenc_api) {
        Ok(api) => Ok(api),
        Err(error) => Err(EncodeError::InitFailed(error.clone())),
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn load_nvenc_api() -> Result<NvencApi, String> {
    let mut errors = Vec::new();
    for candidate in NVENC_RUNTIME_LIBRARY_CANDIDATES {
        match unsafe { Library::new(candidate) } {
            Ok(library) => match unsafe { load_nvenc_api_from_library(library) } {
                Ok(api) => return Ok(api),
                Err(error) => errors.push(format!("{candidate}: {error}")),
            },
            Err(error) => errors.push(format!("{candidate}: {error}")),
        }
    }

    Err(format!(
        "failed to load the NVIDIA NVENC runtime library (tried {}): {}",
        NVENC_RUNTIME_LIBRARY_CANDIDATES.join(", "),
        errors.join("; ")
    ))
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn load_nvenc_api_from_library(library: Library) -> Result<NvencApi, String> {
    let get_max_supported_version = *library
        .get::<NvEncodeApiGetMaxSupportedVersionFn>(b"NvEncodeAPIGetMaxSupportedVersion\0")
        .map_err(|error| format!("missing NvEncodeAPIGetMaxSupportedVersion export: {error}"))?;
    let create_instance = *library
        .get::<NvEncodeApiCreateInstanceFn>(b"NvEncodeAPICreateInstance\0")
        .map_err(|error| format!("missing NvEncodeAPICreateInstance export: {error}"))?;

    let mut max_supported_version = 0;
    let status = get_max_supported_version(&mut max_supported_version);
    if !nvenc_status_success(status) {
        return Err(format!(
            "NvEncodeAPIGetMaxSupportedVersion failed with {status:?}"
        ));
    }
    assert_nvenc_versions_match(max_supported_version)?;

    let mut function_list = NV_ENCODE_API_FUNCTION_LIST {
        version: NV_ENCODE_API_FUNCTION_LIST_VER,
        ..Default::default()
    };
    let status = create_instance(&mut function_list);
    if !nvenc_status_success(status) {
        return Err(format!("NvEncodeAPICreateInstance failed with {status:?}"));
    }

    std::mem::forget(library);
    Ok(NvencApi { function_list })
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn assert_nvenc_versions_match(max_supported_version: u32) -> Result<(), String> {
    let major_version = max_supported_version >> 4;
    let minor_version = max_supported_version & 0b1111;
    if (major_version, minor_version) < (NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION) {
        return Err(format!(
            "NVENC runtime version {major_version}.{minor_version} is older than the header version {}.{}",
            NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION
        ));
    }
    Ok(())
}

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
        let runtime = NvencRuntimeSession::create(ctx, prepared, self.codec_string())?;
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
            if let Some(config) = session.runtime.decoder_config() {
                return Some(config.clone());
            }
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
    codec_string: String,
    decoder_config: Option<DecoderConfig>,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
unsafe impl Send for NvencRuntimeSession {}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl NvencRuntimeSession {
    fn create(
        ctx: Arc<CudaContext>,
        prepared: &NvencPreparedFrame,
        codec_string: &str,
    ) -> Result<Self, EncodeError> {
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
            codec_string: codec_string.to_owned(),
            decoder_config: None,
        };
        session.open_encode_session()?;
        session.initialize_encoder()?;
        session.decoder_config = session.query_decoder_config()?;
        Ok(session)
    }

    fn descriptor(&self) -> &NvencSessionDescriptor {
        &self.descriptor
    }

    fn decoder_config(&self) -> Option<&DecoderConfig> {
        self.decoder_config.as_ref()
    }

    fn open_encode_session(&mut self) -> Result<(), EncodeError> {
        let api = nvenc_api()?;
        let mut params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
            version: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
            deviceType: NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_CUDA,
            apiVersion: NVENCAPI_VERSION,
            device: self.ctx.cu_ctx().cast::<c_void>(),
            ..Default::default()
        };
        let mut encoder = ptr::null_mut();
        let status = unsafe { api.open_encode_session_ex(&mut params, &mut encoder) };
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
        let api = nvenc_api()?;
        let status = unsafe { api.initialize_encoder(self.encoder, &mut initialize_params) };
        nvenc_init_result(status, self.encoder, "failed to initialize NVENC encoder")?;

        let mut create_bitstream = NV_ENC_CREATE_BITSTREAM_BUFFER {
            version: NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
            bitstreamBuffer: ptr::null_mut(),
            ..Default::default()
        };
        let status = unsafe { api.create_bitstream_buffer(self.encoder, &mut create_bitstream) };
        nvenc_init_result(
            status,
            self.encoder,
            "failed to create NVENC output bitstream",
        )?;
        self.output_bitstream = create_bitstream.bitstreamBuffer;
        Ok(())
    }

    fn query_decoder_config(&self) -> Result<Option<DecoderConfig>, EncodeError> {
        match self.descriptor.codec {
            NvencCodec::Hevc => Ok(Some(self.query_hevc_decoder_config()?)),
            NvencCodec::Av1 => Ok(None),
        }
    }

    fn query_hevc_decoder_config(&self) -> Result<DecoderConfig, EncodeError> {
        let sequence_payload = self.query_sequence_payload()?;
        let description = build_hevc_hvcc_description(&sequence_payload).map_err(|error| {
            EncodeError::InitFailed(format!(
                "failed to build HEVC decoder configuration from NVENC sequence parameters: {error}"
            ))
        })?;
        Ok(DecoderConfig {
            codec: self.codec_string.clone(),
            description: Some(description),
            coded_width: self.descriptor.width,
            coded_height: self.descriptor.height,
        })
    }

    fn query_sequence_payload(&self) -> Result<Vec<u8>, EncodeError> {
        let api = nvenc_api()?;
        let mut payload = vec![0u8; NVENC_SEQUENCE_PAYLOAD_BUFFER_SIZE];
        let mut payload_size = 0u32;
        let mut params = NV_ENC_SEQUENCE_PARAM_PAYLOAD {
            version: NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER,
            inBufferSize: payload.len().min(u32::MAX as usize) as u32,
            spsId: 0,
            ppsId: 0,
            spsppsBuffer: payload.as_mut_ptr().cast::<c_void>(),
            outSPSPPSPayloadSize: &mut payload_size,
            ..Default::default()
        };
        let status = unsafe { api.get_sequence_params(self.encoder, &mut params) };
        nvenc_init_result(
            status,
            self.encoder,
            "failed to query NVENC sequence parameters",
        )?;
        let payload_len = payload_size as usize;
        if payload_len == 0 {
            return Err(EncodeError::InitFailed(
                "NVENC returned an empty sequence-parameter payload".into(),
            ));
        }
        payload.truncate(payload_len.min(payload.len()));
        Ok(payload)
    }

    fn normalize_access_unit(&self, data: Vec<u8>) -> Result<Vec<u8>, EncodeError> {
        match self.descriptor.codec {
            NvencCodec::Hevc => normalize_hevc_access_unit(&data).map_err(|error| {
                EncodeError::EncodeFailed(format!(
                    "failed to normalize NVENC HEVC access unit for browser decode: {error}"
                ))
            }),
            NvencCodec::Av1 => Ok(data),
        }
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
        let api = nvenc_api()?;
        let status = unsafe { api.encode_picture(self.encoder, &mut encode_pic_params) };
        nvenc_encode_result(status, self.encoder, "failed to encode NVENC picture")?;

        let output = self.read_output_bitstream()?;
        if output.data.is_empty() {
            return Err(EncodeError::EncodeFailed(
                "NVENC encode completed but produced an empty output bitstream".into(),
            ));
        }
        let bitstream = self.normalize_access_unit(output.data)?;

        Ok(EncodedFrame {
            data: bitstream,
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
        let api = nvenc_api()?;
        let status = unsafe { api.lock_bitstream(self.encoder, &mut lock_params) };
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
        let unlock_status = unsafe { api.unlock_bitstream(self.encoder, self.output_bitstream) };
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
        let mut eos = NV_ENC_PIC_PARAMS {
            version: NV_ENC_PIC_PARAMS_VER,
            encodePicFlags: NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_EOS as u32,
            ..Default::default()
        };
        let api = nvenc_api()?;
        let status = unsafe { api.encode_picture(self.encoder, &mut eos) };
        nvenc_encode_result(status, self.encoder, "failed to flush NVENC encoder")?;
        Ok(Vec::new())
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl Drop for NvencRuntimeSession {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if let Ok(api) = nvenc_api() {
            if !self.output_bitstream.is_null() {
                let _ =
                    unsafe { api.destroy_bitstream_buffer(self.encoder, self.output_bitstream) };
            }
            if !self.encoder.is_null() {
                let _ = unsafe { api.destroy_encoder(self.encoder) };
            }
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
        let api = nvenc_api()?;
        let mut register_resource = NV_ENC_REGISTER_RESOURCE {
            version: NV_ENC_REGISTER_RESOURCE_VER,
            resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
            width: descriptor.width,
            height: descriptor.height,
            pitch,
            resourceToRegister: imported.device_ptr as *mut c_void,
            registeredResource: ptr::null_mut(),
            bufferFormat: descriptor.buffer_format(),
            ..Default::default()
        };
        let status = unsafe { api.register_resource(encoder, &mut register_resource) };
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
        let status = unsafe { api.map_input_resource(encoder, &mut map_input) };
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
        if let Ok(api) = nvenc_api() {
            if !self.mapped_resource.is_null() {
                let _ = unsafe { api.unmap_input_resource(self.encoder, self.mapped_resource) };
            }
            if !self.registered_resource.is_null() {
                let _ = unsafe { api.unregister_resource(self.encoder, self.registered_resource) };
            }
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
    let api = nvenc_api()?;
    let mut supported_count = 0;
    let status = unsafe { api.get_encode_guid_count(encoder, &mut supported_count) };
    nvenc_init_result(status, encoder, "failed to query NVENC codec count")?;

    let mut actual_count = 0;
    let mut encode_guids = vec![GUID::default(); supported_count as usize];
    let status = unsafe {
        api.get_encode_guids(
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
    let api = nvenc_api()?;
    let mut format_count = 0;
    let status = unsafe { api.get_input_format_count(encoder, codec_guid, &mut format_count) };
    nvenc_init_result(status, encoder, "failed to query NVENC input-format count")?;

    let mut actual_count = 0;
    let mut formats =
        vec![NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED; format_count as usize];
    let status = unsafe {
        api.get_input_formats(
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
    let api = nvenc_api()?;
    let mut preset_config = NV_ENC_PRESET_CONFIG {
        version: NV_ENC_PRESET_CONFIG_VER,
        presetCfg: NV_ENC_CONFIG {
            version: NV_ENC_CONFIG_VER,
            ..Default::default()
        },
        ..Default::default()
    };
    let status = unsafe {
        api.get_encode_preset_config_ex(
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
#[derive(Debug, Clone, PartialEq, Eq)]
struct HevcParameterSets {
    vps: Vec<u8>,
    sps: Vec<u8>,
    pps: Vec<u8>,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HevcSpsMetadata {
    general_profile_space: u8,
    general_tier_flag: bool,
    general_profile_idc: u8,
    general_profile_compatibility_flags: u32,
    general_constraint_indicator_flags: u64,
    general_level_idc: u8,
    chroma_format_idc: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    num_temporal_layers: u8,
    temporal_id_nested: bool,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
struct BitReader<'a> {
    data: &'a [u8],
    bit_offset: usize,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_offset: 0,
        }
    }

    fn read_bit(&mut self) -> Result<u8, String> {
        self.read_bits(1).map(|value| value as u8)
    }

    fn read_bits(&mut self, bit_count: u8) -> Result<u64, String> {
        let mut value = 0u64;
        for _ in 0..bit_count {
            if self.bit_offset >= self.data.len().saturating_mul(8) {
                return Err("unexpected end of HEVC RBSP".into());
            }
            let byte = self.data[self.bit_offset / 8];
            let shift = 7 - (self.bit_offset % 8);
            value = (value << 1) | u64::from((byte >> shift) & 1);
            self.bit_offset += 1;
        }
        Ok(value)
    }

    fn read_ue(&mut self) -> Result<u32, String> {
        let mut leading_zero_bits = 0u32;
        while self.read_bit()? == 0 {
            leading_zero_bits = leading_zero_bits.saturating_add(1);
            if leading_zero_bits > 31 {
                return Err("HEVC Exp-Golomb value exceeds supported range".into());
            }
        }
        let suffix = if leading_zero_bits == 0 {
            0
        } else {
            self.read_bits(leading_zero_bits as u8)? as u32
        };
        Ok((1u32 << leading_zero_bits) - 1 + suffix)
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn normalize_hevc_access_unit(data: &[u8]) -> Result<Vec<u8>, String> {
    let nal_units = extract_annex_b_nalus(data);
    if nal_units.is_empty() {
        return Ok(data.to_vec());
    }

    let mut output = Vec::with_capacity(data.len());
    for nal_unit in nal_units {
        let length = u32::try_from(nal_unit.len())
            .map_err(|_| "HEVC NAL unit exceeded 32-bit length field".to_string())?;
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(nal_unit);
    }
    Ok(output)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn build_hevc_hvcc_description(sequence_payload: &[u8]) -> Result<Vec<u8>, String> {
    let parameter_sets = extract_hevc_parameter_sets(sequence_payload)?;
    let metadata = parse_hevc_sps_metadata(&parameter_sets.sps)?;

    let mut description = Vec::with_capacity(
        23 + parameter_sets.vps.len() + parameter_sets.sps.len() + parameter_sets.pps.len() + 18,
    );
    description.push(1);
    description.push(
        ((metadata.general_profile_space & 0x03) << 6)
            | (u8::from(metadata.general_tier_flag) << 5)
            | (metadata.general_profile_idc & 0x1f),
    );
    description.extend_from_slice(&metadata.general_profile_compatibility_flags.to_be_bytes());
    description.extend_from_slice(&metadata.general_constraint_indicator_flags.to_be_bytes()[2..]);
    description.push(metadata.general_level_idc);
    description.extend_from_slice(&0xF000u16.to_be_bytes());
    description.push(0xFC);
    description.push(0xFC | (metadata.chroma_format_idc & 0x03));
    description.push(0xF8 | (metadata.bit_depth_luma_minus8 & 0x07));
    description.push(0xF8 | (metadata.bit_depth_chroma_minus8 & 0x07));
    description.extend_from_slice(&0u16.to_be_bytes());
    description.push(
        ((metadata.num_temporal_layers.max(1).min(7) & 0x07) << 3)
            | (u8::from(metadata.temporal_id_nested) << 2)
            | (HEVC_HVCC_LENGTH_SIZE_MINUS_ONE & 0x03),
    );
    description.push(3);
    append_hvcc_array(&mut description, HEVC_NAL_TYPE_VPS, &parameter_sets.vps);
    append_hvcc_array(&mut description, HEVC_NAL_TYPE_SPS, &parameter_sets.sps);
    append_hvcc_array(&mut description, HEVC_NAL_TYPE_PPS, &parameter_sets.pps);
    Ok(description)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn append_hvcc_array(description: &mut Vec<u8>, nal_type: u8, nal_unit: &[u8]) {
    description.push(0x80 | (nal_type & 0x3f));
    description.extend_from_slice(&1u16.to_be_bytes());
    description.extend_from_slice(&(nal_unit.len().min(u16::MAX as usize) as u16).to_be_bytes());
    description.extend_from_slice(nal_unit);
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn extract_hevc_parameter_sets(sequence_payload: &[u8]) -> Result<HevcParameterSets, String> {
    let mut vps = None;
    let mut sps = None;
    let mut pps = None;

    for nal_unit in extract_annex_b_nalus(sequence_payload) {
        let Some(nal_type) = hevc_nal_type(nal_unit) else {
            continue;
        };
        match nal_type {
            HEVC_NAL_TYPE_VPS if vps.is_none() => vps = Some(nal_unit.to_vec()),
            HEVC_NAL_TYPE_SPS if sps.is_none() => sps = Some(nal_unit.to_vec()),
            HEVC_NAL_TYPE_PPS if pps.is_none() => pps = Some(nal_unit.to_vec()),
            _ => {}
        }
    }

    Ok(HevcParameterSets {
        vps: vps.ok_or_else(|| {
            "NVENC sequence payload did not contain a HEVC VPS NAL unit".to_string()
        })?,
        sps: sps.ok_or_else(|| {
            "NVENC sequence payload did not contain a HEVC SPS NAL unit".to_string()
        })?,
        pps: pps.ok_or_else(|| {
            "NVENC sequence payload did not contain a HEVC PPS NAL unit".to_string()
        })?,
    })
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn hevc_nal_type(nal_unit: &[u8]) -> Option<u8> {
    nal_unit.first().map(|byte| (byte >> 1) & 0x3f)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn extract_annex_b_nalus(data: &[u8]) -> Vec<&[u8]> {
    let mut nal_units = Vec::new();
    let mut search_from = 0usize;

    while let Some((start_code_offset, start_code_len)) = find_annex_b_start_code(data, search_from)
    {
        let nal_start = start_code_offset + start_code_len;
        let next_start = find_annex_b_start_code(data, nal_start)
            .map(|(offset, _)| offset)
            .unwrap_or(data.len());
        let mut nal_end = next_start;
        while nal_end > nal_start && data[nal_end - 1] == 0 {
            nal_end -= 1;
        }
        if nal_end > nal_start {
            nal_units.push(&data[nal_start..nal_end]);
        }
        search_from = next_start;
    }

    nal_units
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn find_annex_b_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut index = from;
    while index + 3 <= data.len() {
        if data[index] == 0 && data[index + 1] == 0 {
            if data.get(index + 2) == Some(&1) {
                return Some((index, 3));
            }
            if data.get(index + 2) == Some(&0) && data.get(index + 3) == Some(&1) {
                return Some((index, 4));
            }
        }
        index += 1;
    }
    None
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn parse_hevc_sps_metadata(sps_nal_unit: &[u8]) -> Result<HevcSpsMetadata, String> {
    if sps_nal_unit.len() < 3 {
        return Err("HEVC SPS NAL unit was too short".into());
    }
    let rbsp = remove_emulation_prevention_bytes(&sps_nal_unit[2..]);
    let mut bits = BitReader::new(&rbsp);

    let _sps_video_parameter_set_id = bits.read_bits(4)?;
    let sps_max_sub_layers_minus1 = bits.read_bits(3)? as u8;
    let temporal_id_nested = bits.read_bit()? != 0;
    let general_profile_space = bits.read_bits(2)? as u8;
    let general_tier_flag = bits.read_bit()? != 0;
    let general_profile_idc = bits.read_bits(5)? as u8;
    let general_profile_compatibility_flags = bits.read_bits(32)? as u32;
    let general_constraint_indicator_flags = bits.read_bits(48)?;
    let general_level_idc = bits.read_bits(8)? as u8;

    let mut sub_layer_profile_present_flags =
        Vec::with_capacity(sps_max_sub_layers_minus1 as usize);
    let mut sub_layer_level_present_flags = Vec::with_capacity(sps_max_sub_layers_minus1 as usize);
    for _ in 0..sps_max_sub_layers_minus1 {
        sub_layer_profile_present_flags.push(bits.read_bit()? != 0);
        sub_layer_level_present_flags.push(bits.read_bit()? != 0);
    }
    if sps_max_sub_layers_minus1 > 0 {
        for _ in sps_max_sub_layers_minus1..8 {
            let _reserved_zero_2bits = bits.read_bits(2)?;
        }
    }
    for (profile_present, level_present) in sub_layer_profile_present_flags
        .into_iter()
        .zip(sub_layer_level_present_flags.into_iter())
    {
        if profile_present {
            let _sub_layer_profile_space = bits.read_bits(2)?;
            let _sub_layer_tier_flag = bits.read_bit()?;
            let _sub_layer_profile_idc = bits.read_bits(5)?;
            let _sub_layer_profile_compatibility_flags = bits.read_bits(32)?;
            let _sub_layer_constraint_indicator_flags = bits.read_bits(48)?;
        }
        if level_present {
            let _sub_layer_level_idc = bits.read_bits(8)?;
        }
    }

    let _sps_seq_parameter_set_id = bits.read_ue()?;
    let chroma_format_idc = bits.read_ue()?.min(3) as u8;
    if chroma_format_idc == 3 {
        let _separate_colour_plane_flag = bits.read_bit()?;
    }
    let _pic_width_in_luma_samples = bits.read_ue()?;
    let _pic_height_in_luma_samples = bits.read_ue()?;
    let conformance_window_flag = bits.read_bit()? != 0;
    if conformance_window_flag {
        let _left = bits.read_ue()?;
        let _right = bits.read_ue()?;
        let _top = bits.read_ue()?;
        let _bottom = bits.read_ue()?;
    }
    let bit_depth_luma_minus8 = bits.read_ue()?.min(7) as u8;
    let bit_depth_chroma_minus8 = bits.read_ue()?.min(7) as u8;

    Ok(HevcSpsMetadata {
        general_profile_space,
        general_tier_flag,
        general_profile_idc,
        general_profile_compatibility_flags,
        general_constraint_indicator_flags,
        general_level_idc,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        num_temporal_layers: sps_max_sub_layers_minus1.saturating_add(1),
        temporal_id_nested,
    })
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn remove_emulation_prevention_bytes(data: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(data.len());
    let mut consecutive_zeros = 0u8;

    for &byte in data {
        if consecutive_zeros >= 2 && byte == 0x03 {
            consecutive_zeros = 0;
            continue;
        }
        rbsp.push(byte);
        if byte == 0 {
            consecutive_zeros = consecutive_zeros.saturating_add(1);
        } else {
            consecutive_zeros = 0;
        }
    }

    rbsp
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvenc_init_result(
    status: NVENCSTATUS,
    encoder: *mut c_void,
    action: &str,
) -> Result<(), EncodeError> {
    if nvenc_status_success(status) {
        Ok(())
    } else {
        Err(EncodeError::InitFailed(nvenc_status_message(
            status, encoder, action,
        )))
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvenc_encode_result(
    status: NVENCSTATUS,
    encoder: *mut c_void,
    action: &str,
) -> Result<(), EncodeError> {
    if nvenc_status_success(status) {
        Ok(())
    } else {
        Err(EncodeError::EncodeFailed(nvenc_status_message(
            status, encoder, action,
        )))
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvenc_status_message(status: NVENCSTATUS, encoder: *mut c_void, action: &str) -> String {
    if nvenc_status_success(status) {
        return action.to_owned();
    }
    let base = format!("{status:?}");
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
    let api = nvenc_api().ok()?;
    let ptr = unsafe { api.get_last_error_string(encoder) };
    if ptr.is_null() {
        return None;
    }
    let text = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .trim()
        .to_owned();
    if text.is_empty() { None } else { Some(text) }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvenc_status_success(status: NVENCSTATUS) -> bool {
    matches!(status, NVENCSTATUS::NV_ENC_SUCCESS)
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    use std::fs::File;

    #[cfg(target_os = "linux")]
    use ustreamer_capture::{VulkanExternalImage, VulkanExternalMemoryHandle};
    #[cfg(target_os = "linux")]
    use ustreamer_proto::quality::EncodeMode;
    use ustreamer_proto::quality::EncodeParams;

    use super::{
        HEVC_HVCC_LENGTH_SIZE_MINUS_ONE, HEVC_NAL_TYPE_PPS, HEVC_NAL_TYPE_SPS, HEVC_NAL_TYPE_VPS,
        NvencEncoder, NvencInputFormat, build_hevc_hvcc_description,
        extract_hevc_parameter_sets, normalize_hevc_access_unit,
    };
    #[cfg(target_os = "linux")]
    use super::{NvencExternalMemoryHandleDescriptor, NvencExternalSyncDescriptor};
    use ustreamer_capture::CapturedFrame;

    fn sample_hevc_sequence_payload() -> Vec<u8> {
        [
            &[0x00, 0x00, 0x00, 0x01][..],
            &[
                0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0xb0, 0x00,
                0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x5d, 0xac, 0x59,
            ],
            &[0x00, 0x00, 0x00, 0x01][..],
            &[
                0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0xb0, 0x00, 0x00, 0x03, 0x00,
                0x00, 0x03, 0x00, 0x5d, 0xa0, 0x02, 0x80, 0x80, 0x2d, 0x16, 0x59, 0x59, 0xa4, 0x93,
                0x2b, 0xc0, 0x5a, 0x02, 0x02, 0x02, 0x80,
            ],
            &[0x00, 0x00, 0x00, 0x01][..],
            &[0x44, 0x01, 0xc1, 0x73, 0xd1, 0x89],
        ]
        .concat()
    }

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

    #[test]
    fn normalizes_annex_b_hevc_access_units_to_length_prefixed() {
        let normalized = normalize_hevc_access_unit(&[
            0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0xaa, 0xbb, 0x00, 0x00, 0x01, 0x02, 0x01, 0xcc,
        ])
        .unwrap();

        assert_eq!(
            normalized,
            vec![
                0x00, 0x00, 0x00, 0x04, 0x26, 0x01, 0xaa, 0xbb, 0x00, 0x00, 0x00, 0x03, 0x02, 0x01,
                0xcc,
            ]
        );
    }

    #[test]
    fn extracts_hevc_parameter_sets_from_sequence_payload() {
        let parameter_sets = extract_hevc_parameter_sets(&sample_hevc_sequence_payload()).unwrap();

        assert_eq!((parameter_sets.vps[0] >> 1) & 0x3f, HEVC_NAL_TYPE_VPS);
        assert_eq!((parameter_sets.sps[0] >> 1) & 0x3f, HEVC_NAL_TYPE_SPS);
        assert_eq!((parameter_sets.pps[0] >> 1) & 0x3f, HEVC_NAL_TYPE_PPS);
    }

    #[test]
    fn builds_hvcc_description_from_hevc_sequence_payload() {
        let description = build_hevc_hvcc_description(&sample_hevc_sequence_payload()).unwrap();

        assert_eq!(description[0], 1);
        assert_eq!(description[21] & 0x03, HEVC_HVCC_LENGTH_SIZE_MINUS_ONE);
        assert_eq!(description[22], 3);
        assert_eq!(description[23], 0x80 | HEVC_NAL_TYPE_VPS);
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
