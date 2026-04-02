//! Direct NVENC encoder groundwork for Vulkan external-memory frames.
//!
//! This module does not yet perform CUDA import or invoke the NVIDIA Video Codec
//! SDK. Instead, it validates exported Vulkan frames and translates them into the
//! input/resource descriptor shape that the next FFI slice will consume. On
//! Linux and Windows, it can now also import exported Vulkan external-memory
//! handles into CUDA via `cudarc`, but it still stops short of NVENC session
//! creation and bitstream output.

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(target_os = "windows")]
use std::os::windows::io::AsRawHandle;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::{fs::File, mem::ManuallyDrop, sync::Arc};

#[cfg(any(target_os = "linux", target_os = "windows"))]
use cudarc::driver::{
    result::{self, DriverError as CudaDriverError},
    safe::CudaContext,
    sys,
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
            let imported = self.import_to_cuda(frame, params)?;
            return Err(EncodeError::InitFailed(format!(
                "direct NVENC bitstream output is not wired yet; CUDA import succeeded for {}x{} frame (device_ptr=0x{:x}, pitch={} bytes) but NVENC session creation and resource registration remain pending",
                imported.width, imported.height, imported.device_ptr, imported.pitch_bytes
            )));
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
