//! Vulkan external-memory capture for future CUDA/NVENC zero-copy import.
//!
//! This path allocates an exportable Vulkan image, wraps it back into `wgpu`,
//! uses a normal `copy_texture_to_texture` blit, and then exports an OS handle
//! for the bound device memory. That keeps the integration on public
//! `wgpu`/`wgpu-hal` APIs and avoids patching either crate.

use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;
#[cfg(feature = "vulkan-external-test-utils")]
use ash::vk::Handle;

use crate::{CaptureError, CapturedFrame, FrameCapture};

#[cfg(target_os = "linux")]
use std::os::fd::{AsFd, BorrowedFd, FromRawFd, OwnedFd};
#[cfg(target_os = "windows")]
use std::os::windows::io::{AsHandle, BorrowedHandle, FromRawHandle, OwnedHandle, RawHandle};

const EXPORT_TEXTURE_LABEL: &str = "ustreamer-vulkan-external";

static NEXT_RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(target_os = "windows")]
const WIN32_SHARED_HANDLE_ACCESS: u32 = 0x8000_0000 | 0x4000_0000;

/// Exported OS handle for a Vulkan external-memory allocation.
#[derive(Debug)]
pub enum VulkanExternalMemoryHandle {
    #[cfg(target_os = "linux")]
    OpaqueFd(OwnedFd),
    #[cfg(target_os = "windows")]
    OpaqueWin32Handle(OwnedHandle),
}

/// Exported OS handle for a Vulkan external synchronization primitive.
#[derive(Debug)]
pub enum VulkanExternalSyncHandle {
    #[cfg(target_os = "linux")]
    OpaqueFd(OwnedFd),
    #[cfg(target_os = "windows")]
    OpaqueWin32Handle(OwnedHandle),
}

/// Synchronization contract for an exported Vulkan image.
#[derive(Debug)]
pub enum VulkanExternalSync {
    /// The capture path waited on the host before handing the frame to encode.
    HostSynchronized,
    /// A future GPU-only handoff path via exported semaphore handle.
    ExternalSemaphore {
        handle: VulkanExternalSyncHandle,
        value: u64,
    },
}

/// Capture-side synchronization strategy for exported Vulkan frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VulkanCaptureSyncMode {
    /// Wait on the host before returning the frame to encode.
    #[default]
    HostSynchronized,
    /// After the host wait, signal an exportable timeline semaphore and hand its
    /// external handle to the encode path.
    ExportedTimelineSemaphore,
}

/// Vulkan image + exported external-memory handle suitable for a future CUDA/NVENC import step.
#[derive(Debug)]
pub struct VulkanExternalImage {
    resource_id: u64,
    image: vk::Image,
    device_memory: vk::DeviceMemory,
    allocation_size: u64,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    handle: VulkanExternalMemoryHandle,
    sync: VulkanExternalSync,
    _backing_texture: Option<wgpu::Texture>,
}

impl VulkanExternalImage {
    pub fn resource_id(&self) -> u64 {
        self.resource_id
    }

    pub fn raw_image(&self) -> vk::Image {
        self.image
    }

    pub fn raw_device_memory(&self) -> vk::DeviceMemory {
        self.device_memory
    }

    pub fn allocation_size(&self) -> u64 {
        self.allocation_size
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn memory_handle(&self) -> &VulkanExternalMemoryHandle {
        &self.handle
    }

    pub fn sync(&self) -> &VulkanExternalSync {
        &self.sync
    }

    #[cfg(target_os = "linux")]
    pub fn opaque_fd(&self) -> BorrowedFd<'_> {
        match &self.handle {
            VulkanExternalMemoryHandle::OpaqueFd(fd) => fd.as_fd(),
        }
    }

    #[cfg(target_os = "linux")]
    pub fn try_clone_opaque_fd(&self) -> std::io::Result<OwnedFd> {
        self.opaque_fd().try_clone_to_owned()
    }

    #[cfg(target_os = "windows")]
    pub fn opaque_win32_handle(&self) -> BorrowedHandle<'_> {
        match &self.handle {
            VulkanExternalMemoryHandle::OpaqueWin32Handle(handle) => handle.as_handle(),
        }
    }

    #[cfg(target_os = "windows")]
    pub fn try_clone_opaque_win32_handle(&self) -> std::io::Result<OwnedHandle> {
        self.opaque_win32_handle().try_clone_to_owned()
    }
}

impl VulkanExternalSync {
    pub fn is_host_synchronized(&self) -> bool {
        matches!(self, Self::HostSynchronized)
    }
}

impl VulkanExternalSyncHandle {
    #[cfg(target_os = "linux")]
    pub fn opaque_fd(&self) -> BorrowedFd<'_> {
        match self {
            VulkanExternalSyncHandle::OpaqueFd(fd) => fd.as_fd(),
        }
    }

    #[cfg(target_os = "linux")]
    pub fn try_clone_opaque_fd(&self) -> std::io::Result<OwnedFd> {
        self.opaque_fd().try_clone_to_owned()
    }

    #[cfg(target_os = "windows")]
    pub fn opaque_win32_handle(&self) -> BorrowedHandle<'_> {
        match self {
            VulkanExternalSyncHandle::OpaqueWin32Handle(handle) => handle.as_handle(),
        }
    }

    #[cfg(target_os = "windows")]
    pub fn try_clone_opaque_win32_handle(&self) -> std::io::Result<OwnedHandle> {
        self.opaque_win32_handle().try_clone_to_owned()
    }
}

#[derive(Debug)]
struct CachedExportTexture {
    resource_id: u64,
    image: vk::Image,
    device_memory: vk::DeviceMemory,
    sync_semaphore: Option<vk::Semaphore>,
    next_sync_value: AtomicU64,
    allocation_size: u64,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    #[cfg(target_os = "windows")]
    exported_memory_handle: OwnedHandle,
    #[cfg(target_os = "windows")]
    exported_sync_handle: Option<OwnedHandle>,
    texture: wgpu::Texture,
}

impl CachedExportTexture {
    fn matches_source(&self, texture: &wgpu::Texture) -> bool {
        let size = texture.size();
        self.width == size.width
            && self.height == size.height
            && self.format == texture.format()
            && texture.dimension() == wgpu::TextureDimension::D2
            && texture.depth_or_array_layers() == 1
            && texture.mip_level_count() == 1
            && texture.sample_count() == 1
    }

    fn into_frame(
        &self,
        handle: VulkanExternalMemoryHandle,
        sync: VulkanExternalSync,
    ) -> VulkanExternalImage {
        VulkanExternalImage {
            resource_id: self.resource_id,
            image: self.image,
            device_memory: self.device_memory,
            allocation_size: self.allocation_size,
            width: self.width,
            height: self.height,
            format: self.format,
            handle,
            sync,
            _backing_texture: Some(self.texture.clone()),
        }
    }

    fn next_sync_value(&self) -> u64 {
        self.next_sync_value.fetch_add(1, Ordering::Relaxed) + 1
    }

    #[cfg(target_os = "windows")]
    fn clone_memory_handle(&self) -> Result<VulkanExternalMemoryHandle, CaptureError> {
        self.exported_memory_handle
            .try_clone()
            .map(VulkanExternalMemoryHandle::OpaqueWin32Handle)
            .map_err(|error| {
                CaptureError::VulkanInteropFailed(format!(
                    "failed to clone cached exported Vulkan Win32 memory handle: {error}"
                ))
            })
    }

    #[cfg(target_os = "windows")]
    fn clone_sync_handle(&self) -> Result<VulkanExternalSyncHandle, CaptureError> {
        self.exported_sync_handle
            .as_ref()
            .ok_or_else(|| {
                CaptureError::VulkanInteropFailed(
                    "cached export texture is missing its cached exported Win32 semaphore handle"
                        .into(),
                )
            })?
            .try_clone()
            .map(VulkanExternalSyncHandle::OpaqueWin32Handle)
            .map_err(|error| {
                CaptureError::VulkanInteropFailed(format!(
                    "failed to clone cached exported Vulkan Win32 semaphore handle: {error}"
                ))
            })
    }
}

/// Zero-copy Vulkan capture entry point for future NVENC integration.
#[derive(Debug, Default)]
pub struct VulkanExternalCapture {
    cached: Option<CachedExportTexture>,
    sync_mode: VulkanCaptureSyncMode,
}

impl VulkanExternalCapture {
    pub fn new() -> Self {
        Self::with_sync_mode(VulkanCaptureSyncMode::HostSynchronized)
    }

    pub fn with_sync_mode(sync_mode: VulkanCaptureSyncMode) -> Self {
        Self {
            cached: None,
            sync_mode,
        }
    }

    pub fn sync_mode(&self) -> VulkanCaptureSyncMode {
        self.sync_mode
    }

    /// Choose the best currently-usable sync mode for this Vulkan device.
    ///
    /// This preserves `HostSynchronized` as the safe default while upgrading to
    /// exported timeline semaphores only when the required Vulkan export
    /// extensions are already enabled on the active `wgpu` device.
    pub fn best_available_sync_mode(
        device: &wgpu::Device,
    ) -> Result<VulkanCaptureSyncMode, CaptureError> {
        let device_hal =
            unsafe {
                device.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu device is not using Vulkan"),
                )?
            };
        best_sync_mode_for_enabled_extensions(device_hal.enabled_device_extensions())
    }

    fn ensure_cached_export_texture<'a>(
        &'a mut self,
        instance: &wgpu::Instance,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
    ) -> Result<&'a CachedExportTexture, CaptureError> {
        let needs_recreate = self
            .cached
            .as_ref()
            .is_none_or(|cached| !cached.matches_source(texture));

        if needs_recreate {
            self.cached = Some(create_cached_export_texture(
                instance,
                device,
                texture,
                self.sync_mode,
            )?);
        }

        self.cached.as_ref().ok_or_else(|| {
            CaptureError::VulkanInteropFailed(
                "cached export texture was missing after recreation".into(),
            )
        })
    }
}

impl VulkanExternalImage {
    #[cfg(feature = "vulkan-external-test-utils")]
    #[doc(hidden)]
    pub unsafe fn from_raw_export_with_sync_for_test(
        resource_id: u64,
        raw_image: u64,
        raw_device_memory: u64,
        allocation_size: u64,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        handle: VulkanExternalMemoryHandle,
        sync: VulkanExternalSync,
    ) -> Self {
        Self {
            resource_id,
            image: vk::Image::from_raw(raw_image),
            device_memory: vk::DeviceMemory::from_raw(raw_device_memory),
            allocation_size,
            width,
            height,
            format,
            handle,
            sync,
            _backing_texture: None,
        }
    }

    #[cfg(feature = "vulkan-external-test-utils")]
    #[doc(hidden)]
    pub unsafe fn from_raw_export_for_test(
        resource_id: u64,
        raw_image: u64,
        raw_device_memory: u64,
        allocation_size: u64,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        handle: VulkanExternalMemoryHandle,
    ) -> Self {
        Self::from_raw_export_with_sync_for_test(
            resource_id,
            raw_image,
            raw_device_memory,
            allocation_size,
            width,
            height,
            format,
            handle,
            VulkanExternalSync::HostSynchronized,
        )
    }
}

impl FrameCapture for VulkanExternalCapture {
    fn capture(
        &mut self,
        instance: &wgpu::Instance,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<CapturedFrame, CaptureError> {
        validate_source_texture(texture)?;
        let sync_mode = self.sync_mode;
        let cached = self.ensure_cached_export_texture(instance, device, texture)?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ustreamer-vulkan-external-copy"),
        });
        encoder.copy_texture_to_texture(
            texture.as_image_copy(),
            cached.texture.as_image_copy(),
            texture.size(),
        );
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        #[cfg(target_os = "linux")]
        let handle = export_memory_handle(instance, device, cached.device_memory)?;
        #[cfg(target_os = "windows")]
        let handle = cached.clone_memory_handle()?;
        let sync = match sync_mode {
            VulkanCaptureSyncMode::HostSynchronized => VulkanExternalSync::HostSynchronized,
            VulkanCaptureSyncMode::ExportedTimelineSemaphore => {
                let semaphore = cached.sync_semaphore.ok_or_else(|| {
                    CaptureError::VulkanInteropFailed(
                        "cached export texture is missing its exportable timeline semaphore".into(),
                    )
                })?;
                let value = cached.next_sync_value();
                signal_exported_timeline_semaphore(device, semaphore, value)?;
                #[cfg(target_os = "linux")]
                let handle = export_sync_handle(instance, device, semaphore)?;
                #[cfg(target_os = "windows")]
                let handle = cached.clone_sync_handle()?;
                VulkanExternalSync::ExternalSemaphore { handle, value }
            }
        };
        Ok(CapturedFrame::VulkanExternalImage(
            cached.into_frame(handle, sync),
        ))
    }
}

fn validate_source_texture(texture: &wgpu::Texture) -> Result<(), CaptureError> {
    if !texture.usage().contains(wgpu::TextureUsages::COPY_SRC) {
        return Err(CaptureError::InvalidTexture(
            "source texture must include COPY_SRC usage".into(),
        ));
    }
    if texture.dimension() != wgpu::TextureDimension::D2 {
        return Err(CaptureError::InvalidTexture(format!(
            "only 2D textures are currently supported, got {:?}",
            texture.dimension()
        )));
    }
    if texture.depth_or_array_layers() != 1 {
        return Err(CaptureError::InvalidTexture(format!(
            "array and 3D textures are not supported yet (depth_or_array_layers={})",
            texture.depth_or_array_layers()
        )));
    }
    if texture.mip_level_count() != 1 {
        return Err(CaptureError::InvalidTexture(format!(
            "mipmapped textures are not supported yet (mip_level_count={})",
            texture.mip_level_count()
        )));
    }
    if texture.sample_count() != 1 {
        return Err(CaptureError::InvalidTexture(format!(
            "multisampled textures are not supported yet (sample_count={})",
            texture.sample_count()
        )));
    }

    let _ = map_texture_format(texture.format())?;
    Ok(())
}

fn create_cached_export_texture(
    instance: &wgpu::Instance,
    device: &wgpu::Device,
    source_texture: &wgpu::Texture,
    sync_mode: VulkanCaptureSyncMode,
) -> Result<CachedExportTexture, CaptureError> {
    let handle_type = export_handle_type()?;
    let size = source_texture.size();
    let format = source_texture.format();
    let hal_desc = hal_texture_descriptor(source_texture);
    let wgpu_desc = wgpu::TextureDescriptor {
        label: Some(EXPORT_TEXTURE_LABEL),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };

    #[cfg(target_os = "linux")]
    let (image, memory, sync_semaphore, allocation_size, hal_texture) = {
        let instance_hal =
            unsafe {
                instance.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu instance is not using Vulkan"),
                )?
            };
        let device_hal =
            unsafe {
                device.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu device is not using Vulkan"),
                )?
            };
        ensure_external_memory_extension(device_hal.enabled_device_extensions())?;
        if sync_mode == VulkanCaptureSyncMode::ExportedTimelineSemaphore {
            ensure_external_semaphore_extension(device_hal.enabled_device_extensions())?;
        }

        let raw_instance = instance_hal.shared_instance().raw_instance();
        let raw_device = device_hal.raw_device();

        let mut external_memory_image_info =
            vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(map_texture_format(format)?)
            .extent(vk::Extent3D {
                width: size.width,
                height: size.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory_image_info);

        let image = unsafe { raw_device.create_image(&image_info, None) }
            .map_err(|error| map_vk_error("vkCreateImage", error))?;
        let requirements = unsafe { raw_device.get_image_memory_requirements(image) };
        let memory_type_index = match find_device_local_memory_type(
            raw_instance,
            device_hal.raw_physical_device(),
            requirements.memory_type_bits,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe {
                    raw_device.destroy_image(image, None);
                }
                return Err(error);
            }
        };

        let mut dedicated_allocate_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut export_memory_info =
            vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
        #[cfg(target_os = "windows")]
        let mut export_memory_win32_info =
            vk::ExportMemoryWin32HandleInfoKHR::default().dw_access(WIN32_SHARED_HANDLE_ACCESS);
        #[cfg(target_os = "linux")]
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut dedicated_allocate_info)
            .push_next(&mut export_memory_info);
        #[cfg(target_os = "windows")]
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut dedicated_allocate_info)
            .push_next(&mut export_memory_info)
            .push_next(&mut export_memory_win32_info);

        let memory = match unsafe { raw_device.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe {
                    raw_device.destroy_image(image, None);
                }
                return Err(map_vk_error("vkAllocateMemory", error));
            }
        };

        if let Err(error) = unsafe { raw_device.bind_image_memory(image, memory, 0) } {
            unsafe {
                raw_device.free_memory(memory, None);
                raw_device.destroy_image(image, None);
            }
            return Err(map_vk_error("vkBindImageMemory", error));
        }

        let sync_semaphore =
            create_exportable_timeline_semaphore(raw_instance, raw_device, sync_mode)?;
        #[cfg(target_os = "windows")]
        let exported_memory_handle = match export_memory_handle(instance, device, memory)? {
            VulkanExternalMemoryHandle::OpaqueWin32Handle(handle) => handle,
        };
        #[cfg(target_os = "windows")]
        let exported_sync_handle = match sync_semaphore {
            Some(semaphore) => Some(match export_sync_handle(instance, device, semaphore)? {
                VulkanExternalSyncHandle::OpaqueWin32Handle(handle) => handle,
            }),
            None => None,
        };

        let drop_device = raw_device.clone();
        let drop_sync_semaphore = sync_semaphore;
        let drop_callback: wgpu::hal::DropCallback = Box::new(move || unsafe {
            if let Some(semaphore) = drop_sync_semaphore {
                drop_device.destroy_semaphore(semaphore, None);
            }
            drop_device.destroy_image(image, None);
            drop_device.free_memory(memory, None);
        });
        let hal_texture =
            unsafe { device_hal.texture_from_raw(image, &hal_desc, Some(drop_callback)) };

        (
            image,
            memory,
            sync_semaphore,
            requirements.size,
            hal_texture,
        )
    };

    #[cfg(target_os = "windows")]
    let (
        image,
        memory,
        sync_semaphore,
        allocation_size,
        exported_memory_handle,
        exported_sync_handle,
        hal_texture,
    ) = {
        let instance_hal =
            unsafe {
                instance.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu instance is not using Vulkan"),
                )?
            };
        let device_hal =
            unsafe {
                device.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu device is not using Vulkan"),
                )?
            };
        ensure_external_memory_extension(device_hal.enabled_device_extensions())?;
        if sync_mode == VulkanCaptureSyncMode::ExportedTimelineSemaphore {
            ensure_external_semaphore_extension(device_hal.enabled_device_extensions())?;
        }

        let raw_instance = instance_hal.shared_instance().raw_instance();
        let raw_device = device_hal.raw_device();

        let mut external_memory_image_info =
            vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(map_texture_format(format)?)
            .extent(vk::Extent3D {
                width: size.width,
                height: size.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory_image_info);

        let image = unsafe { raw_device.create_image(&image_info, None) }
            .map_err(|error| map_vk_error("vkCreateImage", error))?;
        let requirements = unsafe { raw_device.get_image_memory_requirements(image) };
        let memory_type_index = match find_device_local_memory_type(
            raw_instance,
            device_hal.raw_physical_device(),
            requirements.memory_type_bits,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe {
                    raw_device.destroy_image(image, None);
                }
                return Err(error);
            }
        };

        let mut dedicated_allocate_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut export_memory_info =
            vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
        let mut export_memory_win32_info =
            vk::ExportMemoryWin32HandleInfoKHR::default().dw_access(WIN32_SHARED_HANDLE_ACCESS);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut dedicated_allocate_info)
            .push_next(&mut export_memory_info)
            .push_next(&mut export_memory_win32_info);

        let memory = match unsafe { raw_device.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe {
                    raw_device.destroy_image(image, None);
                }
                return Err(map_vk_error("vkAllocateMemory", error));
            }
        };

        if let Err(error) = unsafe { raw_device.bind_image_memory(image, memory, 0) } {
            unsafe {
                raw_device.free_memory(memory, None);
                raw_device.destroy_image(image, None);
            }
            return Err(map_vk_error("vkBindImageMemory", error));
        }

        let sync_semaphore =
            create_exportable_timeline_semaphore(raw_instance, raw_device, sync_mode)?;
        let exported_memory_handle = match export_memory_handle(instance, device, memory)? {
            VulkanExternalMemoryHandle::OpaqueWin32Handle(handle) => handle,
        };
        let exported_sync_handle = match sync_semaphore {
            Some(semaphore) => Some(match export_sync_handle(instance, device, semaphore)? {
                VulkanExternalSyncHandle::OpaqueWin32Handle(handle) => handle,
            }),
            None => None,
        };

        let drop_device = raw_device.clone();
        let drop_sync_semaphore = sync_semaphore;
        let drop_callback: wgpu::hal::DropCallback = Box::new(move || unsafe {
            if let Some(semaphore) = drop_sync_semaphore {
                drop_device.destroy_semaphore(semaphore, None);
            }
            drop_device.destroy_image(image, None);
            drop_device.free_memory(memory, None);
        });
        let hal_texture =
            unsafe { device_hal.texture_from_raw(image, &hal_desc, Some(drop_callback)) };

        (
            image,
            memory,
            sync_semaphore,
            requirements.size,
            exported_memory_handle,
            exported_sync_handle,
            hal_texture,
        )
    };

    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &wgpu_desc)
    };

    #[cfg(target_os = "linux")]
    return Ok(CachedExportTexture {
        resource_id: NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed),
        image,
        device_memory: memory,
        sync_semaphore,
        next_sync_value: AtomicU64::new(0),
        allocation_size,
        width: size.width,
        height: size.height,
        format,
        texture,
    });

    #[cfg(target_os = "windows")]
    return Ok(CachedExportTexture {
        resource_id: NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed),
        image,
        device_memory: memory,
        sync_semaphore,
        next_sync_value: AtomicU64::new(0),
        allocation_size,
        width: size.width,
        height: size.height,
        format,
        exported_memory_handle,
        exported_sync_handle,
        texture,
    });
}

fn hal_texture_descriptor(texture: &wgpu::Texture) -> wgpu::hal::TextureDescriptor<'static> {
    wgpu::hal::TextureDescriptor {
        label: Some(EXPORT_TEXTURE_LABEL),
        size: texture.size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: texture.format(),
        usage: wgpu::TextureUses::COPY_DST,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    }
}

fn map_texture_format(format: wgpu::TextureFormat) -> Result<vk::Format, CaptureError> {
    let vk_format = match format {
        wgpu::TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        wgpu::TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        wgpu::TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        wgpu::TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        other => return Err(CaptureError::UnsupportedFormat(other)),
    };
    Ok(vk_format)
}

fn ensure_external_memory_extension(
    enabled_extensions: &[&'static std::ffi::CStr],
) -> Result<(), CaptureError> {
    #[cfg(target_os = "linux")]
    if enabled_extensions.contains(&ash::khr::external_memory_fd::NAME) {
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    if enabled_extensions.contains(&ash::khr::external_memory_win32::NAME) {
        return Ok(());
    }

    Err(CaptureError::ExternalMemoryUnavailable(
        "required Vulkan external-memory export extension is not enabled on this device".into(),
    ))
}

fn ensure_external_semaphore_extension(
    enabled_extensions: &[&'static std::ffi::CStr],
) -> Result<(), CaptureError> {
    #[cfg(target_os = "linux")]
    if enabled_extensions.contains(&ash::khr::external_semaphore_fd::NAME) {
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    if enabled_extensions.contains(&ash::khr::external_semaphore_win32::NAME) {
        return Ok(());
    }

    Err(CaptureError::ExternalMemoryUnavailable(
        "required Vulkan external-semaphore export extension is not enabled on this device".into(),
    ))
}

fn best_sync_mode_for_enabled_extensions(
    enabled_extensions: &[&'static std::ffi::CStr],
) -> Result<VulkanCaptureSyncMode, CaptureError> {
    ensure_external_memory_extension(enabled_extensions)?;
    if ensure_external_semaphore_extension(enabled_extensions).is_ok() {
        return Ok(VulkanCaptureSyncMode::ExportedTimelineSemaphore);
    }
    Ok(VulkanCaptureSyncMode::HostSynchronized)
}

fn find_device_local_memory_type(
    raw_instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    memory_type_bits: u32,
) -> Result<u32, CaptureError> {
    let properties = unsafe { raw_instance.get_physical_device_memory_properties(physical_device) };

    for index in 0..properties.memory_type_count {
        let memory_type = properties.memory_types[index as usize];
        let supported = memory_type_bits & (1 << index) != 0;
        if supported
            && memory_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            return Ok(index);
        }
    }

    Err(CaptureError::ExternalMemoryUnavailable(
        "failed to find a Vulkan DEVICE_LOCAL memory type compatible with the exportable image"
            .into(),
    ))
}

fn map_vk_error(operation: &str, error: vk::Result) -> CaptureError {
    CaptureError::VulkanInteropFailed(format!("{operation} failed: {error:?}"))
}

#[cfg(test)]
mod tests {
    use super::{CaptureError, VulkanCaptureSyncMode, best_sync_mode_for_enabled_extensions};

    #[test]
    fn prefers_timeline_sync_when_memory_and_semaphore_export_are_available() {
        #[cfg(target_os = "linux")]
        let enabled_extensions = [
            ash::khr::external_memory_fd::NAME,
            ash::khr::external_semaphore_fd::NAME,
        ];
        #[cfg(target_os = "windows")]
        let enabled_extensions = [
            ash::khr::external_memory_win32::NAME,
            ash::khr::external_semaphore_win32::NAME,
        ];

        assert_eq!(
            best_sync_mode_for_enabled_extensions(&enabled_extensions).unwrap(),
            VulkanCaptureSyncMode::ExportedTimelineSemaphore
        );
    }

    #[test]
    fn falls_back_to_host_sync_when_only_memory_export_is_available() {
        #[cfg(target_os = "linux")]
        let enabled_extensions = [ash::khr::external_memory_fd::NAME];
        #[cfg(target_os = "windows")]
        let enabled_extensions = [ash::khr::external_memory_win32::NAME];

        assert_eq!(
            best_sync_mode_for_enabled_extensions(&enabled_extensions).unwrap(),
            VulkanCaptureSyncMode::HostSynchronized
        );
    }

    #[test]
    fn rejects_devices_without_external_memory_export_support() {
        let error = best_sync_mode_for_enabled_extensions(&[]).unwrap_err();
        assert!(
            matches!(error, CaptureError::ExternalMemoryUnavailable(message) if message.contains("external-memory"))
        );
    }
}

fn export_handle_type() -> Result<vk::ExternalMemoryHandleTypeFlags, CaptureError> {
    #[cfg(target_os = "linux")]
    {
        return Ok(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    }

    #[cfg(target_os = "windows")]
    {
        return Ok(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    }
}

fn export_semaphore_handle_type() -> Result<vk::ExternalSemaphoreHandleTypeFlags, CaptureError> {
    #[cfg(target_os = "linux")]
    {
        return Ok(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
    }

    #[cfg(target_os = "windows")]
    {
        return Ok(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
    }
}

fn create_exportable_timeline_semaphore(
    raw_instance: &ash::Instance,
    raw_device: &ash::Device,
    sync_mode: VulkanCaptureSyncMode,
) -> Result<Option<vk::Semaphore>, CaptureError> {
    if sync_mode != VulkanCaptureSyncMode::ExportedTimelineSemaphore {
        return Ok(None);
    }

    let handle_type = export_semaphore_handle_type()?;
    let mut export_info = vk::ExportSemaphoreCreateInfo::default().handle_types(handle_type);
    let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);

    #[cfg(target_os = "linux")]
    let create_info = vk::SemaphoreCreateInfo::default()
        .push_next(&mut timeline_info)
        .push_next(&mut export_info);
    #[cfg(target_os = "windows")]
    let mut export_win32_info =
        vk::ExportSemaphoreWin32HandleInfoKHR::default().dw_access(WIN32_SHARED_HANDLE_ACCESS);
    #[cfg(target_os = "windows")]
    let create_info = vk::SemaphoreCreateInfo::default()
        .push_next(&mut timeline_info)
        .push_next(&mut export_info)
        .push_next(&mut export_win32_info);

    let semaphore = unsafe { raw_device.create_semaphore(&create_info, None) }
        .map_err(|error| map_vk_error("vkCreateSemaphore", error))?;

    let _ = raw_instance;
    Ok(Some(semaphore))
}

fn signal_exported_timeline_semaphore(
    device: &wgpu::Device,
    semaphore: vk::Semaphore,
    value: u64,
) -> Result<(), CaptureError> {
    let device_hal = unsafe {
        device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or(CaptureError::UnsupportedBackend(
                "wgpu device is not using Vulkan",
            ))?
    };
    let signal_info = vk::SemaphoreSignalInfo::default()
        .semaphore(semaphore)
        .value(value);
    unsafe { device_hal.raw_device().signal_semaphore(&signal_info) }
        .map_err(|error| map_vk_error("vkSignalSemaphore", error))
}

fn export_memory_handle(
    instance: &wgpu::Instance,
    device: &wgpu::Device,
    memory: vk::DeviceMemory,
) -> Result<VulkanExternalMemoryHandle, CaptureError> {
    #[cfg(target_os = "linux")]
    {
        let instance_hal =
            unsafe {
                instance.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu instance is not using Vulkan"),
                )?
            };
        let device_hal =
            unsafe {
                device.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu device is not using Vulkan"),
                )?
            };
        let external_memory = ash::khr::external_memory_fd::Device::new(
            instance_hal.shared_instance().raw_instance(),
            device_hal.raw_device(),
        );
        let fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let fd = unsafe { external_memory.get_memory_fd(&fd_info) }
            .map_err(|error| map_vk_error("vkGetMemoryFdKHR", error))?;
        let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };
        return Ok(VulkanExternalMemoryHandle::OpaqueFd(owned_fd));
    }

    #[cfg(target_os = "windows")]
    {
        let instance_hal =
            unsafe {
                instance.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu instance is not using Vulkan"),
                )?
            };
        let device_hal =
            unsafe {
                device.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu device is not using Vulkan"),
                )?
            };
        let external_memory = ash::khr::external_memory_win32::Device::new(
            instance_hal.shared_instance().raw_instance(),
            device_hal.raw_device(),
        );
        let handle_info = vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
        let handle = unsafe { external_memory.get_memory_win32_handle(&handle_info) }
            .map_err(|error| map_vk_error("vkGetMemoryWin32HandleKHR", error))?;
        let owned_handle = unsafe { OwnedHandle::from_raw_handle(handle as RawHandle) };
        return Ok(VulkanExternalMemoryHandle::OpaqueWin32Handle(owned_handle));
    }
}

fn export_sync_handle(
    instance: &wgpu::Instance,
    device: &wgpu::Device,
    semaphore: vk::Semaphore,
) -> Result<VulkanExternalSyncHandle, CaptureError> {
    #[cfg(target_os = "linux")]
    {
        let instance_hal =
            unsafe {
                instance.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu instance is not using Vulkan"),
                )?
            };
        let device_hal =
            unsafe {
                device.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu device is not using Vulkan"),
                )?
            };
        let external_semaphore = ash::khr::external_semaphore_fd::Device::new(
            instance_hal.shared_instance().raw_instance(),
            device_hal.raw_device(),
        );
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let fd = unsafe { external_semaphore.get_semaphore_fd(&get_info) }
            .map_err(|error| map_vk_error("vkGetSemaphoreFdKHR", error))?;
        let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };
        return Ok(VulkanExternalSyncHandle::OpaqueFd(owned_fd));
    }

    #[cfg(target_os = "windows")]
    {
        let instance_hal =
            unsafe {
                instance.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu instance is not using Vulkan"),
                )?
            };
        let device_hal =
            unsafe {
                device.as_hal::<wgpu::hal::api::Vulkan>().ok_or(
                    CaptureError::UnsupportedBackend("wgpu device is not using Vulkan"),
                )?
            };
        let external_semaphore = ash::khr::external_semaphore_win32::Device::new(
            instance_hal.shared_instance().raw_instance(),
            device_hal.raw_device(),
        );
        let get_info = vk::SemaphoreGetWin32HandleInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
        let handle = unsafe { external_semaphore.get_semaphore_win32_handle(&get_info) }
            .map_err(|error| map_vk_error("vkGetSemaphoreWin32HandleKHR", error))?;
        let owned_handle = unsafe { OwnedHandle::from_raw_handle(handle as RawHandle) };
        return Ok(VulkanExternalSyncHandle::OpaqueWin32Handle(owned_handle));
    }
}
