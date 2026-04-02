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

const EXPORT_TEXTURE_LABEL: &str = "ustreamer-vulkan-external";

static NEXT_RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

/// Exported OS handle for a Vulkan external-memory allocation.
#[derive(Debug)]
pub enum VulkanExternalMemoryHandle {
    #[cfg(target_os = "linux")]
    OpaqueFd(OwnedFd),
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

    #[cfg(target_os = "linux")]
    pub fn opaque_fd(&self) -> BorrowedFd<'_> {
        match &self.handle {
            VulkanExternalMemoryHandle::OpaqueFd(fd) => fd.as_fd(),
        }
    }
}

#[derive(Debug)]
struct CachedExportTexture {
    resource_id: u64,
    image: vk::Image,
    device_memory: vk::DeviceMemory,
    allocation_size: u64,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
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

    fn into_frame(&self, handle: VulkanExternalMemoryHandle) -> VulkanExternalImage {
        VulkanExternalImage {
            resource_id: self.resource_id,
            image: self.image,
            device_memory: self.device_memory,
            allocation_size: self.allocation_size,
            width: self.width,
            height: self.height,
            format: self.format,
            handle,
            _backing_texture: Some(self.texture.clone()),
        }
    }
}

/// Zero-copy Vulkan capture entry point for future NVENC integration.
#[derive(Debug, Default)]
pub struct VulkanExternalCapture {
    cached: Option<CachedExportTexture>,
}

impl VulkanExternalCapture {
    pub fn new() -> Self {
        Self::default()
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
            self.cached = Some(create_cached_export_texture(instance, device, texture)?);
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
        Self {
            resource_id,
            image: vk::Image::from_raw(raw_image),
            device_memory: vk::DeviceMemory::from_raw(raw_device_memory),
            allocation_size,
            width,
            height,
            format,
            handle,
            _backing_texture: None,
        }
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

        let handle = export_memory_handle(instance, device, cached.device_memory)?;
        Ok(CapturedFrame::VulkanExternalImage(
            cached.into_frame(handle),
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

    let (image, memory, allocation_size, hal_texture) = {
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
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut dedicated_allocate_info)
            .push_next(&mut export_memory_info);

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

        let drop_device = raw_device.clone();
        let drop_callback: wgpu::hal::DropCallback = Box::new(move || unsafe {
            drop_device.destroy_image(image, None);
            drop_device.free_memory(memory, None);
        });
        let hal_texture =
            unsafe { device_hal.texture_from_raw(image, &hal_desc, Some(drop_callback)) };

        (image, memory, requirements.size, hal_texture)
    };

    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &wgpu_desc)
    };

    Ok(CachedExportTexture {
        resource_id: NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed),
        image,
        device_memory: memory,
        allocation_size,
        width: size.width,
        height: size.height,
        format,
        texture,
    })
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

fn export_handle_type() -> Result<vk::ExternalMemoryHandleTypeFlags, CaptureError> {
    #[cfg(target_os = "linux")]
    {
        return Ok(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    }

    #[cfg(target_os = "windows")]
    {
        Err(CaptureError::UnsupportedBackend(
            "Vulkan external-memory export is not implemented on Windows yet",
        ))
    }
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
        let _ = (instance, device, memory);
        Err(CaptureError::UnsupportedBackend(
            "Vulkan external-memory export is not implemented on Windows yet",
        ))
    }
}
