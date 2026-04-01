//! Triple-buffered staging buffer capture (cross-platform fallback).

use crate::{CaptureError, CapturedFrame, FrameCapture};

/// Cross-platform frame capture using GPU→CPU staging buffers.
///
/// Uses a ring of staging buffers to overlap capture of frame N
/// with encoding of frame N-1 and rendering of frame N+1.
pub struct StagingCapture {
    buffers: Vec<Option<wgpu::Buffer>>,
    current: usize,
    capacity: usize,
}

impl StagingCapture {
    /// Create a new staging capture with the given number of ring buffers.
    pub fn new(ring_size: usize) -> Self {
        Self {
            buffers: (0..ring_size).map(|_| None).collect(),
            current: 0,
            capacity: ring_size,
        }
    }
}

impl FrameCapture for StagingCapture {
    fn capture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<CapturedFrame, CaptureError> {
        let size = texture.size();
        let format = texture.format();

        let bytes_per_pixel = format
            .block_copy_size(None)
            .ok_or(CaptureError::UnsupportedFormat(format))?;

        // wgpu requires rows to be aligned to 256 bytes
        let unpadded_row_bytes = size.width * bytes_per_pixel;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let buffer_size = (padded_row_bytes * size.height) as u64;

        // Reuse or create buffer
        let buffer = self.buffers[self.current].get_or_insert_with(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ustreamer-staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });

        // Copy texture to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ustreamer-capture"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(size.height),
                },
            },
            texture.size(),
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        rx.recv()
            .map_err(|e| CaptureError::MapFailed(e.to_string()))?
            .map_err(|e| CaptureError::MapFailed(e.to_string()))?;

        // Copy out the data (removing row padding)
        let mapped = slice.get_mapped_range();
        let mut data = Vec::with_capacity((unpadded_row_bytes * size.height) as usize);
        for row in 0..size.height {
            let start = (row * padded_row_bytes) as usize;
            let end = start + unpadded_row_bytes as usize;
            data.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        buffer.unmap();

        self.current = (self.current + 1) % self.capacity;

        Ok(CapturedFrame::CpuBuffer {
            data,
            width: size.width,
            height: size.height,
            stride: unpadded_row_bytes,
            format,
        })
    }
}
