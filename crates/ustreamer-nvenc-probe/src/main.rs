#[cfg(not(target_os = "windows"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "ustreamer-nvenc-probe currently requires Windows with a Vulkan-capable NVIDIA GPU"
    )
}

#[cfg(target_os = "windows")]
fn main() -> anyhow::Result<()> {
    windows_probe::run()
}

#[cfg(target_os = "windows")]
mod windows_probe {
    use anyhow::{Context, Result, anyhow, bail};
    use std::env;
    use ustreamer_capture::{
        CaptureError, CapturedFrame, FrameCapture, VulkanCaptureSyncMode, VulkanExternalCapture,
        VulkanExternalSync,
    };
    use ustreamer_encode::{
        FrameEncoder,
        nvenc::{NvencCodec, NvencEncoder, NvencEncoderConfig, NvencExternalSyncDescriptor},
    };
    use ustreamer_proto::quality::{EncodeMode, EncodeParams};

    const NVIDIA_VENDOR_ID: u32 = 0x10de;
    const DEFAULT_WIDTH: u32 = 1280;
    const DEFAULT_HEIGHT: u32 = 720;
    const DEFAULT_BITRATE_BPS: u64 = 25_000_000;
    const DEFAULT_MAX_BITRATE_BPS: u64 = 40_000_000;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RequestedSyncMode {
        Host,
        Timeline,
        Both,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ProbeCodec {
        Hevc,
        Av1,
    }

    impl ProbeCodec {
        fn parse(value: &str) -> Result<Self> {
            match value {
                "hevc" => Ok(Self::Hevc),
                "av1" => Ok(Self::Av1),
                other => bail!("unsupported --codec value {other:?}; expected `hevc` or `av1`"),
            }
        }

        fn nvenc_codec(self) -> NvencCodec {
            match self {
                Self::Hevc => NvencCodec::Hevc,
                Self::Av1 => NvencCodec::Av1,
            }
        }

        fn name(self) -> &'static str {
            match self {
                Self::Hevc => "hevc",
                Self::Av1 => "av1",
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ProbeOptions {
        width: u32,
        height: u32,
        cuda_device: usize,
        codec: ProbeCodec,
        sync_mode: RequestedSyncMode,
        verify_encode_boundary: bool,
    }

    #[derive(Debug)]
    struct ProbeContext {
        instance: wgpu::Instance,
        device: wgpu::Device,
        queue: wgpu::Queue,
        texture: wgpu::Texture,
        adapter_name: String,
        adapter_vendor: u32,
        backend: wgpu::Backend,
    }

    enum ParsedCommand {
        Run(ProbeOptions),
        Help,
    }

    enum ProbeRunStatus {
        Passed,
        Skipped(String),
    }

    pub fn run() -> Result<()> {
        let command = parse_args()?;
        match command {
            ParsedCommand::Help => {
                print_help();
                Ok(())
            }
            ParsedCommand::Run(options) => run_probe(options),
        }
    }

    fn run_probe(options: ProbeOptions) -> Result<()> {
        println!("ustreamer-nvenc-probe");
        println!(
            "Configured probe: {}x{}, cuda_device={}, codec={}, sync_mode={}",
            options.width,
            options.height,
            options.cuda_device,
            options.codec.name(),
            requested_sync_mode_name(options.sync_mode)
        );

        let context = pollster::block_on(ProbeContext::new(options.width, options.height))
            .context("failed to initialize probe GPU context")?;

        println!(
            "Using adapter: {} (vendor=0x{:04x}, backend={:?})",
            context.adapter_name, context.adapter_vendor, context.backend
        );

        let mut encoder = NvencEncoder::with_config_and_cuda_device(
            NvencEncoderConfig {
                codec: options.codec.nvenc_codec(),
                ..Default::default()
            },
            options.cuda_device,
        )
        .map_err(|error| {
            anyhow!(
                "failed to create {} CUDA importer for device {}: {error}",
                options.codec.name(),
                options.cuda_device
            )
        })?;

        let mut failures = Vec::new();
        let mut skipped = Vec::new();
        for &sync_mode in selected_sync_modes(options.sync_mode) {
            match run_single_probe(&context, &mut encoder, &options, sync_mode) {
                Ok(ProbeRunStatus::Passed) => {}
                Ok(ProbeRunStatus::Skipped(reason)) => {
                    println!("[skip] {}: {reason}", capture_sync_mode_name(sync_mode));
                    skipped.push((sync_mode, reason));
                }
                Err(error) => {
                    eprintln!("[fail] {}: {error:#}", capture_sync_mode_name(sync_mode));
                    failures.push((sync_mode, error));
                }
            }
        }

        if failures.is_empty() {
            if skipped.is_empty() {
                println!(
                    "[summary] all selected probe modes passed on adapter {}",
                    context.adapter_name
                );
            } else {
                println!(
                    "[summary] probe completed on adapter {} with {} pass(es) and {} skipped mode(s)",
                    context.adapter_name,
                    selected_sync_modes(options.sync_mode).len() - skipped.len(),
                    skipped.len()
                );
            }
            return Ok(());
        }

        eprintln!("[summary] {} probe mode(s) failed:", failures.len());
        for (sync_mode, error) in &failures {
            eprintln!("  - {}: {error}", capture_sync_mode_name(*sync_mode));
        }

        bail!("NVENC probe failed")
    }

    impl ProbeContext {
        async fn new(width: u32, height: u32) -> Result<Self> {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN,
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .context("failed to request a Vulkan adapter from wgpu")?;
            let info = adapter.get_info();
            if info.backend != wgpu::Backend::Vulkan {
                bail!(
                    "probe requires wgpu Vulkan backend, got {:?} on adapter {}",
                    info.backend,
                    info.name
                );
            }
            if info.vendor != NVIDIA_VENDOR_ID {
                bail!(
                    "probe selected non-NVIDIA adapter {} (vendor=0x{:04x}); ensure Windows runs this process on the RTX GPU",
                    info.name,
                    info.vendor
                );
            }

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("ustreamer-nvenc-probe-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                })
                .await
                .context("failed to request wgpu device")?;

            let texture = create_source_texture(&device, width, height);
            upload_test_pattern(&device, &queue, &texture)
                .context("failed to upload probe test pattern")?;

            Ok(Self {
                instance,
                device,
                queue,
                texture,
                adapter_name: info.name,
                adapter_vendor: info.vendor,
                backend: info.backend,
            })
        }
    }

    fn run_single_probe(
        context: &ProbeContext,
        encoder: &mut NvencEncoder,
        options: &ProbeOptions,
        sync_mode: VulkanCaptureSyncMode,
    ) -> Result<ProbeRunStatus> {
        println!("[probe] {}", capture_sync_mode_name(sync_mode));

        let mut capture = VulkanExternalCapture::with_sync_mode(sync_mode);
        let frame = match capture.capture(
            &context.instance,
            &context.device,
            &context.queue,
            &context.texture,
        ) {
            Ok(frame) => frame,
            Err(CaptureError::ExternalMemoryUnavailable(message))
                if sync_mode == VulkanCaptureSyncMode::ExportedTimelineSemaphore
                    && message.contains("external-semaphore export extension") =>
            {
                return Ok(ProbeRunStatus::Skipped(message));
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "capture failed in {} mode",
                        capture_sync_mode_name(sync_mode)
                    )
                });
            }
        };

        let captured_sync_value = verify_capture_sync(&frame, sync_mode)?;
        let params = probe_encode_params(options.width, options.height);
        let prepared = encoder
            .prepare_frame(&frame, &params)
            .context("failed to prepare exported Vulkan frame for NVENC")?;
        verify_descriptor_sync(&prepared.sync, sync_mode, captured_sync_value)
            .context("prepared-frame sync contract mismatch")?;

        let imported = encoder
            .import_to_cuda(&frame, &params)
            .context("CUDA external-memory import/wait failed")?;
        verify_descriptor_sync(&imported.sync, sync_mode, captured_sync_value)
            .context("CUDA-imported sync contract mismatch")?;
        if imported.resource_handle() == 0 {
            bail!(
                "CUDA import returned a null {} handle",
                imported.resource_kind()
            );
        }

        println!(
            "[pass] CUDA import succeeded: kind={}, handle=0x{:x}, pitch={} bytes",
            imported.resource_kind(),
            imported.resource_handle(),
            imported.pitch_bytes
        );
        drop(imported);

        if options.verify_encode_boundary {
            let encoded = encoder
                .encode(&frame, &params)
                .context("encode() failed after successful CUDA import")?;
            if encoded.data.is_empty() {
                bail!("encode() returned an empty bitstream");
            }
            if !encoded.is_keyframe {
                bail!("encode() did not return a keyframe for the forced-keyframe probe");
            }
            println!(
                "[pass] NVENC encode produced {} bytes (keyframe={}, refine={}, lossless={})",
                encoded.data.len(),
                encoded.is_keyframe,
                encoded.is_refine,
                encoded.is_lossless
            );
        }

        println!("[pass] {} complete", capture_sync_mode_name(sync_mode));
        Ok(ProbeRunStatus::Passed)
    }

    fn verify_capture_sync(
        frame: &CapturedFrame,
        requested_mode: VulkanCaptureSyncMode,
    ) -> Result<Option<u64>> {
        let image = match frame {
            CapturedFrame::VulkanExternalImage(image) => image,
            _ => bail!("probe expected CapturedFrame::VulkanExternalImage"),
        };

        match (requested_mode, image.sync()) {
            (VulkanCaptureSyncMode::HostSynchronized, VulkanExternalSync::HostSynchronized) => {
                println!("[pass] capture returned HostSynchronized");
                Ok(None)
            }
            (
                VulkanCaptureSyncMode::ExportedTimelineSemaphore,
                VulkanExternalSync::ExternalSemaphore { value, .. },
            ) => {
                if *value == 0 {
                    bail!("capture exported a timeline semaphore with value 0");
                }
                println!(
                    "[pass] capture exported external semaphore with timeline value {}",
                    value
                );
                Ok(Some(*value))
            }
            (expected, actual) => bail!(
                "capture sync mismatch: requested {} but got {:?}",
                capture_sync_mode_name(expected),
                actual
            ),
        }
    }

    fn verify_descriptor_sync(
        sync: &NvencExternalSyncDescriptor,
        requested_mode: VulkanCaptureSyncMode,
        expected_value: Option<u64>,
    ) -> Result<()> {
        match (requested_mode, sync, expected_value) {
            (
                VulkanCaptureSyncMode::HostSynchronized,
                NvencExternalSyncDescriptor::HostSynchronized,
                None,
            ) => Ok(()),
            (
                VulkanCaptureSyncMode::ExportedTimelineSemaphore,
                NvencExternalSyncDescriptor::ExternalSemaphore { value, .. },
                Some(expected_value),
            ) if *value == expected_value => Ok(()),
            (expected, actual, value) => bail!(
                "expected {} sync {:?}, got {:?}",
                capture_sync_mode_name(expected),
                value,
                actual
            ),
        }
    }

    fn probe_encode_params(width: u32, height: u32) -> EncodeParams {
        EncodeParams {
            width,
            height,
            target_fps: 60,
            bitrate_bps: DEFAULT_BITRATE_BPS,
            max_bitrate_bps: DEFAULT_MAX_BITRATE_BPS,
            mode: EncodeMode::Interactive,
            force_keyframe: true,
        }
    }

    fn create_source_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ustreamer-nvenc-probe-source"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn upload_test_pattern(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<()> {
        let size = texture.size();
        let unpadded_row_bytes = size
            .width
            .checked_mul(4)
            .ok_or_else(|| anyhow!("probe row-byte size overflow"))?;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let buffer_size = padded_row_bytes as u64 * size.height as u64;

        let upload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ustreamer-nvenc-probe-upload"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        {
            let mut mapped = upload.slice(..).get_mapped_range_mut();
            mapped.fill(0);
            for y in 0..size.height {
                let row_start = (y * padded_row_bytes) as usize;
                for x in 0..size.width {
                    let offset = row_start + (x * 4) as usize;
                    mapped[offset] = scale_to_u8(x, size.width);
                    mapped[offset + 1] = scale_to_u8(y, size.height);
                    mapped[offset + 2] = scale_to_u8(x.saturating_add(y), size.width + size.height);
                    mapped[offset + 3] = 0xff;
                }
            }
        }
        upload.unmap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ustreamer-nvenc-probe-upload-encoder"),
        });
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &upload,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(size.height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            size,
        );
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        Ok(())
    }

    fn scale_to_u8(value: u32, extent: u32) -> u8 {
        let max_index = extent.saturating_sub(1).max(1);
        ((value.min(max_index) * 255) / max_index) as u8
    }

    fn parse_args() -> Result<ParsedCommand> {
        let mut options = ProbeOptions {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            cuda_device: 0,
            codec: ProbeCodec::Hevc,
            sync_mode: RequestedSyncMode::Both,
            verify_encode_boundary: true,
        };

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => return Ok(ParsedCommand::Help),
                "--width" => {
                    options.width = parse_u32_arg("--width", args.next())?;
                }
                "--height" => {
                    options.height = parse_u32_arg("--height", args.next())?;
                }
                "--cuda-device" => {
                    options.cuda_device = parse_usize_arg("--cuda-device", args.next())?;
                }
                "--codec" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value for --codec"))?;
                    options.codec = ProbeCodec::parse(&value)?;
                }
                "--sync-mode" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value for --sync-mode"))?;
                    options.sync_mode = match value.as_str() {
                        "host" => RequestedSyncMode::Host,
                        "timeline" => RequestedSyncMode::Timeline,
                        "both" => RequestedSyncMode::Both,
                        other => {
                            bail!(
                                "unsupported --sync-mode value {other:?}; expected host, timeline, or both"
                            )
                        }
                    };
                }
                "--skip-encode-boundary-check" => {
                    options.verify_encode_boundary = false;
                }
                other => bail!("unknown argument {other:?}; use --help for usage"),
            }
        }

        if options.width == 0 || options.height == 0 {
            bail!("--width and --height must both be greater than zero");
        }

        Ok(ParsedCommand::Run(options))
    }

    fn parse_u32_arg(flag: &str, value: Option<String>) -> Result<u32> {
        value
            .ok_or_else(|| anyhow!("missing value for {flag}"))?
            .parse::<u32>()
            .with_context(|| format!("failed to parse {flag} as u32"))
    }

    fn parse_usize_arg(flag: &str, value: Option<String>) -> Result<usize> {
        value
            .ok_or_else(|| anyhow!("missing value for {flag}"))?
            .parse::<usize>()
            .with_context(|| format!("failed to parse {flag} as usize"))
    }

    fn print_help() {
        println!("ustreamer-nvenc-probe");
        println!();
        println!("Windows probe for Vulkan external-memory export -> CUDA import -> NVENC encode.");
        println!();
        println!("Usage:");
        println!("  cargo run -p ustreamer-nvenc-probe -- [options]");
        println!();
        println!("Options:");
        println!(
            "  --width <pixels>                  Source texture width (default: {DEFAULT_WIDTH})"
        );
        println!(
            "  --height <pixels>                 Source texture height (default: {DEFAULT_HEIGHT})"
        );
        println!("  --cuda-device <index>             CUDA device ordinal (default: 0)");
        println!("  --codec <hevc|av1>                NVENC codec to validate (default: hevc)");
        println!("  --sync-mode <host|timeline|both>  Probe sync handoff modes (default: both)");
        println!(
            "  --skip-encode-boundary-check      Skip the final encode() bitstream validation"
        );
        println!("  -h, --help                        Show this help");
    }

    fn selected_sync_modes(mode: RequestedSyncMode) -> &'static [VulkanCaptureSyncMode] {
        match mode {
            RequestedSyncMode::Host => &[VulkanCaptureSyncMode::HostSynchronized],
            RequestedSyncMode::Timeline => &[VulkanCaptureSyncMode::ExportedTimelineSemaphore],
            RequestedSyncMode::Both => &[
                VulkanCaptureSyncMode::HostSynchronized,
                VulkanCaptureSyncMode::ExportedTimelineSemaphore,
            ],
        }
    }

    fn requested_sync_mode_name(mode: RequestedSyncMode) -> &'static str {
        match mode {
            RequestedSyncMode::Host => "host",
            RequestedSyncMode::Timeline => "timeline",
            RequestedSyncMode::Both => "both",
        }
    }

    fn capture_sync_mode_name(mode: VulkanCaptureSyncMode) -> &'static str {
        match mode {
            VulkanCaptureSyncMode::HostSynchronized => "HostSynchronized",
            VulkanCaptureSyncMode::ExportedTimelineSemaphore => "ExportedTimelineSemaphore",
        }
    }
}
