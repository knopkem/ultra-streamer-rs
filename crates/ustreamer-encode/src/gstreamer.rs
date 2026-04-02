use std::{
    borrow::Cow,
    env,
    process::{Command, Stdio},
    sync::OnceLock,
    time::Instant,
};

use gstreamer as gst;
use gstreamer::glib::types::StaticType;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use ustreamer_capture::CapturedFrame;
use ustreamer_proto::quality::{EncodeMode, EncodeParams};

use crate::{
    DecoderConfig, EncodeError, EncodedFrame, FrameEncoder,
    hevc::{decoder_config_from_hevc_access_unit, normalize_hevc_access_unit},
};

const DEFAULT_GST_INSPECT_BINARY: &str = "gst-inspect-1.0";
const SAMPLE_TIMEOUT_MS: u64 = 250;

/// GStreamer codec variants exposed by the current fallback backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GStreamerCodec {
    /// H.265 / HEVC fallback path for AMD and unsupported GPUs.
    Hevc,
}

/// Platform-specific GStreamer hardware encoder choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GStreamerBackend {
    /// Windows fallback using the AMF H.265 encoder plugin.
    WindowsAmfHevc,
    /// Linux fallback using the VA-API H.265 encoder plugin.
    LinuxVaapiHevc,
}

impl GStreamerBackend {
    /// Resolve the best current backend for the active target OS.
    pub fn detect_for_current_platform() -> Result<Self, EncodeError> {
        Self::for_os(env::consts::OS)
    }

    fn for_os(os: &str) -> Result<Self, EncodeError> {
        match os {
            "windows" => Ok(Self::WindowsAmfHevc),
            "linux" => Ok(Self::LinuxVaapiHevc),
            other => Err(EncodeError::UnsupportedConfig(format!(
                "GStreamer fallback is currently planned for Linux/Windows hosts only; platform `{other}` is not wired yet"
            ))),
        }
    }

    /// Human-readable backend name for logs and errors.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::WindowsAmfHevc => "Windows AMF HEVC",
            Self::LinuxVaapiHevc => "Linux VA-API HEVC",
        }
    }

    fn preferred_encoder_elements(self, codec: GStreamerCodec) -> &'static [&'static str] {
        match (self, codec) {
            (Self::WindowsAmfHevc, GStreamerCodec::Hevc) => &["amfh265enc"],
            (Self::LinuxVaapiHevc, GStreamerCodec::Hevc) => &["vah265enc", "vaapih265enc"],
        }
    }

    /// Installation hint used in probe errors.
    pub fn install_hint(self) -> &'static str {
        match self {
            Self::WindowsAmfHevc => {
                "the GStreamer runtime plus the AMF encoder plugin providing `amfh265enc`"
            }
            Self::LinuxVaapiHevc => {
                "the GStreamer runtime plus a VA-API encoder plugin providing `vah265enc` or `vaapih265enc`"
            }
        }
    }

    fn required_elements(self, encoder_element: &str) -> Vec<String> {
        vec![
            "appsrc".to_string(),
            "videoconvert".to_string(),
            encoder_element.to_string(),
            "h265parse".to_string(),
            "appsink".to_string(),
        ]
    }

    fn pipeline_template(
        self,
        codec: GStreamerCodec,
        params: &EncodeParams,
        encoder_element: &str,
    ) -> String {
        let _ = self;
        match codec {
            GStreamerCodec::Hevc => format!(
                concat!(
                    "appsrc name=ustreamer-src is-live=true format=time block=true ",
                    "caps=video/x-raw,format=BGRA,width={width},height={height},framerate={fps}/1 ",
                    "! queue leaky=downstream max-size-buffers=2 ",
                    "! videoconvert ",
                    "! {encoder_element} name=ustreamer-encoder ",
                    "! h265parse config-interval=-1 disable-passthrough=true ",
                    "! video/x-h265,stream-format=byte-stream,alignment=au ",
                    "! appsink name=ustreamer-sink sync=false wait-on-eos=false emit-signals=false max-buffers=1 drop=true"
                ),
                width = params.width,
                height = params.height,
                fps = params.target_fps.max(1),
                encoder_element = encoder_element,
            ),
        }
    }
}

/// Probe result describing the currently selected fallback pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GStreamerPipelinePlan {
    pub backend: GStreamerBackend,
    pub codec: GStreamerCodec,
    pub inspect_binary: String,
    pub encoder_element: String,
    pub required_elements: Vec<String>,
    pub pipeline_template: String,
}

impl GStreamerPipelinePlan {
    /// Probe the current machine for the default GStreamer fallback runtime.
    ///
    /// The probe uses `gst-inspect-1.0` (or `USTREAMER_GST_INSPECT`) and checks
    /// the core `appsrc`/`appsink`/`videoconvert` elements plus the chosen
    /// hardware encoder element (Windows AMF or Linux VA-API candidates). The
    /// encoder element may be overridden with
    /// `USTREAMER_GST_ENCODER_ELEMENT`.
    pub fn probe_default(
        codec: GStreamerCodec,
        params: &EncodeParams,
    ) -> Result<Self, EncodeError> {
        let inspect_binary = env::var("USTREAMER_GST_INSPECT")
            .unwrap_or_else(|_| DEFAULT_GST_INSPECT_BINARY.to_string());
        let encoder_override = env::var("USTREAMER_GST_ENCODER_ELEMENT").ok();
        probe_with_runner(
            env::consts::OS,
            codec,
            params,
            &inspect_binary,
            encoder_override.as_deref(),
            &SystemGstInspectRunner,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PipelineSettings {
    width: u32,
    height: u32,
    target_fps: u32,
}

impl PipelineSettings {
    fn from_params(params: &EncodeParams) -> Self {
        Self {
            width: params.width,
            height: params.height,
            target_fps: params.target_fps.max(1),
        }
    }
}

struct ActivePipeline {
    settings: PipelineSettings,
    pipeline: gst::Pipeline,
    encoder: gst::Element,
    appsrc: gst_app::AppSrc,
    appsink: gst_app::AppSink,
    frame_index: u64,
}

impl Drop for ActivePipeline {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

/// Feature-gated GStreamer fallback encoder for staged CPU BGRA frames.
pub struct GStreamerEncoder {
    codec: GStreamerCodec,
    active: Option<ActivePipeline>,
    decoder_config: Option<DecoderConfig>,
}

impl GStreamerEncoder {
    pub fn new() -> Result<Self, EncodeError> {
        Self::with_codec(GStreamerCodec::Hevc)
    }

    pub fn with_codec(codec: GStreamerCodec) -> Result<Self, EncodeError> {
        ensure_gstreamer_initialized()?;
        Ok(Self {
            codec,
            active: None,
            decoder_config: None,
        })
    }

    fn ensure_pipeline(
        &mut self,
        params: &EncodeParams,
    ) -> Result<&mut ActivePipeline, EncodeError> {
        let settings = PipelineSettings::from_params(params);
        let should_rebuild = self
            .active
            .as_ref()
            .map(|active| {
                active.settings != settings || (params.force_keyframe && active.frame_index > 0)
            })
            .unwrap_or(true);

        if should_rebuild {
            self.rebuild_pipeline(params)?;
        }

        self.active
            .as_mut()
            .ok_or_else(|| EncodeError::InitFailed("GStreamer pipeline was not initialized".into()))
    }

    fn rebuild_pipeline(&mut self, params: &EncodeParams) -> Result<(), EncodeError> {
        ensure_gstreamer_initialized()?;
        self.active.take();
        self.decoder_config = None;

        let plan = GStreamerPipelinePlan::probe_default(self.codec, params)?;
        let pipeline = gst::parse::launch(&plan.pipeline_template)
            .map_err(|error| {
                EncodeError::InitFailed(format!(
                    "failed to construct {} GStreamer pipeline `{}`: {error}",
                    plan.backend.display_name(),
                    plan.pipeline_template
                ))
            })?
            .downcast::<gst::Pipeline>()
            .map_err(|_| {
                EncodeError::InitFailed(format!(
                    "GStreamer pipeline `{}` did not produce a gst::Pipeline",
                    plan.pipeline_template
                ))
            })?;

        let appsrc = pipeline
            .by_name("ustreamer-src")
            .ok_or_else(|| {
                EncodeError::InitFailed("GStreamer pipeline did not expose `ustreamer-src`".into())
            })?
            .downcast::<gst_app::AppSrc>()
            .map_err(|_| {
                EncodeError::InitFailed(
                    "GStreamer pipeline `ustreamer-src` was not an AppSrc".into(),
                )
            })?;
        let appsink = pipeline
            .by_name("ustreamer-sink")
            .ok_or_else(|| {
                EncodeError::InitFailed("GStreamer pipeline did not expose `ustreamer-sink`".into())
            })?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| {
                EncodeError::InitFailed(
                    "GStreamer pipeline `ustreamer-sink` was not an AppSink".into(),
                )
            })?;
        let encoder = pipeline.by_name("ustreamer-encoder").ok_or_else(|| {
            EncodeError::InitFailed("GStreamer pipeline did not expose `ustreamer-encoder`".into())
        })?;

        apply_encoder_properties(&encoder, params);
        pipeline.set_state(gst::State::Playing).map_err(|error| {
            EncodeError::InitFailed(format!(
                "failed to set {} GStreamer pipeline to Playing: {error}",
                plan.backend.display_name()
            ))
        })?;

        self.active = Some(ActivePipeline {
            settings: PipelineSettings::from_params(params),
            pipeline,
            encoder,
            appsrc,
            appsink,
            frame_index: 0,
        });
        Ok(())
    }
}

impl FrameEncoder for GStreamerEncoder {
    fn encode(
        &mut self,
        frame: &CapturedFrame,
        params: &EncodeParams,
    ) -> Result<EncodedFrame, EncodeError> {
        let encode_started_at = Instant::now();
        let (bytes, width, height) = prepare_cpu_frame_bytes(frame)?;
        let decoder_dimensions = (width, height);
        let (encoded_annex_b, is_keyframe) = {
            let active = self.ensure_pipeline(params)?;
            apply_encoder_properties(&active.encoder, params);

            let fps = u64::from(active.settings.target_fps.max(1));
            let frame_index = active.frame_index;
            let pts_ns = frame_index.saturating_mul(1_000_000_000u64) / fps;
            let duration_ns = (1_000_000_000u64 / fps).max(1);

            let mut buffer = gst::Buffer::from_mut_slice(bytes.into_owned());
            let buffer_ref = buffer.get_mut().ok_or_else(|| {
                EncodeError::EncodeFailed("failed to get mutable GStreamer input buffer".into())
            })?;
            buffer_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            buffer_ref.set_dts(gst::ClockTime::from_nseconds(pts_ns));
            buffer_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));

            active.appsrc.push_buffer(buffer).map_err(|error| {
                EncodeError::EncodeFailed(format!(
                    "failed to push staged BGRA frame into GStreamer appsrc: {error}"
                ))
            })?;

            let sample = active
                .appsink
                .try_pull_sample(gst::ClockTime::from_mseconds(SAMPLE_TIMEOUT_MS))
                .ok_or_else(|| {
                    EncodeError::EncodeFailed(format!(
                        "timed out waiting for encoded HEVC access unit from GStreamer after {SAMPLE_TIMEOUT_MS}ms"
                    ))
                })?;
            let sample_buffer = sample.buffer().ok_or_else(|| {
                EncodeError::EncodeFailed(
                    "GStreamer appsink returned a sample without a buffer".into(),
                )
            })?;
            let sample_map = sample_buffer.map_readable().map_err(|error| {
                EncodeError::EncodeFailed(format!(
                    "failed to map GStreamer encoded sample for reading: {error}"
                ))
            })?;
            let encoded_annex_b = sample_map.as_slice().to_vec();
            let is_keyframe = !sample_buffer.flags().contains(gst::BufferFlags::DELTA_UNIT);

            active.frame_index = active.frame_index.saturating_add(1);
            (encoded_annex_b, is_keyframe)
        };

        if is_keyframe && self.decoder_config.is_none() {
            self.decoder_config = Some(
                decoder_config_from_hevc_access_unit(
                    &encoded_annex_b,
                    decoder_dimensions.0,
                    decoder_dimensions.1,
                )
                .map_err(|error| {
                    EncodeError::EncodeFailed(format!(
                        "failed to derive HEVC decoder config from GStreamer keyframe: {error}"
                    ))
                })?,
            );
        }

        let encoded = normalize_hevc_access_unit(&encoded_annex_b).map_err(|error| {
            EncodeError::EncodeFailed(format!(
                "failed to normalize GStreamer HEVC access unit for browser decode: {error}"
            ))
        })?;

        Ok(EncodedFrame {
            data: encoded,
            is_keyframe,
            is_refine: params.mode == EncodeMode::LosslessRefine,
            is_lossless: false,
            encode_time_us: encode_started_at
                .elapsed()
                .as_micros()
                .min(u64::MAX as u128) as u64,
        })
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, EncodeError> {
        Ok(Vec::new())
    }

    fn decoder_config(&self) -> Option<DecoderConfig> {
        self.decoder_config.clone()
    }
}

trait GstInspectRunner {
    fn element_available(&self, inspect_binary: &str, element: &str) -> Result<bool, String>;
}

struct SystemGstInspectRunner;

impl GstInspectRunner for SystemGstInspectRunner {
    fn element_available(&self, inspect_binary: &str, element: &str) -> Result<bool, String> {
        Command::new(inspect_binary)
            .arg(element)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .map_err(|error| error.to_string())
    }
}

fn probe_with_runner(
    os: &str,
    codec: GStreamerCodec,
    params: &EncodeParams,
    inspect_binary: &str,
    encoder_override: Option<&str>,
    runner: &dyn GstInspectRunner,
) -> Result<GStreamerPipelinePlan, EncodeError> {
    let backend = GStreamerBackend::for_os(os)?;
    let encoder_candidates = if let Some(override_element) = encoder_override {
        vec![override_element.to_string()]
    } else {
        backend
            .preferred_encoder_elements(codec)
            .iter()
            .map(|element| (*element).to_string())
            .collect::<Vec<_>>()
    };
    let mut candidate_failures = Vec::new();
    for encoder_element in encoder_candidates {
        let required_elements = backend.required_elements(&encoder_element);
        let mut missing = Vec::new();
        for element in &required_elements {
            match runner.element_available(inspect_binary, element) {
                Ok(true) => {}
                Ok(false) => missing.push(element.clone()),
                Err(error) => {
                    return Err(EncodeError::InitFailed(format!(
                        "failed to run `{inspect_binary}` while probing {} GStreamer fallback: {error}. Install {}.",
                        backend.display_name(),
                        backend.install_hint()
                    )));
                }
            }
        }

        if missing.is_empty() {
            return Ok(GStreamerPipelinePlan {
                backend,
                codec,
                inspect_binary: inspect_binary.to_string(),
                encoder_element: encoder_element.clone(),
                required_elements,
                pipeline_template: backend.pipeline_template(codec, params, &encoder_element),
            });
        }

        candidate_failures.push(format!(
            "{encoder_element} (missing: {})",
            missing.join(", ")
        ));
    }
    Err(EncodeError::UnsupportedConfig(format!(
        "missing GStreamer elements for {} fallback. Tried {}. Install {}.",
        backend.display_name(),
        candidate_failures.join("; "),
        backend.install_hint()
    )))
}

fn ensure_gstreamer_initialized() -> Result<(), EncodeError> {
    static GST_INIT: OnceLock<Result<(), String>> = OnceLock::new();

    match GST_INIT.get_or_init(|| gst::init().map_err(|error| error.to_string())) {
        Ok(()) => Ok(()),
        Err(message) => Err(EncodeError::InitFailed(format!(
            "failed to initialize GStreamer runtime: {message}"
        ))),
    }
}

fn prepare_cpu_frame_bytes(
    frame: &CapturedFrame,
) -> Result<(Cow<'_, [u8]>, u32, u32), EncodeError> {
    let CapturedFrame::CpuBuffer {
        data,
        width,
        height,
        stride,
        format,
    } = frame
    else {
        return Err(EncodeError::UnsupportedFrame(
            "GStreamer fallback currently requires `CapturedFrame::CpuBuffer` input".into(),
        ));
    };

    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| EncodeError::UnsupportedFrame("BGRA row-bytes overflow".into()))?
        as usize;
    let stride = *stride as usize;

    match format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {}
        other => {
            return Err(EncodeError::UnsupportedFrame(format!(
                "GStreamer fallback currently supports only BGRA8 CPU frames; got {other:?}"
            )));
        }
    }

    if stride < row_bytes {
        return Err(EncodeError::UnsupportedFrame(format!(
            "GStreamer fallback received stride {stride} smaller than BGRA row width {row_bytes}"
        )));
    }

    let required_len = stride
        .checked_mul(*height as usize)
        .ok_or_else(|| EncodeError::UnsupportedFrame("BGRA frame size overflow".into()))?;
    if data.len() < required_len {
        return Err(EncodeError::UnsupportedFrame(format!(
            "GStreamer fallback received CPU buffer with length {} but expected at least {required_len}",
            data.len()
        )));
    }

    if stride == row_bytes {
        return Ok((Cow::Borrowed(data.as_slice()), *width, *height));
    }

    let mut packed = Vec::with_capacity(row_bytes * (*height as usize));
    for row in 0..(*height as usize) {
        let row_start = row * stride;
        packed.extend_from_slice(&data[row_start..row_start + row_bytes]);
    }
    Ok((Cow::Owned(packed), *width, *height))
}

fn apply_encoder_properties(encoder: &gst::Element, params: &EncodeParams) {
    let bitrate_kbps = (params.bitrate_bps / 1_000).max(1);
    let max_bitrate_kbps = (params.max_bitrate_bps / 1_000).max(bitrate_kbps);
    let keyframe_interval = u64::from(params.target_fps.max(1));

    set_numeric_property_if_present(encoder, "bitrate", bitrate_kbps);
    set_numeric_property_if_present(encoder, "max-bitrate", max_bitrate_kbps);
    set_numeric_property_if_present(encoder, "target-bitrate", bitrate_kbps);
    set_numeric_property_if_present(encoder, "gop-size", keyframe_interval);
    set_numeric_property_if_present(encoder, "key-int-max", keyframe_interval);
    set_bool_property_if_present(encoder, "aud", true);
}

fn set_numeric_property_if_present(element: &gst::Element, property_name: &str, value: u64) {
    let Some(param_spec) = element.find_property(property_name) else {
        return;
    };
    let value_type = param_spec.value_type();
    if value_type == u32::static_type() {
        element.set_property(property_name, value.min(u32::MAX as u64) as u32);
    } else if value_type == u64::static_type() {
        element.set_property(property_name, value);
    } else if value_type == i32::static_type() {
        element.set_property(property_name, value.min(i32::MAX as u64) as i32);
    } else if value_type == i64::static_type() {
        element.set_property(property_name, value.min(i64::MAX as u64) as i64);
    }
}

fn set_bool_property_if_present(element: &gst::Element, property_name: &str, value: bool) {
    let Some(param_spec) = element.find_property(property_name) else {
        return;
    };
    if param_spec.value_type() == bool::static_type() {
        element.set_property(property_name, value);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        DEFAULT_GST_INSPECT_BINARY, GStreamerBackend, GStreamerCodec, GstInspectRunner,
        prepare_cpu_frame_bytes, probe_with_runner,
    };
    use crate::EncodeError;
    use ustreamer_capture::CapturedFrame;
    use ustreamer_proto::quality::EncodeParams;

    #[derive(Debug, Default)]
    struct FakeGstInspectRunner {
        available_elements: HashSet<String>,
        error: Option<String>,
    }

    impl FakeGstInspectRunner {
        fn with_elements(elements: impl IntoIterator<Item = &'static str>) -> Self {
            Self {
                available_elements: elements.into_iter().map(str::to_string).collect(),
                error: None,
            }
        }

        fn with_error(message: &str) -> Self {
            Self {
                available_elements: HashSet::new(),
                error: Some(message.to_string()),
            }
        }
    }

    impl GstInspectRunner for FakeGstInspectRunner {
        fn element_available(&self, _inspect_binary: &str, element: &str) -> Result<bool, String> {
            if let Some(error) = &self.error {
                return Err(error.clone());
            }
            Ok(self.available_elements.contains(element))
        }
    }

    #[test]
    fn selects_expected_backend_from_target_os() {
        assert_eq!(
            GStreamerBackend::for_os("windows").unwrap(),
            GStreamerBackend::WindowsAmfHevc
        );
        assert_eq!(
            GStreamerBackend::for_os("linux").unwrap(),
            GStreamerBackend::LinuxVaapiHevc
        );
        assert!(matches!(
            GStreamerBackend::for_os("macos"),
            Err(EncodeError::UnsupportedConfig(message))
                if message.contains("macos")
        ));
    }

    #[test]
    fn builds_windows_amf_pipeline_template() {
        let params = EncodeParams {
            width: 2560,
            height: 1440,
            target_fps: 60,
            ..Default::default()
        };
        let runner = FakeGstInspectRunner::with_elements([
            "appsrc",
            "videoconvert",
            "amfh265enc",
            "h265parse",
            "appsink",
        ]);

        let plan = probe_with_runner(
            "windows",
            GStreamerCodec::Hevc,
            &params,
            DEFAULT_GST_INSPECT_BINARY,
            None,
            &runner,
        )
        .unwrap();

        assert_eq!(plan.backend, GStreamerBackend::WindowsAmfHevc);
        assert_eq!(plan.encoder_element, "amfh265enc");
        assert!(plan.pipeline_template.contains("width=2560"));
        assert!(plan.pipeline_template.contains("height=1440"));
        assert!(plan.pipeline_template.contains("framerate=60/1"));
        assert!(plan.pipeline_template.contains("appsrc name=ustreamer-src"));
        assert!(plan.pipeline_template.contains("name=ustreamer-encoder"));
        assert!(plan.pipeline_template.contains("stream-format=byte-stream"));
        assert!(plan.pipeline_template.contains("alignment=au"));
        assert!(
            plan.pipeline_template
                .contains("appsink name=ustreamer-sink")
        );
    }

    #[test]
    fn auto_detects_linux_va_encoder_candidates() {
        let params = EncodeParams::default();
        let runner = FakeGstInspectRunner::with_elements([
            "appsrc",
            "videoconvert",
            "vah265enc",
            "h265parse",
            "appsink",
        ]);

        let plan = probe_with_runner(
            "linux",
            GStreamerCodec::Hevc,
            &params,
            DEFAULT_GST_INSPECT_BINARY,
            None,
            &runner,
        )
        .unwrap();

        assert_eq!(plan.backend, GStreamerBackend::LinuxVaapiHevc);
        assert_eq!(plan.encoder_element, "vah265enc");
    }

    #[test]
    fn falls_back_to_legacy_linux_vaapi_encoder_candidate() {
        let params = EncodeParams::default();
        let runner = FakeGstInspectRunner::with_elements([
            "appsrc",
            "videoconvert",
            "vaapih265enc",
            "h265parse",
            "appsink",
        ]);

        let plan = probe_with_runner(
            "linux",
            GStreamerCodec::Hevc,
            &params,
            DEFAULT_GST_INSPECT_BINARY,
            None,
            &runner,
        )
        .unwrap();

        assert_eq!(plan.backend, GStreamerBackend::LinuxVaapiHevc);
        assert_eq!(plan.encoder_element, "vaapih265enc");
    }

    #[test]
    fn custom_encoder_override_changes_probe_and_pipeline() {
        let params = EncodeParams::default();
        let runner = FakeGstInspectRunner::with_elements([
            "appsrc",
            "videoconvert",
            "vah265enc",
            "h265parse",
            "appsink",
        ]);

        let plan = probe_with_runner(
            "linux",
            GStreamerCodec::Hevc,
            &params,
            DEFAULT_GST_INSPECT_BINARY,
            Some("vah265enc"),
            &runner,
        )
        .unwrap();

        assert_eq!(plan.backend, GStreamerBackend::LinuxVaapiHevc);
        assert_eq!(plan.encoder_element, "vah265enc");
        assert!(
            plan.required_elements
                .iter()
                .any(|element| element == "vah265enc")
        );
        assert!(
            plan.pipeline_template
                .contains("! vah265enc name=ustreamer-encoder ")
        );
    }

    #[test]
    fn reports_missing_elements_with_install_hint() {
        let params = EncodeParams::default();
        let runner = FakeGstInspectRunner::with_elements(["appsrc", "videoconvert", "appsink"]);

        let error = probe_with_runner(
            "linux",
            GStreamerCodec::Hevc,
            &params,
            DEFAULT_GST_INSPECT_BINARY,
            None,
            &runner,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            EncodeError::UnsupportedConfig(message)
                if message.contains("vah265enc")
                    && message.contains("vaapih265enc")
                    && message.contains("h265parse")
                    && message.contains("VA-API")
        ));
    }

    #[test]
    fn reports_missing_gst_inspect_binary_cleanly() {
        let params = EncodeParams::default();
        let runner = FakeGstInspectRunner::with_error("No such file or directory");

        let error = probe_with_runner(
            "windows",
            GStreamerCodec::Hevc,
            &params,
            DEFAULT_GST_INSPECT_BINARY,
            None,
            &runner,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            EncodeError::InitFailed(message)
                if message.contains(DEFAULT_GST_INSPECT_BINARY)
                    && message.contains("amfh265enc")
        ));
    }

    #[test]
    fn packs_strided_cpu_frames_into_tight_bgra_rows() {
        let frame = CapturedFrame::CpuBuffer {
            data: vec![1, 2, 3, 4, 9, 9, 9, 9, 5, 6, 7, 8, 8, 8, 8, 8],
            width: 1,
            height: 2,
            stride: 8,
            format: wgpu::TextureFormat::Bgra8Unorm,
        };

        let (bytes, width, height) = prepare_cpu_frame_bytes(&frame).unwrap();
        assert_eq!(width, 1);
        assert_eq!(height, 2);
        assert_eq!(bytes.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn rejects_non_bgra_cpu_frames() {
        let frame = CapturedFrame::CpuBuffer {
            data: vec![0; 4],
            width: 1,
            height: 1,
            stride: 4,
            format: wgpu::TextureFormat::Rgba8Unorm,
        };

        let error = prepare_cpu_frame_bytes(&frame).unwrap_err();
        assert!(matches!(
            error,
            EncodeError::UnsupportedFrame(message) if message.contains("BGRA8")
        ));
    }
}
