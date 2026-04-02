//! macOS VideoToolbox HEVC encoder backend.
//!
//! The current implementation keeps VideoToolbox's native length-prefixed
//! HEVC access units and extracts `hvcC` decoder configuration for browser-side
//! WebCodecs setup.

use std::ptr::{self, NonNull};
use std::sync::{
    Mutex,
    mpsc::{self, Receiver, SyncSender},
};
use std::time::{Duration, Instant};

use objc2_core_foundation::{
    CFArray, CFBoolean, CFData, CFDictionary, CFNumber, CFRetained, CFString, CFType,
};
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMTime,
    kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms, kCMSampleAttachmentKey_NotSync,
    kCMTimeInvalid, kCMVideoCodecType_HEVC,
};
use objc2_core_video::{
    CVImageBuffer, CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBytesPerRow, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress, kCVPixelBufferHeightKey, kCVPixelBufferPixelFormatTypeKey,
    kCVPixelBufferWidthKey, kCVPixelFormatType_32BGRA, kCVPixelFormatType_64RGBAHalf,
    kCVReturnSuccess,
};
use objc2_video_toolbox::{
    VTCompressionSession, VTSessionSetProperty, kVTCompressionPropertyKey_AllowFrameReordering,
    kVTCompressionPropertyKey_AllowTemporalCompression, kVTCompressionPropertyKey_AverageBitRate,
    kVTCompressionPropertyKey_ExpectedFrameRate, kVTCompressionPropertyKey_MaxKeyFrameInterval,
    kVTCompressionPropertyKey_ProfileLevel, kVTCompressionPropertyKey_Quality,
    kVTCompressionPropertyKey_RealTime, kVTEncodeFrameOptionKey_ForceKeyFrame,
    kVTProfileLevel_HEVC_Main_AutoLevel, kVTProfileLevel_HEVC_Main10_AutoLevel,
    kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
};
use ustreamer_capture::CapturedFrame;
use ustreamer_proto::quality::{EncodeMode, EncodeParams};

use crate::{DecoderConfig, EncodeError, EncodedFrame, FrameEncoder};

const CALLBACK_TIMEOUT: Duration = Duration::from_millis(750);
const DEFAULT_MAIN_CODEC: &str = "hvc1.1.6.L153.B0";
const DEFAULT_MAIN10_CODEC: &str = "hvc1.2.6.L153.B0";

/// Configuration for the VideoToolbox encoder.
#[derive(Debug, Clone)]
pub struct VideoToolboxEncoderConfig {
    /// RFC 6381 codec string used for 8-bit HEVC streams.
    pub main_codec: String,
    /// RFC 6381 codec string used for 10-bit HEVC streams.
    pub main10_codec: String,
    /// Timeout used while waiting for VideoToolbox callbacks.
    pub callback_timeout: Duration,
}

impl Default for VideoToolboxEncoderConfig {
    fn default() -> Self {
        Self {
            main_codec: DEFAULT_MAIN_CODEC.into(),
            main10_codec: DEFAULT_MAIN10_CODEC.into(),
            callback_timeout: CALLBACK_TIMEOUT,
        }
    }
}

/// Direct HEVC encoder backed by `VTCompressionSession`.
#[derive(Debug)]
pub struct VideoToolboxEncoder {
    config: VideoToolboxEncoderConfig,
    session: Option<SessionHandle>,
    callback_state: Box<CallbackState>,
    session_spec: Option<SessionSpec>,
    runtime_config: Option<RuntimeConfig>,
    decoder_config: Option<DecoderConfig>,
    frame_index: u64,
}

#[derive(Debug)]
struct CallbackState {
    pending: Mutex<Option<PendingEncode>>,
}

#[derive(Debug)]
struct PendingEncode {
    sender: SyncSender<Result<CallbackResult, String>>,
    started_at: Instant,
    fallback_keyframe: bool,
    fallback_refine: bool,
    fallback_lossless: bool,
}

#[derive(Debug)]
struct CallbackResult {
    frame: EncodedFrame,
    decoder_config: Option<DecoderConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SessionSpec {
    width: u32,
    height: u32,
    pixel_format: u32,
    profile: HevcProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RuntimeConfig {
    target_fps: u32,
    bitrate_bps: u64,
    max_bitrate_bps: u64,
    mode: EncodeMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HevcProfile {
    Main,
    Main10,
}

#[derive(Debug)]
struct SessionHandle(NonNull<VTCompressionSession>);

unsafe impl Send for SessionHandle {}

impl SessionHandle {
    unsafe fn from_raw(ptr: NonNull<VTCompressionSession>) -> Self {
        Self(ptr)
    }

    fn as_ref(&self) -> &VTCompressionSession {
        unsafe { self.0.as_ref() }
    }

    fn invalidate(&self) {
        unsafe { self.as_ref().invalidate() };
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        let retained = unsafe { CFRetained::<VTCompressionSession>::from_raw(self.0) };
        drop(retained);
    }
}

impl VideoToolboxEncoder {
    /// Create a new VideoToolbox HEVC encoder with default browser codec strings.
    pub fn new() -> Self {
        Self::with_config(VideoToolboxEncoderConfig::default())
    }

    /// Create a new VideoToolbox encoder with explicit browser codec strings.
    pub fn with_config(config: VideoToolboxEncoderConfig) -> Self {
        Self {
            config,
            session: None,
            callback_state: Box::new(CallbackState {
                pending: Mutex::new(None),
            }),
            session_spec: None,
            runtime_config: None,
            decoder_config: None,
            frame_index: 0,
        }
    }

    fn encode_pixel_buffer(
        &mut self,
        pixel_buffer: &CVPixelBuffer,
        width: u32,
        height: u32,
        pixel_format: u32,
        params: &EncodeParams,
    ) -> Result<EncodedFrame, EncodeError> {
        if width != params.width || height != params.height {
            return Err(EncodeError::UnsupportedConfig(format!(
                "VideoToolboxEncoder does not scale frames yet; captured frame is {}x{} but EncodeParams requested {}x{}",
                width, height, params.width, params.height
            )));
        }

        let session_spec = SessionSpec {
            width,
            height,
            pixel_format,
            profile: profile_for_pixel_format(pixel_format)?,
        };
        let runtime_config = RuntimeConfig {
            target_fps: params.target_fps.max(1),
            bitrate_bps: params.bitrate_bps,
            max_bitrate_bps: params.max_bitrate_bps.max(params.bitrate_bps),
            mode: params.mode,
        };

        self.ensure_session(session_spec)?;
        self.apply_runtime_config(runtime_config)?;

        let session = self.session.as_ref().ok_or_else(|| {
            EncodeError::InitFailed("VideoToolbox session was not created".into())
        })?;

        let force_keyframe = self.frame_index == 0 || params.force_keyframe;
        let (tx, rx) = mpsc::sync_channel(1);
        {
            let mut pending =
                self.callback_state.pending.lock().map_err(|_| {
                    EncodeError::EncodeFailed("callback state lock poisoned".into())
                })?;
            *pending = Some(PendingEncode {
                sender: tx,
                started_at: Instant::now(),
                fallback_keyframe: force_keyframe,
                fallback_refine: matches!(params.mode, EncodeMode::LosslessRefine),
                fallback_lossless: false,
            });
        }

        let force_keyframe_dict = force_keyframe.then(force_keyframe_dictionary);
        let frame_properties: Option<&CFDictionary> = force_keyframe_dict
            .as_ref()
            .map(|dict| (&**dict).as_opaque());
        let presentation_time =
            unsafe { CMTime::new(self.frame_index as i64, runtime_config.target_fps as i32) };
        let duration = unsafe { CMTime::new(1, runtime_config.target_fps as i32) };
        let image_buffer: &CVImageBuffer = pixel_buffer;

        let status = unsafe {
            session.as_ref().encode_frame(
                image_buffer,
                presentation_time,
                duration,
                frame_properties,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if status != 0 {
            self.clear_pending();
            return Err(EncodeError::EncodeFailed(format!(
                "VTCompressionSessionEncodeFrame failed with status {status}"
            )));
        }

        let complete_status = unsafe { session.as_ref().complete_frames(presentation_time) };
        if complete_status != 0 {
            self.clear_pending();
            return Err(EncodeError::EncodeFailed(format!(
                "VTCompressionSessionCompleteFrames failed with status {complete_status}"
            )));
        }

        let result = wait_for_callback(&rx, self.config.callback_timeout)?;
        if let Some(decoder_config) = result.decoder_config {
            self.decoder_config = Some(DecoderConfig {
                codec: self.codec_string(session_spec.profile),
                ..decoder_config
            });
        }

        self.frame_index += 1;
        Ok(result.frame)
    }

    fn encode_cpu_buffer(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        stride: u32,
        format: wgpu::TextureFormat,
        params: &EncodeParams,
    ) -> Result<EncodedFrame, EncodeError> {
        let pixel_format =
            cv_pixel_format_for_texture(format).ok_or(EncodeError::UnsupportedConfig(format!(
                "unsupported CPU buffer texture format {format:?}"
            )))?;

        let mut pixel_buffer = ptr::null_mut();
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                width as usize,
                height as usize,
                pixel_format,
                None,
                NonNull::from(&mut pixel_buffer),
            )
        };
        if status != kCVReturnSuccess {
            return Err(EncodeError::EncodeFailed(format!(
                "CVPixelBufferCreate failed with status {status}"
            )));
        }

        let pixel_buffer_ptr = NonNull::new(pixel_buffer).ok_or_else(|| {
            EncodeError::EncodeFailed("CVPixelBufferCreate returned a null pixel buffer".into())
        })?;
        let pixel_buffer = unsafe { CFRetained::from_raw(pixel_buffer_ptr) };
        let lock_flags = CVPixelBufferLockFlags(0);
        let lock_status = unsafe { CVPixelBufferLockBaseAddress(&pixel_buffer, lock_flags) };
        if lock_status != kCVReturnSuccess {
            return Err(EncodeError::EncodeFailed(format!(
                "CVPixelBufferLockBaseAddress failed with status {lock_status}"
            )));
        }

        let copy_result =
            copy_cpu_frame_into_pixel_buffer(&pixel_buffer, data, width, height, stride, format);
        let unlock_status = unsafe { CVPixelBufferUnlockBaseAddress(&pixel_buffer, lock_flags) };
        if unlock_status != kCVReturnSuccess {
            return Err(EncodeError::EncodeFailed(format!(
                "CVPixelBufferUnlockBaseAddress failed with status {unlock_status}"
            )));
        }
        copy_result?;

        self.encode_pixel_buffer(&pixel_buffer, width, height, pixel_format, params)
    }

    fn ensure_session(&mut self, spec: SessionSpec) -> Result<(), EncodeError> {
        if self.session_spec == Some(spec) && self.session.is_some() {
            return Ok(());
        }

        self.invalidate_session();

        let encoder_specification = hardware_encoder_specification();
        let source_attributes = source_image_buffer_attributes(spec);
        let mut session_ptr = ptr::null_mut();
        let output_refcon = (&mut *self.callback_state) as *mut CallbackState as *mut _;
        let status = unsafe {
            VTCompressionSession::create(
                None,
                spec.width as i32,
                spec.height as i32,
                kCMVideoCodecType_HEVC,
                Some((&*encoder_specification).as_opaque()),
                Some((&*source_attributes).as_opaque()),
                None,
                Some(videotoolbox_output_callback),
                output_refcon,
                NonNull::from(&mut session_ptr),
            )
        };
        if status != 0 {
            return Err(EncodeError::InitFailed(format!(
                "VTCompressionSessionCreate failed with status {status}"
            )));
        }

        let session_ptr = NonNull::new(session_ptr).ok_or_else(|| {
            EncodeError::InitFailed("VideoToolbox returned a null session".into())
        })?;
        let session = unsafe { SessionHandle::from_raw(session_ptr) };

        set_property(
            session.as_ref(),
            unsafe { kVTCompressionPropertyKey_RealTime },
            CFBoolean::new(true),
        )?;
        set_property(
            session.as_ref(),
            unsafe { kVTCompressionPropertyKey_AllowFrameReordering },
            CFBoolean::new(false),
        )?;
        set_property(
            session.as_ref(),
            unsafe { kVTCompressionPropertyKey_ProfileLevel },
            profile_level(spec.profile),
        )?;

        let prepare_status = unsafe { session.as_ref().prepare_to_encode_frames() };
        if prepare_status != 0 {
            return Err(EncodeError::InitFailed(format!(
                "VTCompressionSessionPrepareToEncodeFrames failed with status {prepare_status}"
            )));
        }

        self.session = Some(session);
        self.session_spec = Some(spec);
        self.runtime_config = None;
        self.decoder_config = None;
        self.frame_index = 0;
        Ok(())
    }

    fn apply_runtime_config(&mut self, config: RuntimeConfig) -> Result<(), EncodeError> {
        if self.runtime_config == Some(config) {
            return Ok(());
        }

        let session = self
            .session
            .as_ref()
            .ok_or_else(|| EncodeError::InitFailed("VideoToolbox session is missing".into()))?;
        let session = session.as_ref();

        set_property(
            session,
            unsafe { kVTCompressionPropertyKey_AllowTemporalCompression },
            CFBoolean::new(config.mode != EncodeMode::LosslessRefine),
        )?;
        let expected_fps = CFNumber::new_i32(config.target_fps.min(i32::MAX as u32) as i32);
        set_property(
            session,
            unsafe { kVTCompressionPropertyKey_ExpectedFrameRate },
            &expected_fps,
        )?;
        let max_keyframe_interval =
            CFNumber::new_i32(config.target_fps.min(i32::MAX as u32) as i32);
        set_property(
            session,
            unsafe { kVTCompressionPropertyKey_MaxKeyFrameInterval },
            &max_keyframe_interval,
        )?;
        let bitrate = CFNumber::new_i64(config.bitrate_bps.min(i64::MAX as u64) as i64);
        set_property(
            session,
            unsafe { kVTCompressionPropertyKey_AverageBitRate },
            &bitrate,
        )?;

        let quality = match config.mode {
            EncodeMode::Interactive => 0.78,
            EncodeMode::IdleLowFps => 0.9,
            EncodeMode::LosslessRefine => 1.0,
        };
        let quality = CFNumber::new_f32(quality);
        set_property(
            session,
            unsafe { kVTCompressionPropertyKey_Quality },
            &quality,
        )?;

        self.runtime_config = Some(config);
        Ok(())
    }

    fn clear_pending(&self) {
        if let Ok(mut pending) = self.callback_state.pending.lock() {
            pending.take();
        }
    }

    fn invalidate_session(&mut self) {
        self.clear_pending();
        if let Some(session) = self.session.take() {
            session.invalidate();
        }
        self.session_spec = None;
        self.runtime_config = None;
        self.decoder_config = None;
        self.frame_index = 0;
    }

    fn codec_string(&self, profile: HevcProfile) -> String {
        match profile {
            HevcProfile::Main => self.config.main_codec.clone(),
            HevcProfile::Main10 => self.config.main10_codec.clone(),
        }
    }
}

impl Default for VideoToolboxEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VideoToolboxEncoder {
    fn drop(&mut self) {
        self.invalidate_session();
    }
}

impl FrameEncoder for VideoToolboxEncoder {
    fn encode(
        &mut self,
        frame: &CapturedFrame,
        params: &EncodeParams,
    ) -> Result<EncodedFrame, EncodeError> {
        match frame {
            CapturedFrame::MetalPixelBuffer {
                pixel_buffer,
                width,
                height,
                pixel_format,
                ..
            } => self.encode_pixel_buffer(&*pixel_buffer, *width, *height, *pixel_format, params),
            CapturedFrame::CpuBuffer {
                data,
                width,
                height,
                stride,
                format,
            } => self.encode_cpu_buffer(data, *width, *height, *stride, *format, params),
        }
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, EncodeError> {
        if let Some(session) = &self.session {
            let status = unsafe { session.as_ref().complete_frames(kCMTimeInvalid) };
            if status != 0 {
                return Err(EncodeError::EncodeFailed(format!(
                    "VTCompressionSessionCompleteFrames(flush) failed with status {status}"
                )));
            }
        }

        Ok(Vec::new())
    }

    fn decoder_config(&self) -> Option<DecoderConfig> {
        self.decoder_config.clone()
    }
}

fn profile_for_pixel_format(pixel_format: u32) -> Result<HevcProfile, EncodeError> {
    if pixel_format == kCVPixelFormatType_32BGRA {
        Ok(HevcProfile::Main)
    } else if pixel_format == kCVPixelFormatType_64RGBAHalf {
        Ok(HevcProfile::Main10)
    } else {
        Err(EncodeError::UnsupportedConfig(format!(
            "unsupported VideoToolbox pixel format 0x{pixel_format:08x}"
        )))
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

fn copy_cpu_frame_into_pixel_buffer(
    pixel_buffer: &CVPixelBuffer,
    data: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    format: wgpu::TextureFormat,
) -> Result<(), EncodeError> {
    let row_bytes = row_bytes_for_texture(format, width)?;
    let source_stride = stride as usize;
    if source_stride < row_bytes {
        return Err(EncodeError::EncodeFailed(format!(
            "source stride {source_stride} is smaller than required row width {row_bytes}"
        )));
    }

    let required_len = source_stride
        .checked_mul(height as usize)
        .ok_or_else(|| EncodeError::EncodeFailed("CPU frame size overflow".into()))?;
    if data.len() < required_len {
        return Err(EncodeError::EncodeFailed(format!(
            "CPU frame buffer too small: {} bytes, expected at least {required_len}",
            data.len()
        )));
    }

    let base_address = NonNull::new(CVPixelBufferGetBaseAddress(pixel_buffer))
        .ok_or_else(|| EncodeError::EncodeFailed("CVPixelBuffer base address was null".into()))?;
    let destination_stride = CVPixelBufferGetBytesPerRow(pixel_buffer);
    if destination_stride < row_bytes {
        return Err(EncodeError::EncodeFailed(format!(
            "pixel buffer stride {destination_stride} is smaller than row width {row_bytes}"
        )));
    }

    for row in 0..height as usize {
        let source_offset = row * source_stride;
        let destination_offset = row * destination_stride;
        let source = &data[source_offset..source_offset + row_bytes];
        let destination = unsafe { (base_address.as_ptr() as *mut u8).add(destination_offset) };
        unsafe {
            ptr::copy_nonoverlapping(source.as_ptr(), destination, row_bytes);
        }
    }

    Ok(())
}

fn row_bytes_for_texture(format: wgpu::TextureFormat, width: u32) -> Result<usize, EncodeError> {
    let bytes_per_pixel = format
        .block_copy_size(None)
        .ok_or(EncodeError::UnsupportedConfig(format!(
            "unsupported texture format {format:?}"
        )))?;
    (width as usize)
        .checked_mul(bytes_per_pixel as usize)
        .ok_or_else(|| EncodeError::EncodeFailed("row byte size overflow".into()))
}

fn profile_level(profile: HevcProfile) -> &'static CFString {
    unsafe {
        match profile {
            HevcProfile::Main => kVTProfileLevel_HEVC_Main_AutoLevel,
            HevcProfile::Main10 => kVTProfileLevel_HEVC_Main10_AutoLevel,
        }
    }
}

fn hardware_encoder_specification() -> CFRetained<CFDictionary<CFString, CFType>> {
    CFDictionary::from_slices(
        &[unsafe { kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder }],
        &[CFBoolean::new(true).as_ref()],
    )
}

fn source_image_buffer_attributes(spec: SessionSpec) -> CFRetained<CFDictionary<CFString, CFType>> {
    let pixel_format = CFNumber::new_i32(spec.pixel_format as i32);
    let width = CFNumber::new_i32(spec.width.min(i32::MAX as u32) as i32);
    let height = CFNumber::new_i32(spec.height.min(i32::MAX as u32) as i32);

    CFDictionary::from_slices(
        &[
            unsafe { kCVPixelBufferPixelFormatTypeKey },
            unsafe { kCVPixelBufferWidthKey },
            unsafe { kCVPixelBufferHeightKey },
        ],
        &[pixel_format.as_ref(), width.as_ref(), height.as_ref()],
    )
}

fn force_keyframe_dictionary() -> CFRetained<CFDictionary<CFString, CFType>> {
    CFDictionary::from_slices(
        &[unsafe { kVTEncodeFrameOptionKey_ForceKeyFrame }],
        &[CFBoolean::new(true).as_ref()],
    )
}

fn set_property<V>(
    session: &VTCompressionSession,
    key: &CFString,
    value: &V,
) -> Result<(), EncodeError>
where
    V: AsRef<CFType> + ?Sized,
{
    let session_ref: &CFType = AsRef::<CFType>::as_ref(session);
    let value_ref: &CFType = value.as_ref();
    let status = unsafe { VTSessionSetProperty(session_ref, key, Some(value_ref)) };
    if status == 0 {
        Ok(())
    } else {
        Err(EncodeError::InitFailed(format!(
            "VTSessionSetProperty failed with status {status}"
        )))
    }
}

fn wait_for_callback(
    rx: &Receiver<Result<CallbackResult, String>>,
    timeout: Duration,
) -> Result<CallbackResult, EncodeError> {
    match rx.recv_timeout(timeout) {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(error)) => Err(EncodeError::EncodeFailed(error)),
        Err(_) => Err(EncodeError::EncodeFailed(
            "timed out waiting for VideoToolbox output callback".into(),
        )),
    }
}

unsafe extern "C-unwind" fn videotoolbox_output_callback(
    output_callback_ref_con: *mut std::ffi::c_void,
    _source_frame_ref_con: *mut std::ffi::c_void,
    status: i32,
    _info_flags: objc2_video_toolbox::VTEncodeInfoFlags,
    sample_buffer: *mut CMSampleBuffer,
) {
    let Some(state_ptr) = NonNull::new(output_callback_ref_con.cast::<CallbackState>()) else {
        return;
    };
    let state = unsafe { state_ptr.as_ref() };
    let pending = match state.pending.lock() {
        Ok(mut pending) => pending.take(),
        Err(_) => None,
    };
    let Some(pending) = pending else {
        return;
    };

    if status != 0 {
        let _ = pending.sender.send(Err(format!(
            "VideoToolbox callback returned status {status}"
        )));
        return;
    }

    let Some(sample_ptr) = NonNull::new(sample_buffer) else {
        let _ = pending
            .sender
            .send(Err("VideoToolbox did not return a CMSampleBuffer".into()));
        return;
    };
    let sample = unsafe { sample_ptr.as_ref() };

    let result = sample_buffer_to_result(
        sample,
        pending.started_at,
        pending.fallback_keyframe,
        pending.fallback_refine,
        pending.fallback_lossless,
    );
    let _ = pending.sender.send(result);
}

fn sample_buffer_to_result(
    sample: &CMSampleBuffer,
    started_at: Instant,
    fallback_keyframe: bool,
    fallback_refine: bool,
    fallback_lossless: bool,
) -> Result<CallbackResult, String> {
    let data_buffer = unsafe { sample.data_buffer() }
        .ok_or_else(|| "VideoToolbox sample buffer had no CMBlockBuffer".to_string())?;
    let data = block_buffer_bytes(&data_buffer)?;
    let format_description = unsafe { sample.format_description() }
        .ok_or_else(|| "VideoToolbox sample buffer had no format description".to_string())?;
    let is_keyframe = sample_is_keyframe(sample).unwrap_or(fallback_keyframe);
    let decoder_config = extract_decoder_config(&format_description)?;

    Ok(CallbackResult {
        frame: EncodedFrame {
            data,
            is_keyframe,
            is_refine: fallback_refine,
            is_lossless: fallback_lossless,
            encode_time_us: started_at.elapsed().as_micros().min(u64::MAX as u128) as u64,
        },
        decoder_config,
    })
}

fn block_buffer_bytes(block: &CMBlockBuffer) -> Result<Vec<u8>, String> {
    let length = unsafe { block.data_length() };
    let mut data = vec![0u8; length];
    if length == 0 {
        return Ok(data);
    }

    let status = unsafe {
        block.copy_data_bytes(0, length, NonNull::new(data.as_mut_ptr().cast()).unwrap())
    };
    if status == 0 {
        Ok(data)
    } else {
        Err(format!(
            "CMBlockBufferCopyDataBytes failed with status {status}"
        ))
    }
}

fn sample_is_keyframe(sample: &CMSampleBuffer) -> Option<bool> {
    let attachments = unsafe { sample.sample_attachments_array(false) }?;
    let attachments: &CFArray<CFDictionary> = unsafe { (&*attachments).cast_unchecked() };
    let first = attachments.get(0)?;
    let first: &CFDictionary<CFString, CFType> = unsafe { (&*first).cast_unchecked() };
    let not_sync = first.get(unsafe { kCMSampleAttachmentKey_NotSync })?;
    let flag: CFRetained<CFBoolean> = not_sync.downcast().ok()?;
    Some(!(*flag).as_bool())
}

fn extract_decoder_config(
    format_description: &CMFormatDescription,
) -> Result<Option<DecoderConfig>, String> {
    let Some(description) = extract_hvcc_description(format_description)? else {
        return Ok(None);
    };

    let dimensions =
        unsafe { objc2_core_media::CMVideoFormatDescriptionGetDimensions(format_description) };
    let profile = if description_main10_hint(&description) {
        HevcProfile::Main10
    } else {
        HevcProfile::Main
    };

    Ok(Some(DecoderConfig {
        codec: match profile {
            HevcProfile::Main => DEFAULT_MAIN_CODEC.into(),
            HevcProfile::Main10 => DEFAULT_MAIN10_CODEC.into(),
        },
        description: Some(description),
        coded_width: dimensions.width.max(0) as u32,
        coded_height: dimensions.height.max(0) as u32,
    }))
}

fn extract_hvcc_description(
    format_description: &CMFormatDescription,
) -> Result<Option<Vec<u8>>, String> {
    let Some(extensions) = (unsafe { format_description.extensions() }) else {
        return Ok(None);
    };
    let extensions: &CFDictionary<CFString, CFType> = unsafe { (&*extensions).cast_unchecked() };
    let Some(sample_atoms) =
        extensions.get(unsafe { kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms })
    else {
        return Ok(None);
    };
    let sample_atoms: CFRetained<CFDictionary> = sample_atoms
        .downcast()
        .map_err(|_| "sample description atoms were not a CFDictionary".to_string())?;
    let sample_atoms: &CFDictionary<CFString, CFType> =
        unsafe { (&*sample_atoms).cast_unchecked() };
    let hvcc_key = CFString::from_str("hvcC");
    let Some(hvcc) = sample_atoms.get(&hvcc_key) else {
        return Ok(None);
    };
    let hvcc: CFRetained<CFData> = hvcc
        .downcast()
        .map_err(|_| "hvcC atom payload was not CFData".to_string())?;

    let length = hvcc.length().max(0) as usize;
    let bytes = unsafe { std::slice::from_raw_parts(hvcc.byte_ptr(), length) };
    Ok(Some(bytes.to_vec()))
}

fn description_main10_hint(description: &[u8]) -> bool {
    description.get(1).is_some_and(|byte| (*byte & 0x1f) == 2)
}

#[cfg(test)]
mod tests {
    use super::{HevcProfile, description_main10_hint, profile_for_pixel_format};
    use objc2_core_video::{kCVPixelFormatType_32BGRA, kCVPixelFormatType_64RGBAHalf};

    #[test]
    fn maps_bgra_to_main_profile() {
        assert_eq!(
            profile_for_pixel_format(kCVPixelFormatType_32BGRA).unwrap(),
            HevcProfile::Main
        );
    }

    #[test]
    fn maps_half_float_to_main10_profile() {
        assert_eq!(
            profile_for_pixel_format(kCVPixelFormatType_64RGBAHalf).unwrap(),
            HevcProfile::Main10
        );
    }

    #[test]
    fn detects_main10_from_hvcc_profile_bits() {
        assert!(!description_main10_hint(&[1, 1]));
        assert!(description_main10_hint(&[1, 2]));
    }
}
