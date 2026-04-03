#![allow(unsafe_op_in_unsafe_fn)]

use std::{
    borrow::Cow,
    ffi::c_void,
    ptr,
    sync::OnceLock,
    thread,
    time::{Duration, Instant},
};

use libloading::Library;
use ustreamer_capture::CapturedFrame;
use ustreamer_proto::quality::{EncodeMode, EncodeParams};

use crate::{
    DecoderConfig, EncodeError, EncodedFrame, FrameEncoder,
    hevc::{decoder_config_from_hevc_access_unit, normalize_hevc_access_unit},
};

const SAMPLE_TIMEOUT: Duration = Duration::from_millis(250);
const POLL_INTERVAL: Duration = Duration::from_millis(1);
const AMF_PROBE_DIMENSIONS: &[(u32, u32)] = &[(128, 128), (1920, 1080)];

#[cfg(target_os = "windows")]
const AMF_RUNTIME_LIBRARY_CANDIDATES: &[&str] = &["amfrt64.dll"];
#[cfg(target_os = "linux")]
const AMF_RUNTIME_LIBRARY_CANDIDATES: &[&str] = &["libamfrt64.so.1", "libamfrt64.so"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmfCodec {
    Hevc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmfCapabilities {
    pub codec: AmfCodec,
    pub runtime_version: u64,
    pub runtime_library: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AmfDynamicSettings {
    target_fps: u32,
    target_bitrate_bps: u64,
    peak_bitrate_bps: u64,
}

impl AmfDynamicSettings {
    fn from_params(params: &EncodeParams) -> Self {
        let interactive_target = params.bitrate_bps.max(1);
        let refine_target = params.max_bitrate_bps.max(interactive_target);
        let target_bitrate_bps = if params.mode == EncodeMode::LosslessRefine {
            refine_target
        } else {
            interactive_target
        };
        let peak_bitrate_bps = params.max_bitrate_bps.max(target_bitrate_bps);
        Self {
            target_fps: params.target_fps.max(1),
            target_bitrate_bps,
            peak_bitrate_bps,
        }
    }
}

pub struct AmfEncoder {
    codec: AmfCodec,
    session: Option<AmfSession>,
    decoder_config: Option<DecoderConfig>,
    frame_index: u64,
}

impl AmfEncoder {
    pub fn new() -> Result<Self, EncodeError> {
        Self::with_codec(AmfCodec::Hevc)
    }

    pub fn with_codec(codec: AmfCodec) -> Result<Self, EncodeError> {
        let _ = amf_api()?;
        Ok(Self {
            codec,
            session: None,
            decoder_config: None,
            frame_index: 0,
        })
    }

    pub fn probe() -> Result<AmfCapabilities, EncodeError> {
        let api = amf_api()?;
        let mut last_error = None;
        for &(width, height) in AMF_PROBE_DIMENSIONS {
            match AmfSession::create(AmfCodec::Hevc, &EncodeParams::default(), width, height) {
                Ok(_session) => {
                    return Ok(AmfCapabilities {
                        codec: AmfCodec::Hevc,
                        runtime_version: api.runtime_version,
                        runtime_library: api.runtime_library,
                    });
                }
                Err(error) => {
                    last_error = Some((width, height, error));
                }
            }
        }

        let Some((width, height, error)) = last_error else {
            return Err(EncodeError::InitFailed(
                "AMF probe did not attempt any encoder dimensions".into(),
            ));
        };

        return Err(EncodeError::InitFailed(format!(
            "AMF runtime loaded from {} but probe session creation failed at {}x{}: {}",
            api.runtime_library, width, height, error
        )));
    }

    fn ensure_session(
        &mut self,
        params: &EncodeParams,
        width: u32,
        height: u32,
    ) -> Result<&mut AmfSession, EncodeError> {
        let recreate = self
            .session
            .as_ref()
            .map(|session| session.dimensions != (width, height))
            .unwrap_or(true);
        if recreate {
            self.session = Some(AmfSession::create(self.codec, params, width, height)?);
            self.decoder_config = None;
            self.frame_index = 0;
        }
        self.session.as_mut().ok_or_else(|| {
            EncodeError::InitFailed("AMF encoder session was not initialized".into())
        })
    }
}

impl FrameEncoder for AmfEncoder {
    fn encode(
        &mut self,
        frame: &CapturedFrame,
        params: &EncodeParams,
    ) -> Result<EncodedFrame, EncodeError> {
        let encode_started_at = Instant::now();
        let (frame_bytes, width, height) = prepare_cpu_frame_bytes(frame)?;
        let frame_index = self.frame_index;
        let force_keyframe = frame_index == 0 || params.force_keyframe;
        let output = self.ensure_session(params, width, height)?.encode_frame(
            &frame_bytes,
            width,
            height,
            frame_index,
            params,
            force_keyframe,
        )?;

        if output.is_keyframe && self.decoder_config.is_none() {
            self.decoder_config = Some(
                decoder_config_from_hevc_access_unit(&output.data, width, height).map_err(
                    |error| {
                        EncodeError::EncodeFailed(format!(
                            "failed to derive HEVC decoder config from AMF keyframe: {error}"
                        ))
                    },
                )?,
            );
        }

        let normalized = normalize_hevc_access_unit(&output.data).map_err(|error| {
            EncodeError::EncodeFailed(format!(
                "failed to normalize AMF HEVC access unit for browser decode: {error}"
            ))
        })?;

        self.frame_index = self.frame_index.saturating_add(1);

        Ok(EncodedFrame {
            data: normalized,
            is_keyframe: output.is_keyframe,
            is_refine: params.mode == EncodeMode::LosslessRefine,
            is_lossless: false,
            encode_time_us: encode_started_at
                .elapsed()
                .as_micros()
                .min(u64::MAX as u128) as u64,
        })
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, EncodeError> {
        if let Some(session) = self.session.as_mut() {
            session.drain()?;
        }
        Ok(Vec::new())
    }

    fn decoder_config(&self) -> Option<DecoderConfig> {
        self.decoder_config.clone()
    }
}

#[derive(Debug)]
struct AmfEncodedOutput {
    data: Vec<u8>,
    is_keyframe: bool,
}

struct AmfSession {
    context: ContextHandle,
    component: ComponentHandle,
    dimensions: (u32, u32),
    dynamic_settings: Option<AmfDynamicSettings>,
}

impl AmfSession {
    fn create(
        codec: AmfCodec,
        params: &EncodeParams,
        width: u32,
        height: u32,
    ) -> Result<Self, EncodeError> {
        let context = create_context()?;
        initialize_context(context.as_ptr())?;
        let component = create_component(context.as_ptr(), codec)?;
        apply_static_encoder_properties(component.as_ptr(), width, height, params)?;
        initialize_component(component.as_ptr(), codec, width, height)?;
        let mut session = Self {
            context,
            component,
            dimensions: (width, height),
            dynamic_settings: None,
        };
        session.update_dynamic_properties(params)?;
        Ok(session)
    }

    fn encode_frame(
        &mut self,
        frame_bytes: &[u8],
        width: u32,
        height: u32,
        frame_index: u64,
        params: &EncodeParams,
        force_keyframe: bool,
    ) -> Result<AmfEncodedOutput, EncodeError> {
        self.update_dynamic_properties(params)?;

        let surface = alloc_host_surface(self.context.as_ptr(), width, height)?;
        copy_cpu_frame_into_surface(surface.as_ptr(), frame_bytes, width, height)?;
        set_surface_timing(surface.as_ptr(), frame_index, params.target_fps.max(1));
        if force_keyframe {
            set_surface_property_int64(
                surface.as_ptr(),
                HEVC_FORCE_PICTURE_TYPE_PROPERTY,
                sys::AMF_VIDEO_ENCODER_HEVC_PICTURE_TYPE_IDR as i64,
            )?;
            set_surface_property_bool(surface.as_ptr(), HEVC_INSERT_HEADER_PROPERTY, true)?;
        }

        submit_input(
            self.component.as_ptr(),
            surface.as_ptr() as *mut sys::AMFData,
        )?;
        self.wait_for_output(force_keyframe)
    }

    fn update_dynamic_properties(&mut self, params: &EncodeParams) -> Result<(), EncodeError> {
        let next = AmfDynamicSettings::from_params(params);
        if self.dynamic_settings == Some(next) {
            return Ok(());
        }
        set_component_property_rate(
            self.component.as_ptr(),
            HEVC_FRAMERATE_PROPERTY,
            sys::AMFRate {
                num: next.target_fps,
                den: 1,
            },
        )?;
        set_component_property_int64(
            self.component.as_ptr(),
            HEVC_TARGET_BITRATE_PROPERTY,
            next.target_bitrate_bps.min(i64::MAX as u64) as i64,
        )?;
        set_component_property_int64(
            self.component.as_ptr(),
            HEVC_PEAK_BITRATE_PROPERTY,
            next.peak_bitrate_bps.min(i64::MAX as u64) as i64,
        )?;
        self.dynamic_settings = Some(next);
        Ok(())
    }

    fn wait_for_output(&mut self, forced_keyframe: bool) -> Result<AmfEncodedOutput, EncodeError> {
        let deadline = Instant::now() + SAMPLE_TIMEOUT;
        loop {
            let mut data = ptr::null_mut();
            match query_output(self.component.as_ptr(), &mut data) {
                status if status == sys::AMF_OK => {
                    if data.is_null() {
                        if Instant::now() >= deadline {
                            return Err(EncodeError::EncodeFailed(
                                "AMF QueryOutput returned success without encoded data".into(),
                            ));
                        }
                        thread::sleep(POLL_INTERVAL);
                        continue;
                    }
                    let buffer = query_output_buffer(data)?;
                    let encoded = read_buffer_bytes(buffer.as_ptr())?;
                    let is_keyframe = buffer_output_data_type(buffer.as_ptr())
                        .map(|picture_type| {
                            picture_type == sys::AMF_VIDEO_ENCODER_HEVC_OUTPUT_DATA_TYPE_IDR as i64
                                || picture_type
                                    == sys::AMF_VIDEO_ENCODER_HEVC_OUTPUT_DATA_TYPE_I as i64
                        })
                        .unwrap_or(forced_keyframe);
                    return Ok(AmfEncodedOutput {
                        data: encoded,
                        is_keyframe,
                    });
                }
                status if status == sys::AMF_REPEAT || status == sys::AMF_NEED_MORE_INPUT => {
                    if Instant::now() >= deadline {
                        return Err(EncodeError::EncodeFailed(format!(
                            "timed out waiting for AMF encoded output after {}ms",
                            SAMPLE_TIMEOUT.as_millis()
                        )));
                    }
                    thread::sleep(POLL_INTERVAL);
                }
                status => {
                    return Err(EncodeError::EncodeFailed(format!(
                        "AMF QueryOutput failed with {}",
                        format_amf_status(status)
                    )));
                }
            }
        }
    }

    fn drain(&mut self) -> Result<(), EncodeError> {
        let status =
            unsafe { ((*(*self.component.as_ptr()).pVtbl).Drain)(self.component.as_ptr()) };
        if status != sys::AMF_OK && status != sys::AMF_EOF {
            return Err(EncodeError::EncodeFailed(format!(
                "AMF Drain failed with {}",
                format_amf_status(status)
            )));
        }
        Ok(())
    }

    fn shutdown(&mut self) {
        unsafe {
            if !self.component.as_ptr().is_null() {
                ((*(*self.component.as_ptr()).pVtbl).Terminate)(self.component.as_ptr());
            }
            if !self.context.as_ptr().is_null() {
                ((*(*self.context.as_ptr()).pVtbl).Terminate)(self.context.as_ptr());
            }
        }
    }
}

impl Drop for AmfSession {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn create_context() -> Result<ContextHandle, EncodeError> {
    let api = amf_api()?;
    let mut factory = ptr::null_mut();
    let status = unsafe { (api.init)(sys::AMF_FULL_VERSION, &mut factory) };
    if status != sys::AMF_OK {
        return Err(EncodeError::InitFailed(format!(
            "AMFInit failed with {} while loading {}",
            format_amf_status(status),
            api.runtime_library,
        )));
    }
    if factory.is_null() {
        return Err(EncodeError::InitFailed(
            "AMFInit returned a null factory pointer".into(),
        ));
    }

    let mut context = ptr::null_mut();
    let status = unsafe { ((*(*factory).pVtbl).CreateContext)(factory, &mut context) };
    if status != sys::AMF_OK {
        return Err(EncodeError::InitFailed(format!(
            "AMF CreateContext failed with {}",
            format_amf_status(status)
        )));
    }
    ContextHandle::from_raw(context)
}

fn initialize_context(context: *mut sys::AMFContext) -> Result<(), EncodeError> {
    #[cfg(target_os = "windows")]
    let status =
        unsafe { ((*(*context).pVtbl).InitDX11)(context, ptr::null_mut(), sys::AMF_DX11_0) };

    #[cfg(target_os = "linux")]
    let status = {
        let context1 = query_context1(context)?;
        let status = unsafe {
            ((*(*context1.as_ptr()).pVtbl).InitVulkan)(context1.as_ptr(), ptr::null_mut())
        };
        drop(context1);
        status
    };

    if status != sys::AMF_OK && status != sys::AMF_ALREADY_INITIALIZED {
        return Err(EncodeError::InitFailed(format!(
            "AMF context initialization failed with {}",
            format_amf_status(status)
        )));
    }
    Ok(())
}

fn create_component(
    context: *mut sys::AMFContext,
    codec: AmfCodec,
) -> Result<ComponentHandle, EncodeError> {
    let mut factory = ptr::null_mut();
    let status = unsafe { (amf_api()?.init)(sys::AMF_FULL_VERSION, &mut factory) };
    if status != sys::AMF_OK || factory.is_null() {
        return Err(EncodeError::InitFailed(format!(
            "AMFInit failed while creating encoder component: {}",
            format_amf_status(status)
        )));
    }

    let mut component = ptr::null_mut();
    let component_id = match codec {
        AmfCodec::Hevc => amf_wide(HEVC_COMPONENT_ID),
    };
    let status = unsafe {
        ((*(*factory).pVtbl).CreateComponent)(
            factory,
            context,
            component_id.as_ptr(),
            &mut component,
        )
    };
    if status != sys::AMF_OK {
        return Err(EncodeError::InitFailed(format!(
            "AMF CreateComponent({HEVC_COMPONENT_ID}) failed with {}",
            format_amf_status(status)
        )));
    }
    ComponentHandle::from_raw(component)
}

fn initialize_component(
    component: *mut sys::AMFComponent,
    codec: AmfCodec,
    width: u32,
    height: u32,
) -> Result<(), EncodeError> {
    let format = match codec {
        AmfCodec::Hevc => sys::AMF_SURFACE_BGRA,
    };
    let status = unsafe {
        ((*(*component).pVtbl).Init)(
            component,
            format,
            width.min(i32::MAX as u32) as i32,
            height.min(i32::MAX as u32) as i32,
        )
    };
    if status != sys::AMF_OK {
        return Err(EncodeError::InitFailed(format!(
            "AMF encoder Init failed with {}",
            format_amf_status(status)
        )));
    }
    Ok(())
}

fn apply_static_encoder_properties(
    component: *mut sys::AMFComponent,
    width: u32,
    height: u32,
    params: &EncodeParams,
) -> Result<(), EncodeError> {
    set_component_property_int64(
        component,
        HEVC_USAGE_PROPERTY,
        sys::AMF_VIDEO_ENCODER_HEVC_USAGE_LOW_LATENCY_HIGH_QUALITY as i64,
    )?;
    set_component_property_int64(
        component,
        HEVC_RATE_CONTROL_METHOD_PROPERTY,
        sys::AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR as i64,
    )?;
    set_component_property_int64(
        component,
        HEVC_OUTPUT_MODE_PROPERTY,
        sys::AMF_VIDEO_ENCODER_HEVC_OUTPUT_MODE_FRAME as i64,
    )?;
    set_component_property_int64(
        component,
        HEVC_HEADER_INSERTION_MODE_PROPERTY,
        sys::AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE_IDR_ALIGNED as i64,
    )?;
    set_component_property_int64(
        component,
        HEVC_QUALITY_PRESET_PROPERTY,
        sys::AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_HIGH_QUALITY as i64,
    )?;
    set_component_property_int64(
        component,
        HEVC_GOP_SIZE_PROPERTY,
        params.target_fps.max(1).min(i64::MAX as u32) as i64,
    )?;
    set_component_property_int64(component, HEVC_NUM_GOPS_PER_IDR_PROPERTY, 1)?;
    set_component_property_int64(component, HEVC_SLICES_PER_FRAME_PROPERTY, 1)?;
    set_component_property_int64(component, HEVC_INPUT_QUEUE_SIZE_PROPERTY, 1)?;
    set_component_property_int64(component, HEVC_QUERY_TIMEOUT_PROPERTY, 0)?;
    set_component_property_bool(component, HEVC_LOW_LATENCY_MODE_PROPERTY, true)?;
    set_component_property_bool(component, HEVC_NOMINAL_RANGE_PROPERTY, true)?;
    set_component_property_size(
        component,
        HEVC_FRAMESIZE_PROPERTY,
        sys::AMFSize {
            width: width.min(i32::MAX as u32) as i32,
            height: height.min(i32::MAX as u32) as i32,
        },
    )?;
    Ok(())
}

fn alloc_host_surface(
    context: *mut sys::AMFContext,
    width: u32,
    height: u32,
) -> Result<SurfaceHandle, EncodeError> {
    let mut surface = ptr::null_mut();
    let status = unsafe {
        ((*(*context).pVtbl).AllocSurface)(
            context,
            sys::AMF_MEMORY_HOST,
            sys::AMF_SURFACE_BGRA,
            width.min(i32::MAX as u32) as i32,
            height.min(i32::MAX as u32) as i32,
            &mut surface,
        )
    };
    if status != sys::AMF_OK {
        return Err(EncodeError::EncodeFailed(format!(
            "AMF AllocSurface(BGRA host) failed with {}",
            format_amf_status(status)
        )));
    }
    SurfaceHandle::from_raw(surface)
}

fn copy_cpu_frame_into_surface(
    surface: *mut sys::AMFSurface,
    frame_bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<(), EncodeError> {
    let plane = unsafe { ((*(*surface).pVtbl).GetPlane)(surface, sys::AMF_PLANE_PACKED) };
    let plane = PlaneHandle::from_raw(plane)?;
    let destination = unsafe { ((*(*plane.as_ptr()).pVtbl).GetNative)(plane.as_ptr()) } as *mut u8;
    if destination.is_null() {
        return Err(EncodeError::EncodeFailed(
            "AMF packed BGRA plane exposed a null native pointer".into(),
        ));
    }

    let destination_pitch = unsafe { ((*(*plane.as_ptr()).pVtbl).GetHPitch)(plane.as_ptr()) };
    if destination_pitch <= 0 {
        return Err(EncodeError::EncodeFailed(format!(
            "AMF packed BGRA plane reported invalid pitch {destination_pitch}"
        )));
    }
    let destination_pitch = destination_pitch as usize;
    let source_row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| EncodeError::EncodeFailed("AMF source row-bytes overflow".into()))?
        as usize;
    if destination_pitch < source_row_bytes {
        return Err(EncodeError::EncodeFailed(format!(
            "AMF destination pitch {destination_pitch} is smaller than source row width {source_row_bytes}"
        )));
    }

    for row in 0..height as usize {
        let source_offset = row * source_row_bytes;
        let destination_offset = row * destination_pitch;
        unsafe {
            ptr::copy_nonoverlapping(
                frame_bytes.as_ptr().add(source_offset),
                destination.add(destination_offset),
                source_row_bytes,
            );
        }
    }
    Ok(())
}

fn set_surface_timing(surface: *mut sys::AMFSurface, frame_index: u64, target_fps: u32) {
    let fps = u64::from(target_fps.max(1));
    let pts = frame_index.saturating_mul(sys::AMF_SECOND) / fps;
    let duration = (sys::AMF_SECOND / fps).max(1);
    unsafe {
        ((*(*surface).pVtbl).SetPts)(surface, pts as i64);
        ((*(*surface).pVtbl).SetDuration)(surface, duration as i64);
    }
}

fn submit_input(
    component: *mut sys::AMFComponent,
    data: *mut sys::AMFData,
) -> Result<(), EncodeError> {
    let status = unsafe { ((*(*component).pVtbl).SubmitInput)(component, data) };
    if status == sys::AMF_OK {
        return Ok(());
    }
    Err(EncodeError::EncodeFailed(format!(
        "AMF SubmitInput failed with {}",
        format_amf_status(status)
    )))
}

fn query_output(
    component: *mut sys::AMFComponent,
    data: *mut *mut sys::AMFData,
) -> sys::AMF_RESULT {
    unsafe { ((*(*component).pVtbl).QueryOutput)(component, data) }
}

fn query_output_buffer(data: *mut sys::AMFData) -> Result<BufferHandle, EncodeError> {
    let data = DataHandle::from_raw(data)?;
    let data_type = unsafe { ((*(*data.as_ptr()).pVtbl).GetDataType)(data.as_ptr()) };
    if data_type != sys::AMF_DATA_BUFFER {
        return Err(EncodeError::EncodeFailed(format!(
            "AMF QueryOutput returned unexpected data type {data_type} instead of AMF_DATA_BUFFER"
        )));
    }

    let mut interface: *mut c_void = ptr::null_mut();
    let status = unsafe {
        ((*(*data.as_ptr()).pVtbl).QueryInterface)(
            data.as_ptr(),
            &sys::AMF_BUFFER_IID,
            &mut interface,
        )
    };
    if status != sys::AMF_OK {
        return Err(EncodeError::EncodeFailed(format!(
            "AMFData::QueryInterface(AMFBuffer) failed with {}",
            format_amf_status(status)
        )));
    }
    BufferHandle::from_raw(interface.cast())
}

fn read_buffer_bytes(buffer: *mut sys::AMFBuffer) -> Result<Vec<u8>, EncodeError> {
    let native = unsafe { ((*(*buffer).pVtbl).GetNative)(buffer) };
    let size = unsafe { ((*(*buffer).pVtbl).GetSize)(buffer) };
    if native.is_null() {
        return Err(EncodeError::EncodeFailed(
            "AMF output buffer exposed a null native pointer".into(),
        ));
    }
    let size = size as usize;
    Ok(unsafe { std::slice::from_raw_parts(native as *const u8, size) }.to_vec())
}

fn buffer_output_data_type(buffer: *mut sys::AMFBuffer) -> Option<i64> {
    get_buffer_property_int64(buffer, HEVC_OUTPUT_DATA_TYPE_PROPERTY).ok()
}

fn set_component_property_int64(
    component: *mut sys::AMFComponent,
    name: &str,
    value: i64,
) -> Result<(), EncodeError> {
    let property_name = amf_wide(name);
    let variant = amf_variant_int64(value);
    let status =
        unsafe { ((*(*component).pVtbl).SetProperty)(component, property_name.as_ptr(), variant) };
    if status != sys::AMF_OK {
        return Err(EncodeError::InitFailed(format!(
            "AMF SetProperty({name}) failed with {}",
            format_amf_status(status)
        )));
    }
    Ok(())
}

fn set_component_property_bool(
    component: *mut sys::AMFComponent,
    name: &str,
    value: bool,
) -> Result<(), EncodeError> {
    let property_name = amf_wide(name);
    let variant = amf_variant_bool(value);
    let status =
        unsafe { ((*(*component).pVtbl).SetProperty)(component, property_name.as_ptr(), variant) };
    if status != sys::AMF_OK {
        return Err(EncodeError::InitFailed(format!(
            "AMF SetProperty({name}) failed with {}",
            format_amf_status(status)
        )));
    }
    Ok(())
}

fn set_component_property_size(
    component: *mut sys::AMFComponent,
    name: &str,
    value: sys::AMFSize,
) -> Result<(), EncodeError> {
    let property_name = amf_wide(name);
    let variant = amf_variant_size(value);
    let status =
        unsafe { ((*(*component).pVtbl).SetProperty)(component, property_name.as_ptr(), variant) };
    if status != sys::AMF_OK {
        return Err(EncodeError::InitFailed(format!(
            "AMF SetProperty({name}={}x{}) failed with {}",
            value.width,
            value.height,
            format_amf_status(status),
        )));
    }
    Ok(())
}

fn set_component_property_rate(
    component: *mut sys::AMFComponent,
    name: &str,
    value: sys::AMFRate,
) -> Result<(), EncodeError> {
    let property_name = amf_wide(name);
    let variant = amf_variant_rate(value);
    let status =
        unsafe { ((*(*component).pVtbl).SetProperty)(component, property_name.as_ptr(), variant) };
    if status != sys::AMF_OK {
        return Err(EncodeError::InitFailed(format!(
            "AMF SetProperty({name}) failed with {}",
            format_amf_status(status)
        )));
    }
    Ok(())
}

fn set_surface_property_int64(
    surface: *mut sys::AMFSurface,
    name: &str,
    value: i64,
) -> Result<(), EncodeError> {
    let property_name = amf_wide(name);
    let variant = amf_variant_int64(value);
    let status =
        unsafe { ((*(*surface).pVtbl).SetProperty)(surface, property_name.as_ptr(), variant) };
    if status != sys::AMF_OK {
        return Err(EncodeError::EncodeFailed(format!(
            "AMF surface SetProperty({name}) failed with {}",
            format_amf_status(status)
        )));
    }
    Ok(())
}

fn set_surface_property_bool(
    surface: *mut sys::AMFSurface,
    name: &str,
    value: bool,
) -> Result<(), EncodeError> {
    let property_name = amf_wide(name);
    let variant = amf_variant_bool(value);
    let status =
        unsafe { ((*(*surface).pVtbl).SetProperty)(surface, property_name.as_ptr(), variant) };
    if status != sys::AMF_OK {
        return Err(EncodeError::EncodeFailed(format!(
            "AMF surface SetProperty({name}) failed with {}",
            format_amf_status(status)
        )));
    }
    Ok(())
}

fn get_buffer_property_int64(buffer: *mut sys::AMFBuffer, name: &str) -> Result<i64, EncodeError> {
    let property_name = amf_wide(name);
    let mut value = sys::AMFVariantStruct::default();
    let status =
        unsafe { ((*(*buffer).pVtbl).GetProperty)(buffer, property_name.as_ptr(), &mut value) };
    if status != sys::AMF_OK {
        return Err(EncodeError::EncodeFailed(format!(
            "AMF buffer GetProperty({name}) failed with {}",
            format_amf_status(status)
        )));
    }
    if value.type_ != sys::AMF_VARIANT_INT64 {
        return Err(EncodeError::EncodeFailed(format!(
            "AMF buffer property {name} returned unexpected variant type {}",
            value.type_
        )));
    }
    Ok(unsafe { value.value.int64_value })
}

fn amf_variant_bool(value: bool) -> sys::AMFVariantStruct {
    sys::AMFVariantStruct {
        type_: sys::AMF_VARIANT_BOOL,
        value: sys::AMFVariantValue {
            bool_value: u8::from(value),
        },
    }
}

fn amf_variant_int64(value: i64) -> sys::AMFVariantStruct {
    sys::AMFVariantStruct {
        type_: sys::AMF_VARIANT_INT64,
        value: sys::AMFVariantValue { int64_value: value },
    }
}

fn amf_variant_size(value: sys::AMFSize) -> sys::AMFVariantStruct {
    sys::AMFVariantStruct {
        type_: sys::AMF_VARIANT_SIZE,
        value: sys::AMFVariantValue { size_value: value },
    }
}

fn amf_variant_rate(value: sys::AMFRate) -> sys::AMFVariantStruct {
    sys::AMFVariantStruct {
        type_: sys::AMF_VARIANT_RATE,
        value: sys::AMFVariantValue { rate_value: value },
    }
}

#[cfg(target_os = "windows")]
type AmfWideChar = u16;
#[cfg(target_os = "linux")]
type AmfWideChar = u32;

fn amf_wide(value: &str) -> Vec<AmfWideChar> {
    #[cfg(target_os = "windows")]
    {
        value.encode_utf16().chain(std::iter::once(0)).collect()
    }
    #[cfg(target_os = "linux")]
    {
        value
            .chars()
            .map(|ch| ch as u32)
            .chain(std::iter::once(0))
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct AmfApi {
    init: sys::AMFInit_Fn,
    runtime_version: u64,
    runtime_library: &'static str,
}

unsafe impl Send for AmfApi {}
unsafe impl Sync for AmfApi {}

fn amf_api() -> Result<&'static AmfApi, EncodeError> {
    static API: OnceLock<Result<AmfApi, String>> = OnceLock::new();
    match API.get_or_init(load_amf_api) {
        Ok(api) => Ok(api),
        Err(error) => Err(EncodeError::InitFailed(error.clone())),
    }
}

fn load_amf_api() -> Result<AmfApi, String> {
    let mut errors = Vec::new();
    for candidate in AMF_RUNTIME_LIBRARY_CANDIDATES {
        match unsafe { Library::new(candidate) } {
            Ok(library) => match unsafe { load_amf_api_from_library(candidate, library) } {
                Ok(api) => return Ok(api),
                Err(error) => errors.push(format!("{candidate}: {error}")),
            },
            Err(error) => errors.push(format!("{candidate}: {error}")),
        }
    }

    Err(format!(
        "failed to load the AMD AMF runtime library (tried {}): {}",
        AMF_RUNTIME_LIBRARY_CANDIDATES.join(", "),
        errors.join("; ")
    ))
}

unsafe fn load_amf_api_from_library(
    runtime_library: &'static str,
    library: Library,
) -> Result<AmfApi, String> {
    let init = *library
        .get::<sys::AMFInit_Fn>(b"AMFInit\0")
        .map_err(|error| format!("missing AMFInit export: {error}"))?;
    let query_version = *library
        .get::<sys::AMFQueryVersion_Fn>(b"AMFQueryVersion\0")
        .map_err(|error| format!("missing AMFQueryVersion export: {error}"))?;

    let mut runtime_version = 0;
    let status = query_version(&mut runtime_version);
    if status != sys::AMF_OK {
        return Err(format!(
            "AMFQueryVersion failed with {}",
            format_amf_status(status)
        ));
    }

    std::mem::forget(library);
    Ok(AmfApi {
        init,
        runtime_version,
        runtime_library,
    })
}

#[cfg(target_os = "linux")]
fn query_context1(context: *mut sys::AMFContext) -> Result<Context1Handle, EncodeError> {
    let mut interface: *mut c_void = ptr::null_mut();
    let status = unsafe {
        ((*(*context).pVtbl).QueryInterface)(
            context,
            &sys::AMF_CONTEXT1_IID,
            &mut interface as *mut _ as *mut *mut c_void,
        )
    };
    if status != sys::AMF_OK {
        return Err(EncodeError::InitFailed(format!(
            "AMFContext::QueryInterface(AMFContext1) failed with {}",
            format_amf_status(status)
        )));
    }
    Context1Handle::from_raw(interface.cast())
}

#[allow(irrefutable_let_patterns)]
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
            "direct AMF encoding currently requires `CapturedFrame::CpuBuffer` input".into(),
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
                "direct AMF encoding currently supports only BGRA8 CPU frames; got {other:?}"
            )));
        }
    }

    if stride < row_bytes {
        return Err(EncodeError::UnsupportedFrame(format!(
            "AMF encoder received stride {stride} smaller than BGRA row width {row_bytes}"
        )));
    }

    let required_len = stride
        .checked_mul(*height as usize)
        .ok_or_else(|| EncodeError::UnsupportedFrame("BGRA frame size overflow".into()))?;
    if data.len() < required_len {
        return Err(EncodeError::UnsupportedFrame(format!(
            "AMF encoder received CPU buffer with length {} but expected at least {required_len}",
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

fn format_amf_status(status: sys::AMF_RESULT) -> String {
    let label = match status {
        sys::AMF_OK => "AMF_OK",
        sys::AMF_FAIL => "AMF_FAIL",
        sys::AMF_UNEXPECTED => "AMF_UNEXPECTED",
        sys::AMF_ACCESS_DENIED => "AMF_ACCESS_DENIED",
        sys::AMF_INVALID_ARG => "AMF_INVALID_ARG",
        sys::AMF_OUT_OF_RANGE => "AMF_OUT_OF_RANGE",
        sys::AMF_OUT_OF_MEMORY => "AMF_OUT_OF_MEMORY",
        sys::AMF_INVALID_POINTER => "AMF_INVALID_POINTER",
        sys::AMF_NO_INTERFACE => "AMF_NO_INTERFACE",
        sys::AMF_NOT_IMPLEMENTED => "AMF_NOT_IMPLEMENTED",
        sys::AMF_ALREADY_INITIALIZED => "AMF_ALREADY_INITIALIZED",
        sys::AMF_NOT_INITIALIZED => "AMF_NOT_INITIALIZED",
        sys::AMF_INVALID_FORMAT => "AMF_INVALID_FORMAT",
        sys::AMF_WRONG_STATE => "AMF_WRONG_STATE",
        sys::AMF_FILE_NOT_OPEN => "AMF_FILE_NOT_OPEN",
        sys::AMF_NO_DEVICE => "AMF_NO_DEVICE",
        sys::AMF_INPUT_FULL => "AMF_INPUT_FULL",
        sys::AMF_REPEAT => "AMF_REPEAT",
        sys::AMF_NEED_MORE_INPUT => "AMF_NEED_MORE_INPUT",
        sys::AMF_EOF => "AMF_EOF",
        sys::AMF_NOT_SUPPORTED => "AMF_NOT_SUPPORTED",
        sys::AMF_NOT_FOUND => "AMF_NOT_FOUND",
        sys::AMF_INVALID_DATA_TYPE => "AMF_INVALID_DATA_TYPE",
        sys::AMF_INVALID_RESOLUTION => "AMF_INVALID_RESOLUTION",
        sys::AMF_CODEC_NOT_SUPPORTED => "AMF_CODEC_NOT_SUPPORTED",
        sys::AMF_SURFACE_FORMAT_NOT_SUPPORTED => "AMF_SURFACE_FORMAT_NOT_SUPPORTED",
        sys::AMF_SURFACE_MUST_BE_SHARED => "AMF_SURFACE_MUST_BE_SHARED",
        sys::AMF_DECODER_NOT_PRESENT => "AMF_DECODER_NOT_PRESENT",
        sys::AMF_DECODER_NO_FREE_SURFACES => "AMF_DECODER_NO_FREE_SURFACES",
        sys::AMF_ENCODER_NOT_PRESENT => "AMF_ENCODER_NOT_PRESENT",
        sys::AMF_VULKAN_FAILED => "AMF_VULKAN_FAILED",
        sys::AMF_DIRECTX_FAILED => "AMF_DIRECTX_FAILED",
        other => return format!("AMF status {other}"),
    };
    format!("{label} ({status})")
}

macro_rules! com_handle {
    ($name:ident, $ty:ty) => {
        struct $name(*mut $ty);

        impl $name {
            fn from_raw(ptr: *mut $ty) -> Result<Self, EncodeError> {
                if ptr.is_null() {
                    return Err(EncodeError::InitFailed(format!(
                        "{} returned a null pointer",
                        stringify!($name)
                    )));
                }
                Ok(Self(ptr))
            }

            fn as_ptr(&self) -> *mut $ty {
                self.0
            }
        }

        unsafe impl Send for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    if !self.0.is_null() {
                        ((*(*self.0).pVtbl).Release)(self.0);
                    }
                }
            }
        }
    };
}

com_handle!(ContextHandle, sys::AMFContext);
#[cfg(target_os = "linux")]
com_handle!(Context1Handle, sys::AMFContext1);
com_handle!(ComponentHandle, sys::AMFComponent);
com_handle!(DataHandle, sys::AMFData);
com_handle!(SurfaceHandle, sys::AMFSurface);
com_handle!(BufferHandle, sys::AMFBuffer);
com_handle!(PlaneHandle, sys::AMFPlane);

const HEVC_COMPONENT_ID: &str = "AMFVideoEncoderHW_HEVC";
const HEVC_FRAMESIZE_PROPERTY: &str = "HevcFrameSize";
const HEVC_USAGE_PROPERTY: &str = "HevcUsage";
const HEVC_QUALITY_PRESET_PROPERTY: &str = "HevcQualityPreset";
const HEVC_LOW_LATENCY_MODE_PROPERTY: &str = "LowLatencyInternal";
const HEVC_NOMINAL_RANGE_PROPERTY: &str = "HevcNominalRange";
const HEVC_GOP_SIZE_PROPERTY: &str = "HevcGOPSize";
const HEVC_NUM_GOPS_PER_IDR_PROPERTY: &str = "HevcGOPSPerIDR";
const HEVC_SLICES_PER_FRAME_PROPERTY: &str = "HevcSlicesPerFrame";
const HEVC_HEADER_INSERTION_MODE_PROPERTY: &str = "HevcHeaderInsertionMode";
const HEVC_RATE_CONTROL_METHOD_PROPERTY: &str = "HevcRateControlMethod";
const HEVC_FRAMERATE_PROPERTY: &str = "HevcFrameRate";
const HEVC_TARGET_BITRATE_PROPERTY: &str = "HevcTargetBitrate";
const HEVC_PEAK_BITRATE_PROPERTY: &str = "HevcPeakBitrate";
const HEVC_QUERY_TIMEOUT_PROPERTY: &str = "HevcQueryTimeout";
const HEVC_INPUT_QUEUE_SIZE_PROPERTY: &str = "HevcInputQueueSize";
const HEVC_FORCE_PICTURE_TYPE_PROPERTY: &str = "HevcForcePictureType";
const HEVC_INSERT_HEADER_PROPERTY: &str = "HevcInsertHeader";
const HEVC_OUTPUT_MODE_PROPERTY: &str = "HevcOutputMode";
const HEVC_OUTPUT_DATA_TYPE_PROPERTY: &str = "HevcOutputDataType";

#[cfg(test)]
mod tests {
    use super::{amf_wide, prepare_cpu_frame_bytes};
    use ustreamer_capture::CapturedFrame;

    #[test]
    fn amf_wide_strings_are_nul_terminated() {
        let wide = amf_wide("HevcUsage");
        assert_eq!(wide.last().copied(), Some(0));
    }

    #[test]
    fn prepare_cpu_frame_packs_strided_bgra_rows() {
        let frame = CapturedFrame::CpuBuffer {
            data: vec![1, 2, 3, 4, 9, 9, 9, 9, 5, 6, 7, 8, 8, 8, 8, 8],
            width: 1,
            height: 2,
            stride: 8,
            format: wgpu::TextureFormat::Bgra8Unorm,
        };

        let (packed, width, height) = prepare_cpu_frame_bytes(&frame).unwrap();
        assert_eq!((width, height), (1, 2));
        assert_eq!(packed.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }
}

#[allow(non_camel_case_types, non_snake_case, dead_code)]
mod sys {
    use super::{AmfWideChar, c_void};

    pub type AMF_RESULT = i32;
    pub type amf_int64 = i64;
    pub type amf_int32 = i32;
    pub type amf_uint64 = u64;
    pub type amf_uint32 = u32;
    pub type amf_uint16 = u16;
    pub type amf_uint8 = u8;
    pub type amf_size = usize;
    pub type amf_bool = u8;
    pub type amf_long = i64;
    pub type amf_pts = i64;
    pub type amf_handle = *mut c_void;
    pub type AMF_MEMORY_TYPE = i32;
    pub type AMF_SURFACE_FORMAT = i32;
    pub type AMF_PLANE_TYPE = i32;
    pub type AMF_DX_VERSION = i32;
    pub type AMF_BUFFER_USAGE = u32;
    pub type AMF_SURFACE_USAGE = u32;
    pub type AMF_MEMORY_CPU_ACCESS = u32;
    pub type AMF_VARIANT_TYPE = i32;
    pub type AMF_DATA_TYPE = i32;
    pub type AMF_FRAME_TYPE = i32;

    pub const AMF_OK: AMF_RESULT = 0;
    pub const AMF_FAIL: AMF_RESULT = 1;
    pub const AMF_UNEXPECTED: AMF_RESULT = 2;
    pub const AMF_ACCESS_DENIED: AMF_RESULT = 3;
    pub const AMF_INVALID_ARG: AMF_RESULT = 4;
    pub const AMF_OUT_OF_RANGE: AMF_RESULT = 5;
    pub const AMF_OUT_OF_MEMORY: AMF_RESULT = 6;
    pub const AMF_INVALID_POINTER: AMF_RESULT = 7;
    pub const AMF_NO_INTERFACE: AMF_RESULT = 8;
    pub const AMF_NOT_IMPLEMENTED: AMF_RESULT = 9;
    pub const AMF_NOT_SUPPORTED: AMF_RESULT = 10;
    pub const AMF_NOT_FOUND: AMF_RESULT = 11;
    pub const AMF_ALREADY_INITIALIZED: AMF_RESULT = 12;
    pub const AMF_NOT_INITIALIZED: AMF_RESULT = 13;
    pub const AMF_INVALID_FORMAT: AMF_RESULT = 14;
    pub const AMF_WRONG_STATE: AMF_RESULT = 15;
    pub const AMF_FILE_NOT_OPEN: AMF_RESULT = 16;
    pub const AMF_NO_DEVICE: AMF_RESULT = 17;
    pub const AMF_EOF: AMF_RESULT = 23;
    pub const AMF_REPEAT: AMF_RESULT = 24;
    pub const AMF_INPUT_FULL: AMF_RESULT = 25;
    pub const AMF_INVALID_DATA_TYPE: AMF_RESULT = 28;
    pub const AMF_INVALID_RESOLUTION: AMF_RESULT = 29;
    pub const AMF_CODEC_NOT_SUPPORTED: AMF_RESULT = 30;
    pub const AMF_SURFACE_FORMAT_NOT_SUPPORTED: AMF_RESULT = 31;
    pub const AMF_SURFACE_MUST_BE_SHARED: AMF_RESULT = 32;
    pub const AMF_DECODER_NOT_PRESENT: AMF_RESULT = 33;
    pub const AMF_DECODER_NO_FREE_SURFACES: AMF_RESULT = 35;
    pub const AMF_ENCODER_NOT_PRESENT: AMF_RESULT = 36;
    pub const AMF_NEED_MORE_INPUT: AMF_RESULT = 44;
    pub const AMF_VULKAN_FAILED: AMF_RESULT = 45;
    pub const AMF_DIRECTX_FAILED: AMF_RESULT = 18;

    pub const AMF_FULL_VERSION: amf_uint64 = (1u64 << 48) | (5u64 << 32);
    pub const AMF_SECOND: u64 = 10_000_000;

    pub const AMF_MEMORY_HOST: AMF_MEMORY_TYPE = 1;
    pub const AMF_DX11_0: AMF_DX_VERSION = 110;
    pub const AMF_DATA_BUFFER: AMF_DATA_TYPE = 0;
    pub const AMF_SURFACE_BGRA: AMF_SURFACE_FORMAT = 3;
    pub const AMF_PLANE_PACKED: AMF_PLANE_TYPE = 1;

    pub const AMF_VIDEO_ENCODER_HEVC_USAGE_LOW_LATENCY_HIGH_QUALITY: i32 = 5;
    pub const AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR: i32 = 3;
    pub const AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_HIGH_QUALITY: i32 = 15;
    pub const AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE_IDR_ALIGNED: i32 = 2;
    pub const AMF_VIDEO_ENCODER_HEVC_OUTPUT_MODE_FRAME: i32 = 0;
    pub const AMF_VIDEO_ENCODER_HEVC_PICTURE_TYPE_IDR: i32 = 2;
    pub const AMF_VIDEO_ENCODER_HEVC_OUTPUT_DATA_TYPE_IDR: i32 = 0;
    pub const AMF_VIDEO_ENCODER_HEVC_OUTPUT_DATA_TYPE_I: i32 = 1;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct AMFSize {
        pub width: amf_int32,
        pub height: amf_int32,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct AMFRate {
        pub num: amf_uint32,
        pub den: amf_uint32,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AMFGuid {
        pub data1: amf_uint32,
        pub data2: amf_uint16,
        pub data3: amf_uint16,
        pub data41: amf_uint8,
        pub data42: amf_uint8,
        pub data43: amf_uint8,
        pub data44: amf_uint8,
        pub data45: amf_uint8,
        pub data46: amf_uint8,
        pub data47: amf_uint8,
        pub data48: amf_uint8,
    }

    pub const AMF_CONTEXT1_IID: AMFGuid = AMFGuid {
        data1: 0xd9e9f868,
        data2: 0x6220,
        data3: 0x44c6,
        data41: 0xa2,
        data42: 0x2f,
        data43: 0x7c,
        data44: 0xd6,
        data45: 0xda,
        data46: 0xc6,
        data47: 0x86,
        data48: 0x46,
    };

    pub const AMF_BUFFER_IID: AMFGuid = AMFGuid {
        data1: 0xb04b7248,
        data2: 0xb6f0,
        data3: 0x4321,
        data41: 0xb6,
        data42: 0x91,
        data43: 0xba,
        data44: 0xa4,
        data45: 0x74,
        data46: 0x0f,
        data47: 0x9f,
        data48: 0xcb,
    };

    pub const AMF_VARIANT_BOOL: AMF_VARIANT_TYPE = 1;
    pub const AMF_VARIANT_INT64: AMF_VARIANT_TYPE = 2;
    pub const AMF_VARIANT_SIZE: AMF_VARIANT_TYPE = 5;
    pub const AMF_VARIANT_RATE: AMF_VARIANT_TYPE = 7;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub union AMFVariantValue {
        pub bool_value: amf_bool,
        pub int64_value: amf_int64,
        pub size_value: AMFSize,
        pub rate_value: AMFRate,
        pub pInterface: *mut c_void,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct AMFVariantStruct {
        pub type_: AMF_VARIANT_TYPE,
        pub value: AMFVariantValue,
    }

    impl Default for AMFVariantStruct {
        fn default() -> Self {
            Self {
                type_: 0,
                value: AMFVariantValue { int64_value: 0 },
            }
        }
    }

    pub enum AMFPropertyStorage {}
    pub enum AMFPropertyStorageObserver {}
    pub enum AMFPropertyInfo {}
    pub enum AMFComputeFactory {}
    pub enum AMFComputeDevice {}
    pub enum AMFCompute {}
    pub enum AMFAudioBuffer {}
    pub enum AMFCaps {}
    pub enum AMFDataAllocatorCB {}
    pub enum AMFComponentOptimizationCallback {}
    pub enum AMFBufferObserver {}
    pub enum AMFSurfaceObserver {}

    pub type AMFInit_Fn = unsafe extern "C" fn(amf_uint64, *mut *mut AMFFactory) -> AMF_RESULT;
    pub type AMFQueryVersion_Fn = unsafe extern "C" fn(*mut amf_uint64) -> AMF_RESULT;

    #[repr(C)]
    pub struct AMFFactory {
        pub pVtbl: *const AMFFactoryVtbl,
    }

    #[repr(C)]
    pub struct AMFFactoryVtbl {
        pub CreateContext:
            unsafe extern "system" fn(*mut AMFFactory, *mut *mut AMFContext) -> AMF_RESULT,
        pub CreateComponent: unsafe extern "system" fn(
            *mut AMFFactory,
            *mut AMFContext,
            *const AmfWideChar,
            *mut *mut AMFComponent,
        ) -> AMF_RESULT,
        pub SetCacheFolder:
            unsafe extern "system" fn(*mut AMFFactory, *const AmfWideChar) -> AMF_RESULT,
        pub GetCacheFolder: unsafe extern "system" fn(*mut AMFFactory) -> *const AmfWideChar,
        pub GetDebug: unsafe extern "system" fn(*mut AMFFactory, *mut *mut c_void) -> AMF_RESULT,
        pub GetTrace: unsafe extern "system" fn(*mut AMFFactory, *mut *mut c_void) -> AMF_RESULT,
        pub GetPrograms: unsafe extern "system" fn(*mut AMFFactory, *mut *mut c_void) -> AMF_RESULT,
    }

    #[repr(C)]
    pub struct AMFContext {
        pub pVtbl: *const AMFContextVtbl,
    }

    #[repr(C)]
    pub struct AMFContextVtbl {
        pub Acquire: unsafe extern "system" fn(*mut AMFContext) -> amf_long,
        pub Release: unsafe extern "system" fn(*mut AMFContext) -> amf_long,
        pub QueryInterface: unsafe extern "system" fn(
            *mut AMFContext,
            *const AMFGuid,
            *mut *mut c_void,
        ) -> AMF_RESULT,
        pub SetProperty: unsafe extern "system" fn(
            *mut AMFContext,
            *const AmfWideChar,
            AMFVariantStruct,
        ) -> AMF_RESULT,
        pub GetProperty: unsafe extern "system" fn(
            *mut AMFContext,
            *const AmfWideChar,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub HasProperty: unsafe extern "system" fn(*mut AMFContext, *const AmfWideChar) -> amf_bool,
        pub GetPropertyCount: unsafe extern "system" fn(*mut AMFContext) -> amf_size,
        pub GetPropertyAt: unsafe extern "system" fn(
            *mut AMFContext,
            amf_size,
            *mut AmfWideChar,
            amf_size,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub Clear: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub AddTo: unsafe extern "system" fn(
            *mut AMFContext,
            *mut AMFPropertyStorage,
            amf_bool,
            amf_bool,
        ) -> AMF_RESULT,
        pub CopyTo: unsafe extern "system" fn(
            *mut AMFContext,
            *mut AMFPropertyStorage,
            amf_bool,
        ) -> AMF_RESULT,
        pub AddObserver:
            unsafe extern "system" fn(*mut AMFContext, *mut AMFPropertyStorageObserver),
        pub RemoveObserver:
            unsafe extern "system" fn(*mut AMFContext, *mut AMFPropertyStorageObserver),
        pub Terminate: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub InitDX9: unsafe extern "system" fn(*mut AMFContext, *mut c_void) -> AMF_RESULT,
        pub GetDX9Device: unsafe extern "system" fn(*mut AMFContext, AMF_DX_VERSION) -> *mut c_void,
        pub LockDX9: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub UnlockDX9: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub InitDX11:
            unsafe extern "system" fn(*mut AMFContext, *mut c_void, AMF_DX_VERSION) -> AMF_RESULT,
        pub GetDX11Device:
            unsafe extern "system" fn(*mut AMFContext, AMF_DX_VERSION) -> *mut c_void,
        pub LockDX11: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub UnlockDX11: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub InitOpenCL: unsafe extern "system" fn(*mut AMFContext, *mut c_void) -> AMF_RESULT,
        pub GetOpenCLContext: unsafe extern "system" fn(*mut AMFContext) -> *mut c_void,
        pub GetOpenCLCommandQueue: unsafe extern "system" fn(*mut AMFContext) -> *mut c_void,
        pub GetOpenCLDeviceID: unsafe extern "system" fn(*mut AMFContext) -> *mut c_void,
        pub GetOpenCLComputeFactory:
            unsafe extern "system" fn(*mut AMFContext, *mut *mut AMFComputeFactory) -> AMF_RESULT,
        pub InitOpenCLEx:
            unsafe extern "system" fn(*mut AMFContext, *mut AMFComputeDevice) -> AMF_RESULT,
        pub LockOpenCL: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub UnlockOpenCL: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub InitOpenGL: unsafe extern "system" fn(
            *mut AMFContext,
            amf_handle,
            amf_handle,
            amf_handle,
        ) -> AMF_RESULT,
        pub GetOpenGLContext: unsafe extern "system" fn(*mut AMFContext) -> amf_handle,
        pub GetOpenGLDrawable: unsafe extern "system" fn(*mut AMFContext) -> amf_handle,
        pub LockOpenGL: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub UnlockOpenGL: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub InitXV: unsafe extern "system" fn(*mut AMFContext, *mut c_void) -> AMF_RESULT,
        pub GetXVDevice: unsafe extern "system" fn(*mut AMFContext) -> *mut c_void,
        pub LockXV: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub UnlockXV: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub InitGralloc: unsafe extern "system" fn(*mut AMFContext, *mut c_void) -> AMF_RESULT,
        pub GetGrallocDevice: unsafe extern "system" fn(*mut AMFContext) -> *mut c_void,
        pub LockGralloc: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub UnlockGralloc: unsafe extern "system" fn(*mut AMFContext) -> AMF_RESULT,
        pub AllocBuffer: unsafe extern "system" fn(
            *mut AMFContext,
            AMF_MEMORY_TYPE,
            amf_size,
            *mut *mut AMFBuffer,
        ) -> AMF_RESULT,
        pub AllocSurface: unsafe extern "system" fn(
            *mut AMFContext,
            AMF_MEMORY_TYPE,
            AMF_SURFACE_FORMAT,
            amf_int32,
            amf_int32,
            *mut *mut AMFSurface,
        ) -> AMF_RESULT,
    }

    #[repr(C)]
    pub struct AMFContext1 {
        pub pVtbl: *const AMFContext1Vtbl,
    }

    #[repr(C)]
    pub struct AMFContext1Vtbl {
        pub Acquire: unsafe extern "system" fn(*mut AMFContext1) -> amf_long,
        pub Release: unsafe extern "system" fn(*mut AMFContext1) -> amf_long,
        pub QueryInterface: unsafe extern "system" fn(
            *mut AMFContext1,
            *const AMFGuid,
            *mut *mut c_void,
        ) -> AMF_RESULT,
        pub SetProperty: unsafe extern "system" fn(
            *mut AMFContext1,
            *const AmfWideChar,
            AMFVariantStruct,
        ) -> AMF_RESULT,
        pub GetProperty: unsafe extern "system" fn(
            *mut AMFContext1,
            *const AmfWideChar,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub HasProperty:
            unsafe extern "system" fn(*mut AMFContext1, *const AmfWideChar) -> amf_bool,
        pub GetPropertyCount: unsafe extern "system" fn(*mut AMFContext1) -> amf_size,
        pub GetPropertyAt: unsafe extern "system" fn(
            *mut AMFContext1,
            amf_size,
            *mut AmfWideChar,
            amf_size,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub Clear: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub AddTo: unsafe extern "system" fn(
            *mut AMFContext1,
            *mut AMFPropertyStorage,
            amf_bool,
            amf_bool,
        ) -> AMF_RESULT,
        pub CopyTo: unsafe extern "system" fn(
            *mut AMFContext1,
            *mut AMFPropertyStorage,
            amf_bool,
        ) -> AMF_RESULT,
        pub AddObserver:
            unsafe extern "system" fn(*mut AMFContext1, *mut AMFPropertyStorageObserver),
        pub RemoveObserver:
            unsafe extern "system" fn(*mut AMFContext1, *mut AMFPropertyStorageObserver),
        pub Terminate: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub InitDX9: unsafe extern "system" fn(*mut AMFContext1, *mut c_void) -> AMF_RESULT,
        pub GetDX9Device:
            unsafe extern "system" fn(*mut AMFContext1, AMF_DX_VERSION) -> *mut c_void,
        pub LockDX9: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub UnlockDX9: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub InitDX11:
            unsafe extern "system" fn(*mut AMFContext1, *mut c_void, AMF_DX_VERSION) -> AMF_RESULT,
        pub GetDX11Device:
            unsafe extern "system" fn(*mut AMFContext1, AMF_DX_VERSION) -> *mut c_void,
        pub LockDX11: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub UnlockDX11: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub InitOpenCL: unsafe extern "system" fn(*mut AMFContext1, *mut c_void) -> AMF_RESULT,
        pub GetOpenCLContext: unsafe extern "system" fn(*mut AMFContext1) -> *mut c_void,
        pub GetOpenCLCommandQueue: unsafe extern "system" fn(*mut AMFContext1) -> *mut c_void,
        pub GetOpenCLDeviceID: unsafe extern "system" fn(*mut AMFContext1) -> *mut c_void,
        pub GetOpenCLComputeFactory:
            unsafe extern "system" fn(*mut AMFContext1, *mut *mut AMFComputeFactory) -> AMF_RESULT,
        pub InitOpenCLEx:
            unsafe extern "system" fn(*mut AMFContext1, *mut AMFComputeDevice) -> AMF_RESULT,
        pub LockOpenCL: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub UnlockOpenCL: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub InitOpenGL: unsafe extern "system" fn(
            *mut AMFContext1,
            amf_handle,
            amf_handle,
            amf_handle,
        ) -> AMF_RESULT,
        pub GetOpenGLContext: unsafe extern "system" fn(*mut AMFContext1) -> amf_handle,
        pub GetOpenGLDrawable: unsafe extern "system" fn(*mut AMFContext1) -> amf_handle,
        pub LockOpenGL: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub UnlockOpenGL: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub InitXV: unsafe extern "system" fn(*mut AMFContext1, *mut c_void) -> AMF_RESULT,
        pub GetXVDevice: unsafe extern "system" fn(*mut AMFContext1) -> *mut c_void,
        pub LockXV: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub UnlockXV: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub InitGralloc: unsafe extern "system" fn(*mut AMFContext1, *mut c_void) -> AMF_RESULT,
        pub GetGrallocDevice: unsafe extern "system" fn(*mut AMFContext1) -> *mut c_void,
        pub LockGralloc: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub UnlockGralloc: unsafe extern "system" fn(*mut AMFContext1) -> AMF_RESULT,
        pub AllocBuffer: unsafe extern "system" fn(
            *mut AMFContext1,
            AMF_MEMORY_TYPE,
            amf_size,
            *mut *mut AMFBuffer,
        ) -> AMF_RESULT,
        pub AllocSurface: unsafe extern "system" fn(
            *mut AMFContext1,
            AMF_MEMORY_TYPE,
            AMF_SURFACE_FORMAT,
            amf_int32,
            amf_int32,
            *mut *mut AMFSurface,
        ) -> AMF_RESULT,
        pub AllocAudioBuffer: unsafe extern "system" fn(
            *mut AMFContext1,
            i32,
            i32,
            i32,
            i32,
            i32,
            *mut *mut AMFAudioBuffer,
        ) -> AMF_RESULT,
        pub CreateBufferFromHostNative: unsafe extern "system" fn(
            *mut AMFContext1,
            *mut c_void,
            amf_size,
            *mut *mut AMFBuffer,
            *mut AMFBufferObserver,
        ) -> AMF_RESULT,
        pub CreateSurfaceFromHostNative: unsafe extern "system" fn(
            *mut AMFContext1,
            AMF_SURFACE_FORMAT,
            amf_int32,
            amf_int32,
            amf_int32,
            amf_int32,
            *mut c_void,
            *mut *mut AMFSurface,
            *mut AMFSurfaceObserver,
        ) -> AMF_RESULT,
        pub CreateSurfaceFromDX9Native: unsafe extern "system" fn(
            *mut AMFContext1,
            *mut c_void,
            *mut *mut AMFSurface,
            *mut AMFSurfaceObserver,
        ) -> AMF_RESULT,
        pub CreateSurfaceFromDX11Native: unsafe extern "system" fn(
            *mut AMFContext1,
            *mut c_void,
            *mut *mut AMFSurface,
            *mut AMFSurfaceObserver,
        ) -> AMF_RESULT,
        pub CreateSurfaceFromOpenGLNative: unsafe extern "system" fn(
            *mut AMFContext1,
            AMF_SURFACE_FORMAT,
            amf_handle,
            *mut *mut AMFSurface,
            *mut AMFSurfaceObserver,
        ) -> AMF_RESULT,
        pub CreateSurfaceFromGrallocNative: unsafe extern "system" fn(
            *mut AMFContext1,
            amf_handle,
            *mut *mut AMFSurface,
            *mut AMFSurfaceObserver,
        ) -> AMF_RESULT,
        pub CreateSurfaceFromOpenCLNative: unsafe extern "system" fn(
            *mut AMFContext1,
            AMF_SURFACE_FORMAT,
            amf_int32,
            amf_int32,
            *mut *mut c_void,
            *mut *mut AMFSurface,
            *mut AMFSurfaceObserver,
        ) -> AMF_RESULT,
        pub CreateBufferFromOpenCLNative: unsafe extern "system" fn(
            *mut AMFContext1,
            *mut c_void,
            amf_size,
            *mut *mut AMFBuffer,
        ) -> AMF_RESULT,
        pub GetCompute: unsafe extern "system" fn(
            *mut AMFContext1,
            AMF_MEMORY_TYPE,
            *mut *mut AMFCompute,
        ) -> AMF_RESULT,
        pub CreateBufferFromDX11NativeEx: unsafe extern "system" fn(
            *mut AMFContext1,
            *mut c_void,
            *mut *mut AMFBuffer,
            *mut AMFBufferObserver,
        ) -> AMF_RESULT,
        pub AllocBufferEx: unsafe extern "system" fn(
            *mut AMFContext1,
            AMF_MEMORY_TYPE,
            amf_size,
            AMF_BUFFER_USAGE,
            AMF_MEMORY_CPU_ACCESS,
            *mut *mut AMFBuffer,
        ) -> AMF_RESULT,
        pub AllocSurfaceEx: unsafe extern "system" fn(
            *mut AMFContext1,
            AMF_MEMORY_TYPE,
            AMF_SURFACE_FORMAT,
            amf_int32,
            amf_int32,
            AMF_SURFACE_USAGE,
            AMF_MEMORY_CPU_ACCESS,
            *mut *mut AMFSurface,
        ) -> AMF_RESULT,
        pub InitVulkan: unsafe extern "system" fn(*mut AMFContext1, *mut c_void) -> AMF_RESULT,
    }

    #[repr(C)]
    pub struct AMFComponent {
        pub pVtbl: *const AMFComponentVtbl,
    }

    #[repr(C)]
    pub struct AMFComponentVtbl {
        pub Acquire: unsafe extern "system" fn(*mut AMFComponent) -> amf_long,
        pub Release: unsafe extern "system" fn(*mut AMFComponent) -> amf_long,
        pub QueryInterface: unsafe extern "system" fn(
            *mut AMFComponent,
            *const AMFGuid,
            *mut *mut c_void,
        ) -> AMF_RESULT,
        pub SetProperty: unsafe extern "system" fn(
            *mut AMFComponent,
            *const AmfWideChar,
            AMFVariantStruct,
        ) -> AMF_RESULT,
        pub GetProperty: unsafe extern "system" fn(
            *mut AMFComponent,
            *const AmfWideChar,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub HasProperty:
            unsafe extern "system" fn(*mut AMFComponent, *const AmfWideChar) -> amf_bool,
        pub GetPropertyCount: unsafe extern "system" fn(*mut AMFComponent) -> amf_size,
        pub GetPropertyAt: unsafe extern "system" fn(
            *mut AMFComponent,
            amf_size,
            *mut AmfWideChar,
            amf_size,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub Clear: unsafe extern "system" fn(*mut AMFComponent) -> AMF_RESULT,
        pub AddTo: unsafe extern "system" fn(
            *mut AMFComponent,
            *mut AMFPropertyStorage,
            amf_bool,
            amf_bool,
        ) -> AMF_RESULT,
        pub CopyTo: unsafe extern "system" fn(
            *mut AMFComponent,
            *mut AMFPropertyStorage,
            amf_bool,
        ) -> AMF_RESULT,
        pub AddObserver:
            unsafe extern "system" fn(*mut AMFComponent, *mut AMFPropertyStorageObserver),
        pub RemoveObserver:
            unsafe extern "system" fn(*mut AMFComponent, *mut AMFPropertyStorageObserver),
        pub GetPropertiesInfoCount: unsafe extern "system" fn(*mut AMFComponent) -> amf_size,
        pub GetPropertyInfoAt: unsafe extern "system" fn(
            *mut AMFComponent,
            amf_size,
            *mut *const AMFPropertyInfo,
        ) -> AMF_RESULT,
        pub GetPropertyInfo: unsafe extern "system" fn(
            *mut AMFComponent,
            *const AmfWideChar,
            *mut *const AMFPropertyInfo,
        ) -> AMF_RESULT,
        pub ValidateProperty: unsafe extern "system" fn(
            *mut AMFComponent,
            *const AmfWideChar,
            AMFVariantStruct,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub Init: unsafe extern "system" fn(
            *mut AMFComponent,
            AMF_SURFACE_FORMAT,
            amf_int32,
            amf_int32,
        ) -> AMF_RESULT,
        pub ReInit:
            unsafe extern "system" fn(*mut AMFComponent, amf_int32, amf_int32) -> AMF_RESULT,
        pub Terminate: unsafe extern "system" fn(*mut AMFComponent) -> AMF_RESULT,
        pub Drain: unsafe extern "system" fn(*mut AMFComponent) -> AMF_RESULT,
        pub Flush: unsafe extern "system" fn(*mut AMFComponent) -> AMF_RESULT,
        pub SubmitInput: unsafe extern "system" fn(*mut AMFComponent, *mut AMFData) -> AMF_RESULT,
        pub QueryOutput:
            unsafe extern "system" fn(*mut AMFComponent, *mut *mut AMFData) -> AMF_RESULT,
    }

    #[repr(C)]
    pub struct AMFData {
        pub pVtbl: *const AMFDataVtbl,
    }

    #[repr(C)]
    pub struct AMFDataVtbl {
        pub Acquire: unsafe extern "system" fn(*mut AMFData) -> amf_long,
        pub Release: unsafe extern "system" fn(*mut AMFData) -> amf_long,
        pub QueryInterface:
            unsafe extern "system" fn(*mut AMFData, *const AMFGuid, *mut *mut c_void) -> AMF_RESULT,
        pub SetProperty: unsafe extern "system" fn(
            *mut AMFData,
            *const AmfWideChar,
            AMFVariantStruct,
        ) -> AMF_RESULT,
        pub GetProperty: unsafe extern "system" fn(
            *mut AMFData,
            *const AmfWideChar,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub HasProperty: unsafe extern "system" fn(*mut AMFData, *const AmfWideChar) -> amf_bool,
        pub GetPropertyCount: unsafe extern "system" fn(*mut AMFData) -> amf_size,
        pub GetPropertyAt: unsafe extern "system" fn(
            *mut AMFData,
            amf_size,
            *mut AmfWideChar,
            amf_size,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub Clear: unsafe extern "system" fn(*mut AMFData) -> AMF_RESULT,
        pub AddTo: unsafe extern "system" fn(
            *mut AMFData,
            *mut AMFPropertyStorage,
            amf_bool,
            amf_bool,
        ) -> AMF_RESULT,
        pub CopyTo: unsafe extern "system" fn(
            *mut AMFData,
            *mut AMFPropertyStorage,
            amf_bool,
        ) -> AMF_RESULT,
        pub AddObserver: unsafe extern "system" fn(*mut AMFData, *mut AMFPropertyStorageObserver),
        pub RemoveObserver:
            unsafe extern "system" fn(*mut AMFData, *mut AMFPropertyStorageObserver),
        pub GetMemoryType: unsafe extern "system" fn(*mut AMFData) -> AMF_MEMORY_TYPE,
        pub Duplicate: unsafe extern "system" fn(
            *mut AMFData,
            AMF_MEMORY_TYPE,
            *mut *mut AMFData,
        ) -> AMF_RESULT,
        pub Convert: unsafe extern "system" fn(*mut AMFData, AMF_MEMORY_TYPE) -> AMF_RESULT,
        pub Interop: unsafe extern "system" fn(*mut AMFData, AMF_MEMORY_TYPE) -> AMF_RESULT,
        pub GetDataType: unsafe extern "system" fn(*mut AMFData) -> AMF_DATA_TYPE,
        pub IsReusable: unsafe extern "system" fn(*mut AMFData) -> amf_bool,
        pub SetPts: unsafe extern "system" fn(*mut AMFData, amf_pts),
        pub GetPts: unsafe extern "system" fn(*mut AMFData) -> amf_pts,
        pub SetDuration: unsafe extern "system" fn(*mut AMFData, amf_pts),
        pub GetDuration: unsafe extern "system" fn(*mut AMFData) -> amf_pts,
    }

    #[repr(C)]
    pub struct AMFSurface {
        pub pVtbl: *const AMFSurfaceVtbl,
    }

    #[repr(C)]
    pub struct AMFSurfaceVtbl {
        pub Acquire: unsafe extern "system" fn(*mut AMFSurface) -> amf_long,
        pub Release: unsafe extern "system" fn(*mut AMFSurface) -> amf_long,
        pub QueryInterface: unsafe extern "system" fn(
            *mut AMFSurface,
            *const AMFGuid,
            *mut *mut c_void,
        ) -> AMF_RESULT,
        pub SetProperty: unsafe extern "system" fn(
            *mut AMFSurface,
            *const AmfWideChar,
            AMFVariantStruct,
        ) -> AMF_RESULT,
        pub GetProperty: unsafe extern "system" fn(
            *mut AMFSurface,
            *const AmfWideChar,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub HasProperty: unsafe extern "system" fn(*mut AMFSurface, *const AmfWideChar) -> amf_bool,
        pub GetPropertyCount: unsafe extern "system" fn(*mut AMFSurface) -> amf_size,
        pub GetPropertyAt: unsafe extern "system" fn(
            *mut AMFSurface,
            amf_size,
            *mut AmfWideChar,
            amf_size,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub Clear: unsafe extern "system" fn(*mut AMFSurface) -> AMF_RESULT,
        pub AddTo: unsafe extern "system" fn(
            *mut AMFSurface,
            *mut AMFPropertyStorage,
            amf_bool,
            amf_bool,
        ) -> AMF_RESULT,
        pub CopyTo: unsafe extern "system" fn(
            *mut AMFSurface,
            *mut AMFPropertyStorage,
            amf_bool,
        ) -> AMF_RESULT,
        pub AddObserver:
            unsafe extern "system" fn(*mut AMFSurface, *mut AMFPropertyStorageObserver),
        pub RemoveObserver:
            unsafe extern "system" fn(*mut AMFSurface, *mut AMFPropertyStorageObserver),
        pub GetMemoryType: unsafe extern "system" fn(*mut AMFSurface) -> AMF_MEMORY_TYPE,
        pub Duplicate: unsafe extern "system" fn(
            *mut AMFSurface,
            AMF_MEMORY_TYPE,
            *mut *mut AMFData,
        ) -> AMF_RESULT,
        pub Convert: unsafe extern "system" fn(*mut AMFSurface, AMF_MEMORY_TYPE) -> AMF_RESULT,
        pub Interop: unsafe extern "system" fn(*mut AMFSurface, AMF_MEMORY_TYPE) -> AMF_RESULT,
        pub GetDataType: unsafe extern "system" fn(*mut AMFSurface) -> AMF_DATA_TYPE,
        pub IsReusable: unsafe extern "system" fn(*mut AMFSurface) -> amf_bool,
        pub SetPts: unsafe extern "system" fn(*mut AMFSurface, amf_pts),
        pub GetPts: unsafe extern "system" fn(*mut AMFSurface) -> amf_pts,
        pub SetDuration: unsafe extern "system" fn(*mut AMFSurface, amf_pts),
        pub GetDuration: unsafe extern "system" fn(*mut AMFSurface) -> amf_pts,
        pub GetFormat: unsafe extern "system" fn(*mut AMFSurface) -> AMF_SURFACE_FORMAT,
        pub GetPlanesCount: unsafe extern "system" fn(*mut AMFSurface) -> amf_size,
        pub GetPlaneAt: unsafe extern "system" fn(*mut AMFSurface, amf_size) -> *mut AMFPlane,
        pub GetPlane: unsafe extern "system" fn(*mut AMFSurface, AMF_PLANE_TYPE) -> *mut AMFPlane,
        pub GetFrameType: unsafe extern "system" fn(*mut AMFSurface) -> AMF_FRAME_TYPE,
        pub SetFrameType: unsafe extern "system" fn(*mut AMFSurface, AMF_FRAME_TYPE),
        pub SetCrop: unsafe extern "system" fn(
            *mut AMFSurface,
            amf_int32,
            amf_int32,
            amf_int32,
            amf_int32,
        ) -> AMF_RESULT,
    }

    #[repr(C)]
    pub struct AMFPlane {
        pub pVtbl: *const AMFPlaneVtbl,
    }

    #[repr(C)]
    pub struct AMFPlaneVtbl {
        pub Acquire: unsafe extern "system" fn(*mut AMFPlane) -> amf_long,
        pub Release: unsafe extern "system" fn(*mut AMFPlane) -> amf_long,
        pub QueryInterface: unsafe extern "system" fn(
            *mut AMFPlane,
            *const AMFGuid,
            *mut *mut c_void,
        ) -> AMF_RESULT,
        pub GetType: unsafe extern "system" fn(*mut AMFPlane) -> AMF_PLANE_TYPE,
        pub GetNative: unsafe extern "system" fn(*mut AMFPlane) -> *mut c_void,
        pub GetPixelSizeInBytes: unsafe extern "system" fn(*mut AMFPlane) -> amf_int32,
        pub GetOffsetX: unsafe extern "system" fn(*mut AMFPlane) -> amf_int32,
        pub GetOffsetY: unsafe extern "system" fn(*mut AMFPlane) -> amf_int32,
        pub GetWidth: unsafe extern "system" fn(*mut AMFPlane) -> amf_int32,
        pub GetHeight: unsafe extern "system" fn(*mut AMFPlane) -> amf_int32,
        pub GetHPitch: unsafe extern "system" fn(*mut AMFPlane) -> amf_int32,
        pub GetVPitch: unsafe extern "system" fn(*mut AMFPlane) -> amf_int32,
        pub IsTiled: unsafe extern "system" fn(*mut AMFPlane) -> amf_bool,
    }

    #[repr(C)]
    pub struct AMFBuffer {
        pub pVtbl: *const AMFBufferVtbl,
    }

    #[repr(C)]
    pub struct AMFBufferVtbl {
        pub Acquire: unsafe extern "system" fn(*mut AMFBuffer) -> amf_long,
        pub Release: unsafe extern "system" fn(*mut AMFBuffer) -> amf_long,
        pub QueryInterface: unsafe extern "system" fn(
            *mut AMFBuffer,
            *const AMFGuid,
            *mut *mut c_void,
        ) -> AMF_RESULT,
        pub SetProperty: unsafe extern "system" fn(
            *mut AMFBuffer,
            *const AmfWideChar,
            AMFVariantStruct,
        ) -> AMF_RESULT,
        pub GetProperty: unsafe extern "system" fn(
            *mut AMFBuffer,
            *const AmfWideChar,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub HasProperty: unsafe extern "system" fn(*mut AMFBuffer, *const AmfWideChar) -> amf_bool,
        pub GetPropertyCount: unsafe extern "system" fn(*mut AMFBuffer) -> amf_size,
        pub GetPropertyAt: unsafe extern "system" fn(
            *mut AMFBuffer,
            amf_size,
            *mut AmfWideChar,
            amf_size,
            *mut AMFVariantStruct,
        ) -> AMF_RESULT,
        pub Clear: unsafe extern "system" fn(*mut AMFBuffer) -> AMF_RESULT,
        pub AddTo: unsafe extern "system" fn(
            *mut AMFBuffer,
            *mut AMFPropertyStorage,
            amf_bool,
            amf_bool,
        ) -> AMF_RESULT,
        pub CopyTo: unsafe extern "system" fn(
            *mut AMFBuffer,
            *mut AMFPropertyStorage,
            amf_bool,
        ) -> AMF_RESULT,
        pub AddObserver: unsafe extern "system" fn(*mut AMFBuffer, *mut AMFPropertyStorageObserver),
        pub RemoveObserver:
            unsafe extern "system" fn(*mut AMFBuffer, *mut AMFPropertyStorageObserver),
        pub GetMemoryType: unsafe extern "system" fn(*mut AMFBuffer) -> AMF_MEMORY_TYPE,
        pub Duplicate: unsafe extern "system" fn(
            *mut AMFBuffer,
            AMF_MEMORY_TYPE,
            *mut *mut AMFData,
        ) -> AMF_RESULT,
        pub Convert: unsafe extern "system" fn(*mut AMFBuffer, AMF_MEMORY_TYPE) -> AMF_RESULT,
        pub Interop: unsafe extern "system" fn(*mut AMFBuffer, AMF_MEMORY_TYPE) -> AMF_RESULT,
        pub GetDataType: unsafe extern "system" fn(*mut AMFBuffer) -> AMF_DATA_TYPE,
        pub IsReusable: unsafe extern "system" fn(*mut AMFBuffer) -> amf_bool,
        pub SetPts: unsafe extern "system" fn(*mut AMFBuffer, amf_pts),
        pub GetPts: unsafe extern "system" fn(*mut AMFBuffer) -> amf_pts,
        pub SetDuration: unsafe extern "system" fn(*mut AMFBuffer, amf_pts),
        pub GetDuration: unsafe extern "system" fn(*mut AMFBuffer) -> amf_pts,
        pub SetSize: unsafe extern "system" fn(*mut AMFBuffer, amf_size) -> AMF_RESULT,
        pub GetSize: unsafe extern "system" fn(*mut AMFBuffer) -> amf_size,
        pub GetNative: unsafe extern "system" fn(*mut AMFBuffer) -> *mut c_void,
    }
}
