#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("ustreamer-demo currently requires macOS for VideoToolbox encoding")
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use tracing_subscriber::EnvFilter;

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("ustreamer_demo=info,ustreamer_transport=info"));
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();

    macos_demo::run()
}

#[cfg(target_os = "macos")]
mod macos_demo {
    use std::borrow::Cow;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    use anyhow::{Context, Result, anyhow};
    use bytemuck::{Pod, Zeroable};
    use tokio::runtime::Runtime;
    use tracing::{error, info, warn};
    use ustreamer_capture::{FrameCapture, staging::StagingCapture};
    use ustreamer_encode::{FrameEncoder, videotoolbox::VideoToolboxEncoder};
    use ustreamer_input::{AppAction, InputMapper, InteractionMode};
    use ustreamer_proto::control::{ControlMessage, SessionMetricsMessage, StatusMessage};
    use ustreamer_proto::frame::packetize_frame;
    use ustreamer_proto::input::InputEvent;
    use ustreamer_proto::quality::QualityTier;
    use ustreamer_quality::{QualityConfig, QualityController};
    use ustreamer_transport::{WebSocketServer, WebSocketSession};

    const STREAM_PORT: u16 = 8080;
    const HTTP_PORT: u16 = 8090;
    const METRICS_INTERVAL: Duration = Duration::from_millis(250);
    const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(25);
    const CLIENT_HTML: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../client/index.html"
    ));

    pub fn run() -> Result<()> {
        let stream_addr = SocketAddr::from(([127, 0, 0, 1], STREAM_PORT));
        let http_addr = SocketAddr::from(([127, 0, 0, 1], HTTP_PORT));

        let _http_thread = spawn_http_server(http_addr)?;
        let runtime = Runtime::new().context("failed to create tokio runtime")?;

        let current_session = Arc::new(Mutex::new(None::<SessionSlot>));
        let session_counter = Arc::new(AtomicU64::new(1));
        let (input_tx, input_rx) = mpsc::channel();

        {
            let current_session = Arc::clone(&current_session);
            let session_counter = Arc::clone(&session_counter);
            runtime.spawn(async move {
                if let Err(error) =
                    accept_websocket_sessions(stream_addr, current_session, session_counter, input_tx)
                        .await
                {
                    error!("websocket accept loop exited: {error:#}");
                }
            });
        }

        let mut quality = QualityController::new(QualityConfig::default());
        quality.set_tier(QualityTier::Low);
        let initial_params = quality.frame_params();
        let mut renderer = runtime
            .block_on(HeadlessRenderer::new(initial_params.width, initial_params.height))
            .context("failed to create headless renderer")?;
        let mut capture = StagingCapture::new(3);
        let mut encoder = VideoToolboxEncoder::new();
        let mut scene = DemoScene::default();
        let mut frame_id = 0u32;
        let mut configured_generation = None::<u64>;
        let mut last_metrics_sent = Instant::now();
        let start_time = Instant::now();

        info!("Headless demo ready.");
        info!("Open http://127.0.0.1:{HTTP_PORT}/ in Chrome/Chromium.");
        info!("WebSocket stream endpoint: ws://127.0.0.1:{STREAM_PORT}/stream");
        info!("Controls: drag to interact, wheel to shift hue, keys 1-4 switch drag mode, R resets.");

        loop {
            let frame_started_at = Instant::now();
            let pending_status = drain_input_events(&input_rx, &mut quality, &mut scene);
            let Some(session) = snapshot_session(&current_session) else {
                configured_generation = None;
                thread::sleep(IDLE_POLL_INTERVAL);
                continue;
            };

            let params = quality.frame_params();
            renderer
                .ensure_size(params.width, params.height)
                .context("failed to resize headless renderer")?;
            renderer
                .render(start_time.elapsed().as_secs_f32(), &scene)
                .context("failed to render demo frame")?;

            let captured_frame = capture
                .capture(renderer.device(), renderer.queue(), renderer.texture())
                .context("failed to capture rendered frame")?;
            let encoded_frame = encoder
                .encode(&captured_frame, &params)
                .context("failed to encode captured frame")?;

            if configured_generation != Some(session.generation) {
                let decoder_config = encoder
                    .decoder_config()
                    .ok_or_else(|| anyhow!("encoder did not expose decoder config yet"))?;
                let help_message = ControlMessage::Status(StatusMessage::new(scene.help_text()))
                    .to_bytes()
                    .context("failed to serialize demo help message")?;
                let decoder_message = decoder_config.to_control_message_bytes();
                let session_clone = session.session.clone();
                let send_result: Result<()> = runtime.block_on(async {
                    session_clone
                        .send_control_message(&decoder_message)
                        .await
                        .context("failed to send decoder config")?;
                    session_clone
                        .send_control_message(&help_message)
                        .await
                        .context("failed to send demo help message")?;
                    Ok(())
                });
                if let Err(error) = send_result {
                    warn!("failed to initialize browser session: {error:#}");
                    clear_session_if_current(&current_session, session.generation);
                    configured_generation = None;
                    continue;
                }
                configured_generation = Some(session.generation);
            }

            if let Some(status) = pending_status {
                let status_message = ControlMessage::Status(StatusMessage::new(status))
                    .to_bytes()
                    .context("failed to serialize status message")?;
                let session_clone = session.session.clone();
                if let Err(error) = runtime.block_on(async {
                    session_clone
                        .send_control_message(&status_message)
                        .await
                        .context("failed to send status message")
                }) {
                    warn!("failed to push status message: {error:#}");
                    clear_session_if_current(&current_session, session.generation);
                    configured_generation = None;
                    continue;
                }
            }

            let timestamp_us = start_time.elapsed().as_micros().min(u64::MAX as u128) as u64;
            let packets = packetize_frame(
                frame_id,
                timestamp_us,
                encoded_frame.is_keyframe,
                encoded_frame.is_lossless,
                &encoded_frame.data,
            );
            frame_id = frame_id.wrapping_add(1);

            let mut control_messages = Vec::new();
            if last_metrics_sent.elapsed() >= METRICS_INTERVAL {
                control_messages.push(
                    ControlMessage::SessionMetrics(
                        SessionMetricsMessage::new()
                            .with_encode_time_us(encoded_frame.encode_time_us),
                    )
                    .to_bytes()
                    .context("failed to serialize session metrics")?,
                );
                last_metrics_sent = Instant::now();
            }

            let session_clone = session.session.clone();
            let send_result: Result<()> = runtime.block_on(async {
                session_clone
                    .send_frame_packets(&packets)
                    .await
                    .context("failed to send frame packets")?;
                for message in &control_messages {
                    session_clone
                        .send_control_message(message)
                        .await
                        .context("failed to send control message")?;
                }
                Ok(())
            });
            if let Err(error) = send_result {
                warn!("stream send failed: {error:#}");
                clear_session_if_current(&current_session, session.generation);
                configured_generation = None;
                continue;
            }

            let target_frame_time =
                Duration::from_secs_f64(1.0 / params.target_fps.max(1) as f64);
            let elapsed = frame_started_at.elapsed();
            if elapsed < target_frame_time {
                thread::sleep(target_frame_time - elapsed);
            }
        }
    }

    async fn accept_websocket_sessions(
        bind_address: SocketAddr,
        current_session: Arc<Mutex<Option<SessionSlot>>>,
        session_counter: Arc<AtomicU64>,
        input_tx: mpsc::Sender<InputEvent>,
    ) -> Result<()> {
        let server = WebSocketServer::bind(bind_address)
            .await
            .context("failed to bind websocket server")?;
        info!("Listening for browser sessions on ws://{}/stream", server.local_addr()?);

        loop {
            let accepted = server
                .accept_session()
                .await
                .context("failed to accept websocket session")?;
            if accepted.path != "/stream" {
                warn!("accepting websocket session on unexpected path {}", accepted.path);
            }

            let generation = session_counter.fetch_add(1, Ordering::Relaxed);
            {
                let mut slot = current_session.lock().expect("session mutex poisoned");
                *slot = Some(SessionSlot {
                    generation,
                    session: accepted.session.clone(),
                });
            }
            info!("browser session {generation} connected from {}", accepted.session.remote_address());

            let current_session = Arc::clone(&current_session);
            let input_tx = input_tx.clone();
            let session = accepted.session.clone();
            tokio::spawn(async move {
                loop {
                    match session.recv_input().await {
                        Ok(received) => {
                            if input_tx.send(received.event).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            warn!("browser session {generation} closed: {error}");
                            break;
                        }
                    }
                }

                clear_session_if_current(&current_session, generation);
            });
        }
    }

    fn spawn_http_server(bind_address: SocketAddr) -> Result<thread::JoinHandle<()>> {
        let listener = TcpListener::bind(bind_address)
            .with_context(|| format!("failed to bind demo HTTP server on {bind_address}"))?;

        let handle = thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        if let Err(error) = handle_http_connection(stream) {
                            warn!("http connection error: {error:#}");
                        }
                    }
                    Err(error) => warn!("failed to accept http connection: {error}"),
                }
            }
        });

        Ok(handle)
    }

    fn handle_http_connection(mut stream: TcpStream) -> Result<()> {
        let mut request_line = String::new();
        {
            let mut reader = BufReader::new(&mut stream);
            reader
                .read_line(&mut request_line)
                .context("failed to read HTTP request line")?;
        }

        let path = request_line.split_whitespace().nth(1).unwrap_or("/");
        let (status, content_type, body): (&str, &str, &[u8]) = match path {
            "/" | "/index.html" | "/client/index.html" => {
                ("200 OK", "text/html; charset=utf-8", CLIENT_HTML.as_bytes())
            }
            "/healthz" => ("200 OK", "text/plain; charset=utf-8", b"ok\n"),
            _ => ("404 Not Found", "text/plain; charset=utf-8", b"not found\n"),
        };

        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .context("failed to write HTTP response headers")?;
        stream
            .write_all(body)
            .context("failed to write HTTP response body")?;
        Ok(())
    }

    fn drain_input_events(
        input_rx: &mpsc::Receiver<InputEvent>,
        quality: &mut QualityController,
        scene: &mut DemoScene,
    ) -> Option<String> {
        let mut last_status = None;
        while let Ok(event) = input_rx.try_recv() {
            quality.on_input();
            if let Some(status) = scene.apply_input(event) {
                last_status = Some(status);
            }
        }

        last_status
    }

    fn snapshot_session(current_session: &Arc<Mutex<Option<SessionSlot>>>) -> Option<SessionSlot> {
        current_session
            .lock()
            .expect("session mutex poisoned")
            .clone()
    }

    fn clear_session_if_current(current_session: &Arc<Mutex<Option<SessionSlot>>>, generation: u64) {
        let mut slot = current_session.lock().expect("session mutex poisoned");
        if slot.as_ref().map(|current| current.generation) == Some(generation) {
            *slot = None;
        }
    }

    #[derive(Clone)]
    struct SessionSlot {
        generation: u64,
        session: WebSocketSession,
    }

    struct DemoScene {
        input_mapper: InputMapper,
        pointer: [f32; 2],
        pan: [f32; 2],
        angle: f32,
        zoom: f32,
        hue_shift: f32,
        exposure: f32,
        mode_value: f32,
    }

    impl Default for DemoScene {
        fn default() -> Self {
            Self {
                input_mapper: InputMapper::default(),
                pointer: [0.5, 0.5],
                pan: [0.0, 0.0],
                angle: 0.0,
                zoom: 1.0,
                hue_shift: 0.0,
                exposure: 1.0,
                mode_value: 0.0,
            }
        }
    }

    impl DemoScene {
        fn apply_input(&mut self, event: InputEvent) -> Option<String> {
            match event {
                InputEvent::KeyDown { code } if code == b'1' as u16 => {
                    self.input_mapper.set_mode(InteractionMode::Rotate);
                    self.mode_value = 0.0;
                    return Some("Mode 1: rotate".into());
                }
                InputEvent::KeyDown { code } if code == b'2' as u16 => {
                    self.input_mapper.set_mode(InteractionMode::Pan);
                    self.mode_value = 1.0;
                    return Some("Mode 2: pan".into());
                }
                InputEvent::KeyDown { code } if code == b'3' as u16 => {
                    self.input_mapper.set_mode(InteractionMode::Zoom);
                    self.mode_value = 2.0;
                    return Some("Mode 3: zoom".into());
                }
                InputEvent::KeyDown { code } if code == b'4' as u16 => {
                    self.input_mapper.set_mode(InteractionMode::DragAdjust);
                    self.mode_value = 3.0;
                    return Some("Mode 4: drag adjust".into());
                }
                InputEvent::KeyDown { code } if code == b'R' as u16 => {
                    *self = Self::default();
                    return Some("Scene reset".into());
                }
                _ => {}
            }

            for action in self.input_mapper.process(&event) {
                match action {
                    AppAction::Rotate { dx, dy } => {
                        self.angle += dx * 3.5;
                        self.hue_shift = (self.hue_shift + dy * 0.25).rem_euclid(1.0);
                    }
                    AppAction::Zoom { delta } => {
                        self.zoom = (self.zoom * (1.0 - delta * 1.25)).clamp(0.35, 3.5);
                    }
                    AppAction::Pan { dx, dy } => {
                        self.pan[0] += dx * 1.6 / self.zoom.max(0.35);
                        self.pan[1] -= dy * 1.6 / self.zoom.max(0.35);
                    }
                    AppAction::ScrollStep { delta } => {
                        self.hue_shift = (self.hue_shift + delta as f32 * 0.06).rem_euclid(1.0);
                    }
                    AppAction::DragAdjust { dx, dy } => {
                        self.hue_shift = (self.hue_shift + dx * 0.4).rem_euclid(1.0);
                        self.exposure = (self.exposure - dy * 1.6).clamp(0.5, 1.75);
                    }
                    AppAction::PointerUpdate { x, y } => {
                        self.pointer = [x, y];
                    }
                }
            }

            None
        }

        fn help_text(&self) -> &'static str {
            "Live demo ready. Drag to interact, wheel shifts hue, keys 1-4 switch drag mode, R resets."
        }

        fn uniforms(&self, width: u32, height: u32, time_seconds: f32) -> SceneUniform {
            SceneUniform {
                resolution: [width as f32, height as f32],
                pointer: self.pointer,
                pan: self.pan,
                time: time_seconds,
                angle: self.angle,
                zoom: self.zoom,
                hue_shift: self.hue_shift,
                exposure: self.exposure,
                mode_value: self.mode_value,
                _padding: [0.0; 4],
            }
        }
    }

    struct HeadlessRenderer {
        device: wgpu::Device,
        queue: wgpu::Queue,
        texture: wgpu::Texture,
        texture_view: wgpu::TextureView,
        uniform_buffer: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        pipeline: wgpu::RenderPipeline,
        size: (u32, u32),
    }

    impl HeadlessRenderer {
        async fn new(width: u32, height: u32) -> Result<Self> {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .context("failed to request headless wgpu adapter")?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("ustreamer-demo-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                })
                .await
                .context("failed to request wgpu device")?;

            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ustreamer-demo-uniforms"),
                size: std::mem::size_of::<SceneUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ustreamer-demo-bind-group-layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                NonZeroU64::new(std::mem::size_of::<SceneUniform>() as u64).unwrap(),
                            ),
                        },
                        count: None,
                    }],
                });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ustreamer-demo-bind-group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ustreamer-demo-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DEMO_SHADER)),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ustreamer-demo-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ustreamer-demo-render-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            });

            let (texture, texture_view) = create_render_target(&device, width, height);

            Ok(Self {
                device,
                queue,
                texture,
                texture_view,
                uniform_buffer,
                bind_group,
                pipeline,
                size: (width, height),
            })
        }

        fn ensure_size(&mut self, width: u32, height: u32) -> Result<()> {
            if self.size == (width, height) {
                return Ok(());
            }

            let (texture, texture_view) = create_render_target(&self.device, width, height);
            self.texture = texture;
            self.texture_view = texture_view;
            self.size = (width, height);
            Ok(())
        }

        fn render(&mut self, time_seconds: f32, scene: &DemoScene) -> Result<()> {
            let uniforms = scene.uniforms(self.size.0, self.size.1, time_seconds);
            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ustreamer-demo-render"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ustreamer-demo-render-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.texture_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            Ok(())
        }

        fn device(&self) -> &wgpu::Device {
            &self.device
        }

        fn queue(&self) -> &wgpu::Queue {
            &self.queue
        }

        fn texture(&self) -> &wgpu::Texture {
            &self.texture
        }
    }

    fn create_render_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ustreamer-demo-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct SceneUniform {
        resolution: [f32; 2],
        pointer: [f32; 2],
        pan: [f32; 2],
        time: f32,
        angle: f32,
        zoom: f32,
        hue_shift: f32,
        exposure: f32,
        mode_value: f32,
        _padding: [f32; 4],
    }

    const DEMO_SHADER: &str = r#"
struct SceneUniform {
  resolution: vec2<f32>,
  pointer: vec2<f32>,
  pan: vec2<f32>,
  time: f32,
  angle: f32,
  zoom: f32,
  hue_shift: f32,
  exposure: f32,
  mode_value: f32,
  _padding: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

fn rotate2d(value: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(
    value.x * c - value.y * s,
    value.x * s + value.y * c
  );
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
  let k = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  let p = abs(fract(hsv.xxx + k.xyz) * 6.0 - k.www);
  return hsv.z * mix(k.xxx, clamp(p - k.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), hsv.y);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(3.0, 1.0)
  );

  let position = positions[vertex_index];
  var out: VertexOut;
  out.position = vec4<f32>(position, 0.0, 1.0);
  out.uv = position * 0.5 + vec2<f32>(0.5, 0.5);
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let aspect = scene.resolution.x / max(scene.resolution.y, 1.0);
  let screen = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  var world = screen;
  world.x *= aspect;
  world = rotate2d((world + scene.pan) / max(scene.zoom, 0.2), scene.angle);

  let pulse = 0.35 + 0.07 * sin(scene.time * 0.9 + scene.mode_value * 0.4);
  let ring = smoothstep(0.12, 0.0, abs(length(world) - pulse));
  let orb_center = vec2<f32>(
    0.35 * sin(scene.time * 0.7 + scene.mode_value * 0.3),
    0.25 * cos(scene.time * 1.1)
  );
  let orb = smoothstep(0.22, 0.0, length(world - orb_center));
  let scan = 0.5 + 0.5 * sin(world.x * 10.0 + scene.time * 1.8)
                   * cos(world.y * 9.0 - scene.time * 1.1);

  let pointer = vec2<f32>(scene.pointer.x * 2.0 - 1.0, 1.0 - scene.pointer.y * 2.0);
  let cursor_delta = abs(screen - pointer);
  let cursor = (1.0 - smoothstep(0.0, 0.015, cursor_delta.x))
             + (1.0 - smoothstep(0.0, 0.02, cursor_delta.y));

  let grid = 0.08
    * (1.0 - smoothstep(0.0, 0.02, abs(fract(world.x * 3.0 + 0.5) - 0.5)))
    * (1.0 - smoothstep(0.0, 0.02, abs(fract(world.y * 3.0 + 0.5) - 0.5)));

  let hue = fract(scene.hue_shift + 0.18 * scan + 0.1 * orb + 0.03 * scene.time);
  var color = hsv_to_rgb(vec3<f32>(hue, 0.78, 0.22 + 0.4 * scan));
  color += ring * vec3<f32>(0.25, 0.35, 0.9);
  color += orb * vec3<f32>(0.95, 0.35, 0.2);
  color += grid * vec3<f32>(0.15, 0.18, 0.22);
  color += cursor * vec3<f32>(0.95, 0.95, 0.95);
  color *= scene.exposure;

  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;
}
