fn main() -> anyhow::Result<()> {
    use tracing_subscriber::EnvFilter;

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("ustreamer_demo=info,ustreamer_transport=info"));
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();

    demo::run()
}

mod demo {
    use std::borrow::Cow;
    use std::collections::HashSet;
    use std::env;
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
    use ustreamer_app::{
        AppActionSink, LocalStreamEndpoints, MappedInputApp, StreamFrameProvider,
        drain_mapped_input_events,
    };
    use ustreamer_capture::FrameCapture;
    #[cfg(all(
        feature = "nvenc-direct",
        any(target_os = "linux", target_os = "windows")
    ))]
    use ustreamer_capture::VulkanExternalCapture;
    #[cfg(target_os = "macos")]
    use ustreamer_capture::staging::StagingCapture;
    use ustreamer_encode::FrameEncoder;
    #[cfg(all(
        feature = "nvenc-direct",
        any(target_os = "linux", target_os = "windows")
    ))]
    use ustreamer_encode::nvenc::{NvencCodec, NvencEncoder, NvencEncoderConfig, NvencInputFormat};
    #[cfg(target_os = "macos")]
    use ustreamer_encode::videotoolbox::VideoToolboxEncoder;
    use ustreamer_input::{AppAction, InputMapper, InteractionMode};
    use ustreamer_proto::control::{
        ControlMessage, FrameChecksumMessage, SessionMetricsMessage, StatusMessage,
    };
    use ustreamer_proto::frame::{FramePacket, packetize_frame};
    use ustreamer_proto::input::InputEvent;
    use ustreamer_proto::quality::QualityTier;
    use ustreamer_quality::{QualityConfig, QualityController};
    use ustreamer_transport::{WebSocketServer, WebSocketSession};

    const METRICS_INTERVAL: Duration = Duration::from_millis(250);
    const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(25);
    const CLIENT_HTML: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../client/index.html"
    ));

    type SessionRegistry = Arc<Mutex<Vec<SessionSlot>>>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    enum DemoCodec {
        #[default]
        Hevc,
        Av1,
    }

    impl DemoCodec {
        fn parse(value: &str) -> Result<Self> {
            match value {
                "hevc" => Ok(Self::Hevc),
                "av1" => Ok(Self::Av1),
                other => {
                    anyhow::bail!("unsupported --codec value `{other}`; expected `hevc` or `av1`")
                }
            }
        }

        fn arg_name(self) -> &'static str {
            match self {
                Self::Hevc => "hevc",
                Self::Av1 => "av1",
            }
        }

        #[cfg(all(
            feature = "nvenc-direct",
            any(target_os = "linux", target_os = "windows")
        ))]
        fn display_name(self) -> &'static str {
            match self {
                Self::Hevc => "HEVC",
                Self::Av1 => "AV1",
            }
        }

        #[cfg(all(
            feature = "nvenc-direct",
            any(target_os = "linux", target_os = "windows")
        ))]
        fn nvenc_codec(self) -> NvencCodec {
            match self {
                Self::Hevc => NvencCodec::Hevc,
                Self::Av1 => NvencCodec::Av1,
            }
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct DemoOptions {
        nvenc_device: usize,
        codec_override: Option<DemoCodec>,
    }

    #[derive(Debug, Clone, Copy)]
    enum DemoBackend {
        #[cfg(target_os = "macos")]
        VideoToolbox,
        #[cfg(all(
            feature = "nvenc-direct",
            any(target_os = "linux", target_os = "windows")
        ))]
        NvencDirect {
            cuda_device: usize,
            codec: DemoCodec,
        },
    }

    impl DemoOptions {
        fn parse() -> Result<Self> {
            Self::parse_from_iter(env::args().skip(1))
        }

        fn parse_from_iter<I, S>(args: I) -> Result<Self>
        where
            I: IntoIterator<Item = S>,
            S: AsRef<str>,
        {
            let mut options = Self::default();
            let mut args = args.into_iter();
            while let Some(arg) = args.next() {
                match arg.as_ref() {
                    "--nvenc-device" => {
                        let value = args
                            .next()
                            .ok_or_else(|| anyhow!("--nvenc-device requires a value"))?;
                        options.nvenc_device = value.as_ref().parse().with_context(|| {
                            format!("invalid --nvenc-device value `{}`", value.as_ref())
                        })?;
                    }
                    "--codec" => {
                        let value = args
                            .next()
                            .ok_or_else(|| anyhow!("--codec requires a value"))?;
                        options.codec_override = Some(DemoCodec::parse(value.as_ref())?);
                    }
                    "--help" | "-h" => {
                        anyhow::bail!(
                            "Usage: cargo run -p ustreamer-demo [--features nvenc-direct] [-- --nvenc-device <ordinal> --codec <hevc|av1>]"
                        );
                    }
                    other => {
                        anyhow::bail!(
                            "unrecognized argument `{other}`; supported: --nvenc-device <ordinal>, --codec <hevc|av1>"
                        );
                    }
                }
            }
            Ok(options)
        }
    }

    impl DemoBackend {
        fn select(options: DemoOptions) -> Result<Self> {
            #[cfg(target_os = "macos")]
            {
                if options.codec_override == Some(DemoCodec::Av1) {
                    anyhow::bail!(
                        "ustreamer-demo on macOS currently supports `--codec hevc` only; AV1 is not wired into the VideoToolbox demo path yet"
                    );
                }
                return Ok(Self::VideoToolbox);
            }
            #[cfg(all(
                feature = "nvenc-direct",
                any(target_os = "linux", target_os = "windows")
            ))]
            {
                let supported_codecs = NvencEncoder::supported_codecs_for_cuda_device(
                    options.nvenc_device,
                    NvencInputFormat::Bgra8,
                )
                .map_err(|error| {
                    anyhow!(
                        "failed to probe NVENC codec support on CUDA device {}: {error}",
                        options.nvenc_device
                    )
                })?;
                let codec = resolve_demo_codec(options.codec_override, &supported_codecs).map_err(
                    |error| {
                        anyhow!(
                            "failed to select NVENC codec for CUDA device {}: {error}",
                            options.nvenc_device
                        )
                    },
                )?;
                if let Some(requested_codec) = options.codec_override {
                    info!(
                        "Using requested NVENC codec {} on CUDA device {}.",
                        requested_codec.display_name(),
                        options.nvenc_device
                    );
                } else {
                    info!(
                        "Auto-selected NVENC codec {} on CUDA device {} (supported: {}).",
                        codec.display_name(),
                        options.nvenc_device,
                        format_demo_codecs(&supported_codecs)
                    );
                }
                return Ok(Self::NvencDirect {
                    cuda_device: options.nvenc_device,
                    codec,
                });
            }
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                anyhow::bail!(
                    "ustreamer-demo on {} requires the `nvenc-direct` feature; run `cargo run -p ustreamer-demo --features nvenc-direct -- --nvenc-device {} --codec {}`",
                    env::consts::OS,
                    options.nvenc_device,
                    options.codec_override.unwrap_or(DemoCodec::Hevc).arg_name()
                );
            }
            #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
            anyhow::bail!("ustreamer-demo is not supported on {}", env::consts::OS);
        }

        fn description(self) -> String {
            match self {
                #[cfg(target_os = "macos")]
                Self::VideoToolbox => "VideoToolbox HEVC + staging capture".into(),
                #[cfg(all(
                    feature = "nvenc-direct",
                    any(target_os = "linux", target_os = "windows")
                ))]
                Self::NvencDirect { cuda_device, codec } => format!(
                    "direct NVENC {} + Vulkan external capture (cuda_device={cuda_device})",
                    codec.display_name()
                ),
            }
        }

        fn renderer_backends(self) -> wgpu::Backends {
            match self {
                #[cfg(target_os = "macos")]
                Self::VideoToolbox => wgpu::Backends::PRIMARY,
                #[cfg(all(
                    feature = "nvenc-direct",
                    any(target_os = "linux", target_os = "windows")
                ))]
                Self::NvencDirect { .. } => wgpu::Backends::VULKAN,
            }
        }

        fn create_capture(self) -> Box<dyn FrameCapture> {
            match self {
                #[cfg(target_os = "macos")]
                Self::VideoToolbox => Box::new(StagingCapture::new(3)),
                #[cfg(all(
                    feature = "nvenc-direct",
                    any(target_os = "linux", target_os = "windows")
                ))]
                Self::NvencDirect { .. } => Box::new(VulkanExternalCapture::new()),
            }
        }

        fn create_encoder(self) -> Result<Box<dyn FrameEncoder>> {
            match self {
                #[cfg(target_os = "macos")]
                Self::VideoToolbox => Ok(Box::new(VideoToolboxEncoder::new())),
                #[cfg(all(feature = "nvenc-direct", any(target_os = "linux", target_os = "windows")))]
                Self::NvencDirect { cuda_device, codec } => NvencEncoder::with_config_and_cuda_device(
                    NvencEncoderConfig {
                        codec: codec.nvenc_codec(),
                        ..Default::default()
                    },
                    cuda_device,
                )
                    .map(|encoder| Box::new(encoder) as Box<dyn FrameEncoder>)
                    .map_err(|error| {
                        anyhow!(
                            "failed to create direct NVENC {} encoder on CUDA device {cuda_device}: {error}",
                            codec.display_name()
                        )
                    }),
            }
        }
    }

    fn resolve_demo_codec(
        requested_codec: Option<DemoCodec>,
        supported_codecs: &[DemoCodec],
    ) -> Result<DemoCodec> {
        if let Some(codec) = requested_codec {
            if supported_codecs.contains(&codec) {
                return Ok(codec);
            }
            anyhow::bail!(
                "requested codec `{}` is not supported by the detected NVENC device; supported codecs: {}",
                codec.arg_name(),
                format_demo_codecs(supported_codecs)
            );
        }

        if supported_codecs.contains(&DemoCodec::Av1) {
            return Ok(DemoCodec::Av1);
        }
        if supported_codecs.contains(&DemoCodec::Hevc) {
            return Ok(DemoCodec::Hevc);
        }
        anyhow::bail!(
            "the detected NVENC device did not report usable HEVC or AV1 support for BGRA input"
        );
    }

    fn format_demo_codecs(codecs: &[DemoCodec]) -> String {
        if codecs.is_empty() {
            return "none".into();
        }
        codecs
            .iter()
            .map(|codec| codec.arg_name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn run() -> Result<()> {
        let options = DemoOptions::parse().context("failed to parse demo options")?;
        let backend = DemoBackend::select(options)?;
        let endpoints = LocalStreamEndpoints::default();
        let stream_addr = endpoints.stream;
        let http_addr = endpoints.http;

        let _http_thread = spawn_http_server(http_addr)?;
        let runtime = Runtime::new().context("failed to create tokio runtime")?;

        let sessions = Arc::new(Mutex::new(Vec::<SessionSlot>::new()));
        let session_counter = Arc::new(AtomicU64::new(1));
        let (input_tx, input_rx) = mpsc::channel();

        {
            let sessions = Arc::clone(&sessions);
            let session_counter = Arc::clone(&session_counter);
            runtime.spawn(async move {
                if let Err(error) =
                    accept_websocket_sessions(stream_addr, sessions, session_counter, input_tx)
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
            .block_on(HeadlessRenderer::new(
                initial_params.width,
                initial_params.height,
                backend.renderer_backends(),
            ))
            .context("failed to create headless renderer")?;
        let mut capture = backend.create_capture();
        let mut encoder = backend.create_encoder()?;
        let mut scene = DemoScene::default();
        let mut frame_id = 0u32;
        let mut configured_generations = HashSet::new();
        let mut last_announced_dimensions = None::<(u32, u32)>;
        let mut last_decoder_message = None::<Vec<u8>>;
        let mut last_metrics_sent = Instant::now();
        let start_time = Instant::now();

        info!("Headless demo ready.");
        info!("Using {}.", backend.description());
        info!(
            "Open http://127.0.0.1:{}/ in Chrome/Chromium.",
            endpoints.http.port()
        );
        info!(
            "WebSocket stream endpoint: ws://127.0.0.1:{}/stream",
            endpoints.stream.port()
        );
        info!(
            "Controls: drag to interact, wheel to shift hue, keys 1-4 switch drag mode, R resets."
        );

        loop {
            let frame_started_at = Instant::now();
            let pending_status = drain_mapped_input_events(&input_rx, &mut quality, &mut scene);
            let mut active_sessions = snapshot_sessions(&sessions);
            if active_sessions.is_empty() {
                configured_generations.clear();
                thread::sleep(IDLE_POLL_INTERVAL);
                continue;
            }

            configured_generations.retain(|generation| {
                active_sessions
                    .iter()
                    .any(|session| session.generation == *generation)
            });

            let mut params = quality.frame_params();
            let target_dimensions = (params.width, params.height);
            let has_unconfigured_sessions = active_sessions
                .iter()
                .any(|session| !configured_generations.contains(&session.generation));
            if has_unconfigured_sessions || last_announced_dimensions != Some(target_dimensions) {
                params.force_keyframe = true;
            }
            renderer
                .ensure_size(params.width, params.height)
                .context("failed to resize headless renderer")?;
            renderer
                .render(start_time.elapsed().as_secs_f32(), &scene)
                .context("failed to render demo frame")?;

            let frame_source = renderer.stream_frame_source();
            let captured_frame = capture
                .capture(
                    frame_source.instance,
                    frame_source.device,
                    frame_source.queue,
                    frame_source.texture,
                )
                .context("failed to capture rendered frame")?;
            let frame_checksum = captured_frame
                .diagnostic_checksum()
                .context("failed to compute diagnostic checksum")?;
            let encoded_frame = encoder
                .encode(&captured_frame, &params)
                .context("failed to encode captured frame")?;

            if params.force_keyframe && !encoded_frame.is_keyframe {
                warn!("encoder ignored forced keyframe request for the current frame");
            }

            let decoder_config = encoder
                .decoder_config()
                .ok_or_else(|| anyhow!("encoder did not expose decoder config yet"))?;
            let decoder_message = decoder_config.to_control_message_bytes();
            if last_decoder_message.as_ref() != Some(&decoder_message) {
                configured_generations.clear();
                last_decoder_message = Some(decoder_message.clone());
            }
            last_announced_dimensions =
                Some((decoder_config.coded_width, decoder_config.coded_height));

            let help_message = ControlMessage::Status(StatusMessage::new(scene.help_text()))
                .to_bytes()
                .context("failed to serialize demo help message")?;
            let init_failures = initialize_unconfigured_sessions(
                &runtime,
                &active_sessions,
                &mut configured_generations,
                &decoder_message,
                &help_message,
            );
            if !init_failures.is_empty() {
                remove_sessions(&sessions, &init_failures);
                configured_generations.retain(|generation| !init_failures.contains(generation));
                active_sessions.retain(|session| !init_failures.contains(&session.generation));
            }

            if active_sessions.is_empty() {
                continue;
            }

            let mut broadcast_sessions: Vec<SessionSlot> = active_sessions
                .iter()
                .filter(|session| configured_generations.contains(&session.generation))
                .cloned()
                .collect();
            if broadcast_sessions.is_empty() {
                continue;
            }

            if let Some(status) = pending_status {
                let status_message = ControlMessage::Status(StatusMessage::new(status))
                    .to_bytes()
                    .context("failed to serialize status message")?;
                let status_failures = send_control_messages_to_sessions(
                    &runtime,
                    &broadcast_sessions,
                    &[status_message],
                );
                if !status_failures.is_empty() {
                    remove_sessions(&sessions, &status_failures);
                    configured_generations
                        .retain(|generation| !status_failures.contains(generation));
                    broadcast_sessions
                        .retain(|session| !status_failures.contains(&session.generation));
                }
                if broadcast_sessions.is_empty() {
                    continue;
                }
            }

            let timestamp_us = start_time.elapsed().as_micros().min(u64::MAX as u128) as u64;
            let current_frame_id = frame_id;
            let packets = packetize_frame(
                current_frame_id,
                timestamp_us,
                encoded_frame.is_keyframe,
                encoded_frame.is_refine,
                encoded_frame.is_lossless,
                &encoded_frame.data,
            );
            frame_id = frame_id.wrapping_add(1);

            let mut pre_frame_control_messages = Vec::new();
            if encoded_frame.is_refine || encoded_frame.is_lossless {
                if let Some(checksum) = frame_checksum {
                    pre_frame_control_messages.push(
                        ControlMessage::FrameChecksum(
                            FrameChecksumMessage::rgba8_fnv1a64(
                                current_frame_id,
                                checksum.hex_string(),
                            )
                            .with_dimensions(checksum.width, checksum.height),
                        )
                        .to_bytes()
                        .context("failed to serialize frame checksum")?,
                    );
                }
            }

            let mut post_frame_control_messages = Vec::new();
            if last_metrics_sent.elapsed() >= METRICS_INTERVAL {
                post_frame_control_messages.push(
                    ControlMessage::SessionMetrics(
                        SessionMetricsMessage::new()
                            .with_encode_time_us(encoded_frame.encode_time_us),
                    )
                    .to_bytes()
                    .context("failed to serialize session metrics")?,
                );
                last_metrics_sent = Instant::now();
            }

            let send_failures = send_frame_batches_to_sessions(
                &runtime,
                &broadcast_sessions,
                &pre_frame_control_messages,
                &packets,
                &post_frame_control_messages,
            );
            if !send_failures.is_empty() {
                remove_sessions(&sessions, &send_failures);
                configured_generations.retain(|generation| !send_failures.contains(generation));
            }

            let target_frame_time = Duration::from_secs_f64(1.0 / params.target_fps.max(1) as f64);
            let elapsed = frame_started_at.elapsed();
            if elapsed < target_frame_time {
                thread::sleep(target_frame_time - elapsed);
            }
        }
    }

    async fn accept_websocket_sessions(
        bind_address: SocketAddr,
        sessions: SessionRegistry,
        session_counter: Arc<AtomicU64>,
        input_tx: mpsc::Sender<InputEvent>,
    ) -> Result<()> {
        let server = WebSocketServer::bind(bind_address)
            .await
            .context("failed to bind websocket server")?;
        info!(
            "Listening for browser sessions on ws://{}/stream",
            server.local_addr()?
        );

        loop {
            let accepted = server
                .accept_session()
                .await
                .context("failed to accept websocket session")?;
            if accepted.path != "/stream" {
                warn!(
                    "accepting websocket session on unexpected path {}",
                    accepted.path
                );
            }

            let generation = session_counter.fetch_add(1, Ordering::Relaxed);
            let active_count = add_session(
                &sessions,
                SessionSlot {
                    generation,
                    session: accepted.session.clone(),
                },
            );
            info!(
                "browser session {generation} connected from {} ({active_count} active)",
                accepted.session.remote_address()
            );

            let sessions = Arc::clone(&sessions);
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

                let active_count = remove_session(&sessions, generation);
                info!("browser session {generation} disconnected ({active_count} active)");
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

    fn snapshot_sessions(sessions: &SessionRegistry) -> Vec<SessionSlot> {
        sessions.lock().expect("session mutex poisoned").clone()
    }

    fn add_session(sessions: &SessionRegistry, session: SessionSlot) -> usize {
        let mut slot = sessions.lock().expect("session mutex poisoned");
        slot.push(session);
        slot.len()
    }

    fn remove_session(sessions: &SessionRegistry, generation: u64) -> usize {
        let mut slot = sessions.lock().expect("session mutex poisoned");
        slot.retain(|session| session.generation != generation);
        slot.len()
    }

    fn remove_sessions(sessions: &SessionRegistry, failed_generations: &HashSet<u64>) -> usize {
        let mut slot = sessions.lock().expect("session mutex poisoned");
        slot.retain(|session| !failed_generations.contains(&session.generation));
        slot.len()
    }

    fn initialize_unconfigured_sessions(
        runtime: &Runtime,
        sessions: &[SessionSlot],
        configured_generations: &mut HashSet<u64>,
        decoder_message: &[u8],
        help_message: &[u8],
    ) -> HashSet<u64> {
        let mut failed_generations = HashSet::new();
        for session in sessions {
            if configured_generations.contains(&session.generation) {
                continue;
            }

            let session_clone = session.session.clone();
            let send_result: Result<()> = runtime.block_on(async {
                session_clone
                    .send_control_message(decoder_message)
                    .await
                    .context("failed to send decoder config")?;
                session_clone
                    .send_control_message(help_message)
                    .await
                    .context("failed to send demo help message")?;
                Ok(())
            });

            match send_result {
                Ok(()) => {
                    configured_generations.insert(session.generation);
                }
                Err(error) => {
                    warn!(
                        "failed to initialize browser session {}: {error:#}",
                        session.generation
                    );
                    failed_generations.insert(session.generation);
                }
            }
        }

        failed_generations
    }

    fn send_control_messages_to_sessions(
        runtime: &Runtime,
        sessions: &[SessionSlot],
        messages: &[Vec<u8>],
    ) -> HashSet<u64> {
        let mut failed_generations = HashSet::new();
        for session in sessions {
            let session_clone = session.session.clone();
            let send_result: Result<()> = runtime.block_on(async {
                for message in messages {
                    session_clone
                        .send_control_message(message)
                        .await
                        .context("failed to send control message")?;
                }
                Ok(())
            });

            if let Err(error) = send_result {
                warn!(
                    "failed to send control messages to browser session {}: {error:#}",
                    session.generation
                );
                failed_generations.insert(session.generation);
            }
        }

        failed_generations
    }

    fn send_frame_batches_to_sessions(
        runtime: &Runtime,
        sessions: &[SessionSlot],
        pre_frame_control_messages: &[Vec<u8>],
        packets: &[FramePacket],
        post_frame_control_messages: &[Vec<u8>],
    ) -> HashSet<u64> {
        let mut failed_generations = HashSet::new();
        for session in sessions {
            let session_clone = session.session.clone();
            let send_result: Result<()> = runtime.block_on(async {
                for message in pre_frame_control_messages {
                    session_clone
                        .send_control_message(message)
                        .await
                        .context("failed to send pre-frame control message")?;
                }
                session_clone
                    .send_frame_packets(packets)
                    .await
                    .context("failed to send frame packets")?;
                for message in post_frame_control_messages {
                    session_clone
                        .send_control_message(message)
                        .await
                        .context("failed to send post-frame control message")?;
                }
                Ok(())
            });

            if let Err(error) = send_result {
                warn!(
                    "stream send failed for browser session {}: {error:#}",
                    session.generation
                );
                failed_generations.insert(session.generation);
            }
        }

        failed_generations
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

    impl AppActionSink for DemoScene {
        fn apply_app_action(&mut self, action: AppAction) -> Option<String> {
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
            None
        }
    }

    impl MappedInputApp for DemoScene {
        fn input_mapper(&mut self) -> &mut InputMapper {
            &mut self.input_mapper
        }

        fn handle_input_event(&mut self, event: &InputEvent) -> Option<String> {
            match event {
                InputEvent::KeyDown { code } if *code == b'1' as u16 => {
                    self.input_mapper.set_mode(InteractionMode::Rotate);
                    self.mode_value = 0.0;
                    Some("Mode 1: rotate".into())
                }
                InputEvent::KeyDown { code } if *code == b'2' as u16 => {
                    self.input_mapper.set_mode(InteractionMode::Pan);
                    self.mode_value = 1.0;
                    Some("Mode 2: pan".into())
                }
                InputEvent::KeyDown { code } if *code == b'3' as u16 => {
                    self.input_mapper.set_mode(InteractionMode::Zoom);
                    self.mode_value = 2.0;
                    Some("Mode 3: zoom".into())
                }
                InputEvent::KeyDown { code } if *code == b'4' as u16 => {
                    self.input_mapper.set_mode(InteractionMode::DragAdjust);
                    self.mode_value = 3.0;
                    Some("Mode 4: drag adjust".into())
                }
                InputEvent::KeyDown { code } if *code == b'R' as u16 => {
                    *self = Self::default();
                    Some("Scene reset".into())
                }
                _ => None,
            }
        }
    }

    struct HeadlessRenderer {
        instance: wgpu::Instance,
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
        async fn new(width: u32, height: u32, backends: wgpu::Backends) -> Result<Self> {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends,
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .context("failed to request headless wgpu adapter")?;
            let adapter_info = adapter.get_info();
            info!(
                "Using wgpu adapter: {} (backend={:?})",
                adapter_info.name, adapter_info.backend
            );
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
                                NonZeroU64::new(std::mem::size_of::<SceneUniform>() as u64)
                                    .unwrap(),
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
                instance,
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

        fn instance(&self) -> &wgpu::Instance {
            &self.instance
        }

        fn queue(&self) -> &wgpu::Queue {
            &self.queue
        }

        fn texture(&self) -> &wgpu::Texture {
            &self.texture
        }
    }

    impl StreamFrameProvider for HeadlessRenderer {
        fn stream_frame_source(&self) -> ustreamer_app::StreamFrameSource<'_> {
            ustreamer_app::StreamFrameSource {
                instance: self.instance(),
                device: self.device(),
                queue: self.queue(),
                texture: self.texture(),
            }
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

  // `screen` comes from raster UVs (bottom-up Y), while `scene.pointer` comes
  // from browser coordinates (top-down Y). Convert each from its own origin.
  let pointer = vec2<f32>(scene.pointer.x * 2.0 - 1.0, scene.pointer.y * 2.0 - 1.0);
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

    #[cfg(test)]
    mod tests {
        use super::{DEMO_SHADER, DemoCodec, DemoOptions, format_demo_codecs, resolve_demo_codec};

        #[test]
        fn parses_default_demo_options() {
            let options = DemoOptions::parse_from_iter(std::iter::empty::<&str>()).unwrap();
            assert_eq!(options.nvenc_device, 0);
            assert_eq!(options.codec_override, None);
        }

        #[test]
        fn parses_nvenc_device_flag() {
            let options = DemoOptions::parse_from_iter(["--nvenc-device", "3"]).unwrap();
            assert_eq!(options.nvenc_device, 3);
        }

        #[test]
        fn parses_codec_flag() {
            let options = DemoOptions::parse_from_iter(["--codec", "av1"]).unwrap();
            assert_eq!(options.codec_override, Some(DemoCodec::Av1));
        }

        #[test]
        fn auto_selects_av1_when_supported() {
            let codec = resolve_demo_codec(None, &[DemoCodec::Hevc, DemoCodec::Av1]).unwrap();
            assert_eq!(codec, DemoCodec::Av1);
        }

        #[test]
        fn auto_falls_back_to_hevc_when_av1_is_unavailable() {
            let codec = resolve_demo_codec(None, &[DemoCodec::Hevc]).unwrap();
            assert_eq!(codec, DemoCodec::Hevc);
        }

        #[test]
        fn explicit_codec_override_must_be_supported() {
            let error = resolve_demo_codec(Some(DemoCodec::Av1), &[DemoCodec::Hevc]).unwrap_err();
            assert!(error.to_string().contains("requested codec `av1`"));
            assert!(error.to_string().contains("hevc"));
        }

        #[test]
        fn formats_supported_codecs_for_logs() {
            assert_eq!(format_demo_codecs(&[]), "none");
            assert_eq!(
                format_demo_codecs(&[DemoCodec::Av1, DemoCodec::Hevc]),
                "av1, hevc"
            );
        }

        #[test]
        fn rejects_unknown_demo_flag() {
            let error = DemoOptions::parse_from_iter(["--bogus"]).unwrap_err();
            assert!(error.to_string().contains("unrecognized argument"));
        }

        #[test]
        fn shader_maps_browser_pointer_y_without_extra_inversion() {
            assert!(DEMO_SHADER.contains(
                "let pointer = vec2<f32>(scene.pointer.x * 2.0 - 1.0, scene.pointer.y * 2.0 - 1.0);"
            ));
            assert!(!DEMO_SHADER.contains(
                "let pointer = vec2<f32>(scene.pointer.x * 2.0 - 1.0, 1.0 - scene.pointer.y * 2.0);"
            ));
        }
    }
}
