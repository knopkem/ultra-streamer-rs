//! App-facing integration traits and helpers for `ultra-streamer-rs`.
//!
//! This crate defines the public contract that an adopting `wgpu` application can
//! implement without depending on the demo binary's internal glue. The intent is
//! to keep integration explicit but small:
//!
//! - expose the current render target via [`StreamFrameProvider`]
//! - consume browser input via [`MappedInputApp`] or [`RawInputApp`]
//! - optionally observe session lifecycle events via [`SessionLifecycle`]
//! - reuse [`LocalStreamEndpoints`] for the built-in local browser/bootstrap wiring
//!
//! A minimal integration typically looks like this:
//!
//! ```ignore
//! use std::time::Duration;
//! use ustreamer_app::{AppActionSink, MappedInputApp, StreamFrameProvider, StreamFrameSource};
//! use ustreamer_input::{AppAction, InputMapper};
//! use ustreamer_proto::input::InputEvent;
//!
//! struct MyRenderer {
//!     instance: wgpu::Instance,
//!     device: wgpu::Device,
//!     queue: wgpu::Queue,
//!     texture: wgpu::Texture,
//! }
//!
//! impl StreamFrameProvider for MyRenderer {
//!     fn stream_frame_source(&self) -> StreamFrameSource<'_> {
//!         StreamFrameSource {
//!             instance: &self.instance,
//!             device: &self.device,
//!             queue: &self.queue,
//!             texture: &self.texture,
//!         }
//!     }
//! }
//!
//! struct MyScene {
//!     mapper: InputMapper,
//! }
//!
//! impl AppActionSink for MyScene {
//!     fn apply_app_action(&mut self, action: AppAction) -> Option<String> {
//!         match action {
//!             AppAction::PointerUpdate { .. } => None,
//!             _ => Some(format!("handled {action:?}")),
//!         }
//!     }
//! }
//!
//! impl MappedInputApp for MyScene {
//!     fn input_mapper(&mut self) -> &mut InputMapper {
//!         &mut self.mapper
//!     }
//!
//!     fn handle_input_event(&mut self, event: &InputEvent) -> Option<String> {
//!         let _ = event;
//!         None
//!     }
//! }
//! ```

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::mpsc::Receiver;

use ustreamer_input::{AppAction, InputMapper};
use ustreamer_proto::input::InputEvent;
use ustreamer_quality::QualityController;

/// Default local WebSocket streaming port used by the bundled browser client.
pub const DEFAULT_STREAM_PORT: u16 = 8080;

/// Default local HTTP bootstrap port used by the bundled browser client.
pub const DEFAULT_HTTP_PORT: u16 = 8090;

/// Borrowed `wgpu` objects required by capture backends.
#[derive(Debug, Clone, Copy)]
pub struct StreamFrameSource<'a> {
    pub instance: &'a wgpu::Instance,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub texture: &'a wgpu::Texture,
}

/// Trait for apps/renderers that can expose their current `wgpu` render target.
pub trait StreamFrameProvider {
    fn stream_frame_source(&self) -> StreamFrameSource<'_>;
}

/// Sink for default `ustreamer-input` [`AppAction`] values.
pub trait AppActionSink {
    fn apply_app_action(&mut self, action: AppAction) -> Option<String>;
}

/// App that wants raw input pre-processing plus the default mapped action bridge.
pub trait MappedInputApp: AppActionSink {
    fn input_mapper(&mut self) -> &mut InputMapper;

    fn handle_input_event(&mut self, _event: &InputEvent) -> Option<String> {
        None
    }
}

/// App that wants to consume raw browser input directly.
pub trait RawInputApp {
    fn handle_input_event(&mut self, event: InputEvent) -> Option<String>;
}

/// Optional lifecycle hooks for integrations that want explicit connection events.
pub trait SessionLifecycle {
    fn on_stream_ready(&mut self) {}

    fn on_viewer_connected(&mut self, _session_id: u64) {}

    fn on_viewer_disconnected(&mut self, _session_id: u64) {}
}

/// Loopback endpoints for the built-in demo/browser bootstrap flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalStreamEndpoints {
    pub stream: SocketAddr,
    pub http: SocketAddr,
}

impl LocalStreamEndpoints {
    pub fn loopback(stream_port: u16, http_port: u16) -> Self {
        let loopback = IpAddr::V4(Ipv4Addr::LOCALHOST);
        Self {
            stream: SocketAddr::new(loopback, stream_port),
            http: SocketAddr::new(loopback, http_port),
        }
    }
}

impl Default for LocalStreamEndpoints {
    fn default() -> Self {
        Self::loopback(DEFAULT_STREAM_PORT, DEFAULT_HTTP_PORT)
    }
}

/// Drain raw browser input, feed the adaptive quality controller, and apply the
/// default `InputMapper` translation for an app that uses [`MappedInputApp`].
pub fn drain_mapped_input_events<T: MappedInputApp>(
    input_rx: &Receiver<InputEvent>,
    quality: &mut QualityController,
    app: &mut T,
) -> Option<String> {
    let mut last_status = None;
    while let Ok(event) = input_rx.try_recv() {
        quality.on_input();
        if let Some(status) = app.handle_input_event(&event) {
            last_status = Some(status);
        }
        let actions = {
            let mapper = app.input_mapper();
            mapper.process(&event)
        };
        for action in actions {
            if let Some(status) = app.apply_app_action(action) {
                last_status = Some(status);
            }
        }
    }
    last_status
}

/// Drain raw browser input for an app that wants to handle `InputEvent`s directly.
pub fn drain_raw_input_events<T: RawInputApp>(
    input_rx: &Receiver<InputEvent>,
    quality: &mut QualityController,
    app: &mut T,
) -> Option<String> {
    let mut last_status = None;
    while let Ok(event) = input_rx.try_recv() {
        quality.on_input();
        if let Some(status) = app.handle_input_event(event) {
            last_status = Some(status);
        }
    }
    last_status
}

#[cfg(test)]
mod tests {
    use std::net::IpAddr;
    use std::sync::mpsc;

    use ustreamer_input::{AppAction, InputMapper};
    use ustreamer_proto::input::{InputEvent, ScrollMode};

    use super::{
        AppActionSink, DEFAULT_HTTP_PORT, DEFAULT_STREAM_PORT, LocalStreamEndpoints,
        MappedInputApp, RawInputApp, drain_mapped_input_events, drain_raw_input_events,
    };

    #[derive(Default)]
    struct TestMappedApp {
        mapper: InputMapper,
        seen_actions: Vec<AppAction>,
        raw_events: Vec<InputEvent>,
    }

    impl AppActionSink for TestMappedApp {
        fn apply_app_action(&mut self, action: AppAction) -> Option<String> {
            self.seen_actions.push(action.clone());
            match action {
                AppAction::ScrollStep { delta } => Some(format!("scroll:{delta}")),
                AppAction::PointerUpdate { .. } => Some("pointer".into()),
                _ => None,
            }
        }
    }

    impl MappedInputApp for TestMappedApp {
        fn input_mapper(&mut self) -> &mut InputMapper {
            &mut self.mapper
        }

        fn handle_input_event(&mut self, event: &InputEvent) -> Option<String> {
            self.raw_events.push(*event);
            None
        }
    }

    #[derive(Default)]
    struct TestRawApp {
        seen_events: Vec<InputEvent>,
    }

    impl RawInputApp for TestRawApp {
        fn handle_input_event(&mut self, event: InputEvent) -> Option<String> {
            self.seen_events.push(event);
            Some(format!("events:{}", self.seen_events.len()))
        }
    }

    #[test]
    fn drain_mapped_input_events_processes_raw_events_and_actions() {
        let (tx, rx) = mpsc::channel();
        tx.send(InputEvent::PointerMove {
            x: 0.25,
            y: 0.5,
            buttons: 1,
            timestamp_ms: 1,
        })
        .unwrap();
        tx.send(InputEvent::Scroll {
            delta_x: 0.0,
            delta_y: 12.0,
            mode: ScrollMode::Pixels,
        })
        .unwrap();
        drop(tx);

        let mut quality = ustreamer_quality::QualityController::new(Default::default());
        let mut app = TestMappedApp::default();
        let status = drain_mapped_input_events(&rx, &mut quality, &mut app);

        assert_eq!(status.as_deref(), Some("scroll:1"));
        assert_eq!(app.raw_events.len(), 2);
        assert!(
            app.seen_actions
                .iter()
                .any(|action| matches!(action, AppAction::PointerUpdate { .. }))
        );
        assert!(
            app.seen_actions
                .iter()
                .any(|action| matches!(action, AppAction::Rotate { .. }))
        );
        assert!(
            app.seen_actions
                .iter()
                .any(|action| matches!(action, AppAction::ScrollStep { delta: 1 }))
        );
    }

    #[test]
    fn drain_raw_input_events_reports_last_status() {
        let (tx, rx) = mpsc::channel();
        tx.send(InputEvent::KeyDown { code: b'R' as u16 }).unwrap();
        tx.send(InputEvent::KeyUp { code: b'R' as u16 }).unwrap();
        drop(tx);

        let mut quality = ustreamer_quality::QualityController::new(Default::default());
        let mut app = TestRawApp::default();
        let status = drain_raw_input_events(&rx, &mut quality, &mut app);

        assert_eq!(status.as_deref(), Some("events:2"));
        assert_eq!(app.seen_events.len(), 2);
    }

    #[test]
    fn default_local_stream_endpoints_use_loopback_ports() {
        let endpoints = LocalStreamEndpoints::default();
        assert_eq!(endpoints.stream.port(), DEFAULT_STREAM_PORT);
        assert_eq!(endpoints.http.port(), DEFAULT_HTTP_PORT);
        assert!(matches!(endpoints.stream.ip(), IpAddr::V4(addr) if addr.is_loopback()));
        assert!(matches!(endpoints.http.ip(), IpAddr::V4(addr) if addr.is_loopback()));
    }
}
