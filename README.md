# ultra-streamer-rs

Stream any Rust/wgpu application to a web browser with GPU hardware encoding and sub-30ms latency.

Turn your native wgpu render loop into a remotely-accessible, interactive web application — with zero changes to your rendering code. Ultra-streamer captures GPU frames, encodes them with platform-native hardware encoders, and streams to a browser client over QUIC/WebTransport, with a WebSocket fallback path when WebTransport is unavailable.

The first fast path is now implemented on macOS: `wgpu` Metal texture → IOSurface-backed `CVPixelBuffer` → VideoToolbox HEVC, with extracted `hvcC` decoder configuration for browser-side WebCodecs setup.

## Use Cases

- **3D visualization** — CAD viewers, scientific visualization, game engines
- **Medical imaging** — DICOM viewers, volume renderers
- **Remote GPU workloads** — ML inference visualization, simulation dashboards
- **Cloud rendering** — serve GPU-rendered content to thin browser clients on LAN

## Architecture

See [PLAN.md](./PLAN.md) for the full architecture document.

```
crates/
├── ustreamer-capture    # GPU frame capture (zero-copy Metal/NVENC + staging fallback)
├── ustreamer-demo       # Headless macOS live-test server (wgpu + VideoToolbox + browser client)
├── ustreamer-encode     # HW video encoding (VideoToolbox, NVENC, GStreamer)
├── ustreamer-transport  # WebTransport server/session layer + WebSocket fallback
├── ustreamer-input      # Browser input → application action mapping
├── ustreamer-proto      # Shared wire protocol types (frames, input, quality, control JSON)
└── ustreamer-quality    # Adaptive quality controller (tier switching, lossless refinement)

client/                  # Browser client (WebTransport/WebSocket + WebCodecs + binary input)
```

## Target Hardware

- **Apple Silicon (M4+)** — VideoToolbox H.265 via IOSurface zero-copy
- **NVIDIA RTX 30/40/50** — NVENC H.265/AV1 via CUDA external memory
- **AMD RDNA3+** (fallback) — GStreamer with AMF/VA-API

## Key Features

- **Zero-copy frame capture** from wgpu render targets (Metal IOSurface, Vulkan/CUDA interop)
- **Hardware video encoding** at up to 4K@60fps with < 3ms encode latency
- **macOS VideoToolbox HEVC backend** with native length-prefixed access units and `hvcC` decoder-config extraction
- **WebTransport + WebCodecs** for lowest possible browser delivery latency
- **WebSocket fallback transport** for browsers or environments without WebTransport
- **Settle refinement groundwork** — quality-controller mode switching and forced keyframes on idle refine
- **Adaptive quality** — requested tier capped by RTT/loss feedback with upgrade/downgrade hysteresis
- **Settle/refine propagation** — idle settle forces a higher-bitrate keyframe refine with explicit `refine` vs `lossless` signaling
- **Compact binary input protocol** — sub-millisecond input event delivery
- **Typed control protocol** — shared decoder-config / status / session-metrics JSON messages
- **Browser metrics HUD** — decode timing, frame drops, connection mode, and server-fed RTT/encode telemetry
- **Headless live-test demo** — offscreen `wgpu` renderer streamed to the bundled browser client on macOS

## Quick Live Test

On macOS, you can already run a local smoke-test server:

```bash
cargo run -p ustreamer-demo
```

Then open `http://127.0.0.1:8090/` in Chrome/Chromium.

The demo currently exercises:

- headless `wgpu` rendering
- staging-buffer capture
- VideoToolbox HEVC encode
- WebSocket browser transport
- WebCodecs decode and interactive input round-trip
- settle/refine frame signaling in the browser HUD

On macOS, VideoToolbox HEVC refine frames are currently **high-bitrate visually-lossless settle frames**, not bit-exact lossless frames. The protocol now distinguishes generic `refine` frames from true `lossless-refine` frames so future NVENC/software backends can advertise real lossless output honestly.

Demo controls:

- drag to interact
- mouse wheel to shift hue
- `1` / `2` / `3` / `4` switch drag mode
- `R` resets the scene

## Building

```bash
cargo build
cargo test
```
