# ultra-streamer-rs

Stream any Rust/wgpu application to a web browser with GPU hardware encoding and sub-30ms latency.

Turn your native wgpu render loop into a remotely-accessible, interactive web application — with zero changes to your rendering code. Ultra-streamer captures GPU frames, encodes them with platform-native hardware encoders, and streams to a browser client over QUIC/WebTransport.

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
├── ustreamer-encode     # HW video encoding (VideoToolbox, NVENC, GStreamer)
├── ustreamer-transport  # WebTransport server (QUIC datagrams via quinn)
├── ustreamer-input      # Browser input → application action mapping
├── ustreamer-proto      # Shared wire protocol types (frames, input, quality)
└── ustreamer-quality    # Adaptive quality controller (tier switching, lossless refinement)

client/                  # Browser client (HTML/JS, WebCodecs + WebTransport)
```

## Target Hardware

- **Apple Silicon (M4+)** — VideoToolbox H.265 via IOSurface zero-copy
- **NVIDIA RTX 30/40/50** — NVENC H.265/AV1 via CUDA external memory
- **AMD RDNA3+** (fallback) — GStreamer with AMF/VA-API

## Key Features

- **Zero-copy frame capture** from wgpu render targets (Metal IOSurface, Vulkan/CUDA interop)
- **Hardware video encoding** at up to 4K@60fps with < 3ms encode latency
- **WebTransport + WebCodecs** for lowest possible browser delivery latency
- **Lossless refinement** — pixel-perfect frames on interaction settle
- **Adaptive quality** — automatic tier switching based on network conditions
- **Compact binary input protocol** — sub-millisecond input event delivery

## Building

```bash
cargo build
cargo test
```
