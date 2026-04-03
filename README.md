# ultra-streamer-rs

Stream any Rust/wgpu application to a web browser with GPU hardware encoding and sub-30ms LAN latency.

## Overview

Capture GPU frames from your `wgpu` render loop, encode with platform-native hardware encoders, and stream to Chrome/Chromium via WebTransport or WebSocket. The browser decodes with WebCodecs and forwards user input back to the host.

## Target Hardware

| Platform | Encoder | Capture |
|---|---|---|
| **Apple Silicon** (M1+) | VideoToolbox HEVC | Metal IOSurface zero-copy |
| **NVIDIA** (RTX 20+) | NVENC HEVC/AV1 | Vulkan external memory → CUDA |
| **AMD** (RDNA2+) | AMF HEVC | CPU staging fallback |

## Workspace

```
crates/
├── ustreamer-proto      # Wire protocol types
├── ustreamer-capture    # GPU frame capture
├── ustreamer-encode     # HW video encoding (VideoToolbox, NVENC, AMF)
├── ustreamer-transport  # WebTransport + WebSocket server
├── ustreamer-input      # Browser input mapping
├── ustreamer-quality    # Adaptive quality controller
├── ustreamer-app        # Integration traits for external apps
├── ustreamer-demo       # Headless demo server
└── ustreamer-nvenc-probe # NVENC validation harness (Windows)

client/                  # Browser client
```

## Quick Start

```bash
# macOS (VideoToolbox)
cargo run -p ustreamer-demo

# NVIDIA (direct NVENC)
cargo run -p ustreamer-demo --features nvenc-direct

# AMD (direct AMF)
cargo run -p ustreamer-demo --features amf-direct

# Auto-detect GPU vendor (recommended for mixed hardware)
cargo run -p ustreamer-demo --features nvenc-direct,amf-direct
```

Then open `http://127.0.0.1:8090/` in Chrome/Chromium.

The demo auto-detects your GPU and selects the best encoder. Build with `nvenc-direct,amf-direct` when you want one Windows/Linux binary that can choose NVIDIA or AMD at startup. Override with `--codec hevc|av1` or `--nvenc-device <n>` if needed.

## Building

```bash
cargo build
cargo test
```

## Publishing

```bash
scripts/publish-crates.sh list       # show release order
scripts/publish-crates.sh stage1 --dry-run
```

Stages: `proto`/`capture` → `input`/`quality`/`transport`/`encode` → `app`

## Architecture

See [PLAN.md](./PLAN.md) for the full design document.

## License

MIT OR Apache-2.0
