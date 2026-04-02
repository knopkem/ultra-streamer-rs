# ultra-streamer-rs

Stream any Rust/wgpu application to a web browser with GPU hardware encoding and sub-30ms LAN latency.

## Overview

Capture GPU frames from your `wgpu` render loop, encode with platform-native hardware encoders, and stream to Chrome/Chromium via WebTransport or WebSocket. The browser decodes with WebCodecs and forwards user input back to the host.

## Target Hardware

| Platform | Encoder | Capture |
|---|---|---|
| **Apple Silicon** (M1+) | VideoToolbox HEVC | Metal IOSurface zero-copy |
| **NVIDIA** (RTX 20+) | NVENC HEVC/AV1 | Vulkan external memory → CUDA |
| **AMD** (RDNA2+) | GStreamer AMF/VA-API HEVC | CPU staging fallback |

## Workspace

```
crates/
├── ustreamer-proto      # Wire protocol types
├── ustreamer-capture    # GPU frame capture
├── ustreamer-encode     # HW video encoding (VideoToolbox, NVENC, GStreamer)
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

# AMD / GStreamer fallback
cargo run -p ustreamer-demo --features gstreamer-fallback

# Auto-detect GPU vendor (recommended for mixed hardware)
cargo run -p ustreamer-demo --features nvenc-direct,gstreamer-fallback
```

On **Windows/MSVC**, `gstreamer-fallback` also needs the native **GStreamer development** install at build time, not just the runtime. Install both the official `gstreamer-1.0-msvc-x86_64-*.msi` runtime and `gstreamer-1.0-devel-msvc-x86_64-*.msi` development package, then make sure `C:\gstreamer\1.0\msvc_x86_64\bin` is on `PATH` so both `pkg-config.exe` and `gst-inspect-1.0.exe` are visible to Cargo. If you installed GStreamer somewhere else, point `PKG_CONFIG_PATH` at that tree's `lib\pkgconfig` directory. A quick sanity check before `cargo run` is:

```powershell
pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-base-1.0
gst-inspect-1.0 amfh265enc
```

Use the **MSVC** Rust toolchain with the **MSVC** GStreamer binaries; do not mix MinGW/MSYS2 GStreamer packages into the same build.

Then open `http://127.0.0.1:8090/` in Chrome/Chromium.

The demo auto-detects your GPU and selects the best encoder. Override with `--codec hevc|av1` or `--nvenc-device <n>` if needed.

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
