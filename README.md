# ultra-streamer-rs

Stream any Rust/wgpu application to a web browser with GPU hardware encoding and sub-30ms latency.

Turn your native wgpu render loop into a remotely-accessible, interactive web application — with zero changes to your rendering code. Ultra-streamer captures GPU frames, encodes them with platform-native hardware encoders, and streams to a browser client over QUIC/WebTransport, with a WebSocket fallback path when WebTransport is unavailable.

The first fast path is now implemented on macOS: `wgpu` Metal texture → IOSurface-backed `CVPixelBuffer` → VideoToolbox HEVC, with extracted `hvcC` decoder configuration for browser-side WebCodecs setup.

The first Vulkan/NVIDIA capture slice beyond scaffolding is also in place behind `ustreamer-capture`'s `vulkan-external` feature: `VulkanExternalCapture` now allocates an exportable Vulkan image, wraps it back into `wgpu`, copies into it with normal `wgpu` commands, exports a platform external-memory handle (`OPAQUE_FD` on Linux, `OPAQUE_WIN32` on Windows), and tags each captured frame with an explicit synchronization contract. The default mode remains conservatively `HostSynchronized`, and there is now an opt-in exported timeline semaphore mode that host-signals an external semaphore handle after the capture wait so the CUDA side can exercise a real semaphore import/wait path.

The first encode-side NVENC groundwork is also feature-gated in `ustreamer-encode` behind `nvenc-direct`: `NvencEncoder` now validates exported Vulkan frames, translates them into explicit external-memory/resource-rate-control descriptors, carries explicit sync descriptors, imports both Linux `OPAQUE_FD` and Windows `OPAQUE_WIN32` exports into CUDA device memory, and includes a project-local CUDA external-semaphore shim for exported sync handles. The current CUDA import path now uses an explicit dedicated-allocation import descriptor rather than `cudarc`'s generic helper so Vulkan's dedicated image allocation contract is preserved. On Windows, exported Vulkan memory/semaphore handles also request explicit read/write access rights for CUDA interop. A fully GPU-driven Vulkan→CUDA handoff plus actual NVENC session/bitstream output are still pending.

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
├── ustreamer-nvenc-probe # Windows Vulkan→CUDA/NVENC validation harness
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
- **Feature-gated Vulkan external-memory export path** — allocates exportable Vulkan images, wraps them back into `wgpu`, exports `OPAQUE_FD` (Linux) or `OPAQUE_WIN32` (Windows) handles, and supports both conservative host-sync and opt-in exported-timeline-semaphore handoff modes
- **Feature-gated CUDA import + NVENC descriptor-prep backend** — validates exported Vulkan frames, builds direct-NVENC import/rate-control/sync descriptors, and imports Linux `OPAQUE_FD` plus Windows `OPAQUE_WIN32` exports into CUDA device memory with explicit dedicated-allocation handling
- **Windows NVENC probe binary** — forces Vulkan, uploads a known test texture, validates `HostSynchronized` plus optional exported-timeline-semaphore capture, checks CUDA import/wait, and confirms the current encode placeholder boundary
- **WebTransport + WebCodecs** for lowest possible browser delivery latency
- **WebSocket fallback transport** for browsers or environments without WebTransport
- **Settle refinement groundwork** — quality-controller mode switching and forced keyframes on idle refine
- **Adaptive quality** — requested tier capped by RTT/loss feedback with upgrade/downgrade hysteresis
- **Settle/refine propagation** — idle settle forces a higher-bitrate keyframe refine with explicit `refine` vs `lossless` signaling
- **Compact binary input protocol** — sub-millisecond input event delivery
- **Typed control protocol** — shared decoder-config / status / session-metrics / frame-checksum JSON messages
- **Browser metrics HUD** — decode timing, frame drops, connection mode, server-fed RTT/encode telemetry, and checksum verification status
- **Headless live-test demo** — offscreen `wgpu` renderer streamed to the bundled browser client on macOS
- **Multi-client demo broadcast** — multiple browser viewers can attach to the same headless demo stream simultaneously

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
- diagnostic frame-checksum verification in the browser HUD
- multiple simultaneous WebSocket viewers receiving the same live stream

On macOS, VideoToolbox HEVC refine frames are currently **high-bitrate visually-lossless settle frames**, not bit-exact lossless frames. The protocol now distinguishes generic `refine` frames from true `lossless-refine` frames so future NVENC/software backends can advertise real lossless output honestly. The new checksum path makes that visible in the browser HUD today: true lossless backends should verify cleanly, while the current VideoToolbox refine path is expected to report mismatches.

The current multi-client demo path uses a **shared encoded stream** broadcast to all connected viewers. That is enough for LAN demos and smoke tests; truly independent per-client encoder instances remain a future scaling step if different bitrate ladders or codec negotiation per viewer become necessary.

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

## Windows NVENC Probe

On a Windows machine with an NVIDIA GPU, you can now validate the current direct path up to CUDA import with:

```bash
cargo run -p ustreamer-nvenc-probe -- --sync-mode both
```

Useful options:

- `--cuda-device 0` to pick a different CUDA ordinal
- `--width 1920 --height 1080` to change the probe texture size
- `--sync-mode host|timeline|both` to isolate the conservative handoff or the exported timeline semaphore path
- `--skip-encode-boundary-check` if you only want capture + CUDA import

What success means today:

- `HostSynchronized` passing means Vulkan external-memory export and CUDA import are working on that machine.
- `ExportedTimelineSemaphore` passing means the current host-signaled external semaphore path is also working.
- The final `encode()` check is expected to stop at the current placeholder boundary after CUDA import succeeds; full NVENC session creation, resource registration, and bitstream output are still pending.

If the probe says a non-NVIDIA adapter was selected, force the process onto the RTX GPU in Windows graphics settings. If timeline mode is reported as skipped, the current `wgpu` Vulkan device did not enable `VK_KHR_external_semaphore_win32`, so only the conservative host-sync path is available on that machine right now.
