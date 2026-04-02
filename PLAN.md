# GPU-Encoded Application Streaming to Browser — Architecture Plan

## Problem Statement

Stream any native Rust/wgpu application to a web browser over **internal LAN** with:
- **GPU hardware encoding** — primary targets: **Apple Silicon (M4)** and **modern NVIDIA** (RTX 30/40/50 series). AMD as a plus.
- **Pixel-perfect** image fidelity (lossless ROI / validated visually-lossless)
- **Absolute best performance** — lowest latency, highest quality. Bleeding-edge browser APIs acceptable (can mandate Chrome/Chromium on LAN workstations).
- **Target latency** ≤ 16ms motion-to-photon (same machine), ≤ 30ms (LAN)
- **Resolution targets** up to 4K@60fps, with adaptive tiers

### LAN Deployment Advantages
- **No NAT traversal** — direct TCP/UDP, no STUN/TURN/ICE complexity
- **Bandwidth abundance** — 1–10 Gbps typical internal LAN → can push 100+ Mbps per stream
- **Controlled browser** — can mandate Chrome/Chromium → unlock WebTransport + WebCodecs
- **Low network jitter** — wired ethernet, predictable sub-1ms RTT

### Current Progress Snapshot
- **Implemented:** binary frame/input protocol, staging-buffer capture fallback, Rust input mapping, adaptive quality state machine
- **Implemented:** WebTransport session layer (`wtransport` over `quinn`) with tested datagram + reliable stream paths
- **Implemented:** browser client with WebTransport/WebSocket connect, frame reassembly, WebCodecs decode path, and binary input forwarding
- **Implemented:** macOS zero-copy Metal/IOSurface capture and direct VideoToolbox HEVC encode backend with WebCodecs-ready `hvcC` extraction
- **Implemented:** CPU/staging-backed VideoToolbox input path for headless smoke testing and fallback encode
- **Implemented:** adaptive quality tier capping from RTT/loss samples with recovery/degrade hysteresis
- **Implemented:** settle/refine propagation with explicit `refine` vs `lossless` frame metadata and higher-bitrate settle keyframes
- **Implemented:** WebSocket fallback server/session path with browser fallback wiring
- **Implemented:** typed control-message protocol plus browser metrics dashboard hooks for decode time, frame drops, encode time, and RTT
- **Implemented:** diagnostic frame checksums over staged CPU frames with browser-side verification HUD plumbing for refine/lossless updates
- **Implemented:** headless live-test server (`cargo run -p ustreamer-demo`) serving the browser client and streaming an offscreen `wgpu` scene over WebSocket
- **Implemented:** feature-gated Vulkan external-memory export path that allocates exportable images, wraps them back into `wgpu`, performs a normal `copy_texture_to_texture`, exports `OPAQUE_FD` handles on Linux plus `OPAQUE_WIN32` handles on Windows for future CUDA/NVENC import, and supports both default `HostSynchronized` handoff plus an opt-in exported timeline semaphore sync mode
- **Implemented:** feature-gated direct-NVENC encode path that validates exported Vulkan frames, translates them into explicit external-memory/rate-control/sync descriptors, imports Linux `OPAQUE_FD` plus Windows `OPAQUE_WIN32` exports into CUDA device memory via `cudarc`, and wires a first real NVENC session/resource-registration/bitstream-output slice
- **Implemented:** Windows-focused `ustreamer-nvenc-probe` binary that forces `wgpu` Vulkan, uploads a known test texture, exercises `HostSynchronized` plus optional exported timeline semaphore capture, validates CUDA import/wait, and confirms the current encode placeholder boundary on real NVIDIA hosts
- **Implemented:** Windows `OPAQUE_WIN32` handle export now requests explicit read/write access rights for CUDA interop, and the probe now reports missing external-semaphore-export support as a skip instead of a hard failure
- **Implemented:** direct CUDA external-memory import now uses an explicit dedicated-allocation descriptor for exported Vulkan image memory instead of relying on `cudarc`'s generic `flags = 0` helper
- **Validated:** real Windows RTX 2070 runtime probe now passes for `HostSynchronized` Vulkan external-memory export → CUDA import; `ExportedTimelineSemaphore` remains skipped because the current `wgpu` Vulkan device does not enable `VK_KHR_external_semaphore_win32`
- **Implemented:** multi-client demo broadcast over WebSocket with per-viewer initialization and forced keyframes for newly joined viewers
- **Next up:** validate the new real NVENC bitstream-output slice on the Windows RTX 2070 probe host, then replace the conservative host-side wait with a true GPU-driven handoff and add decoder-config extraction plus backend-specific true-lossless refine where supported

---

## 1. End-to-End Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  RUST HOST (Server or Local Workstation)                                 │
│                                                                          │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────────────┐   │
│  │  wgpu Render │──▶│ Frame Capture    │──▶│ HW Video Encoder        │   │
│  │  (your app    │   │ (zero-copy via   │   │ ┌─────────────────────┐ │   │
│  │  renderer)   │   │  wgpu-hal native │   │ │ Apple: VideoToolbox │ │   │
│  └─────────────┘   │  texture interop) │   │ │ NVIDIA: NVENC       │ │   │
│                     └──────────────────┘   │ │ (AMD: AMF/VA-API)   │ │   │
│                                            │ └─────────────────────┘ │   │
│                                            └────────┬────────────────┘   │
│                                                     │ H.265/AV1 NALUs  │
│  ┌──────────────────┐                               │                   │
│  │  Input Handler   │◀── QUIC datagram ─────────────┤                   │
│  │  (mouse, kbd,    │    (WebTransport)              │                   │
│  │   touch, scroll) │                                ▼                   │
│  └──────────────────┘                    ┌────────────────────────┐      │
│                                          │ WebTransport Server    │      │
│                                          │ (quinn / HTTP3+QUIC)  │      │
│                                          └────────┬───────────────┘      │
└───────────────────────────────────────────────────┼──────────────────┘   
                                                    │ LAN (< 1ms RTT)     
                              ┌──────────────────────┘                     
                              ▼                                            
┌──────────────────────────────────────────────────────────────────────────┐
│  BROWSER CLIENT (Chrome/Chromium — mandated on LAN workstations)    │
│                                                                          │
│  ┌──────────────────┐   ┌────────────────┐   ┌──────────────────────┐   │
│  │ WebCodecs Decode │──▶│ Canvas/WebGL   │──▶│ Overlay UI (native   │   │
│  │ (HW-accelerated) │   │ Presentation   │   │ HTML: toolbars,      │   │
│  └──────────────────┘   └────────────────┘   │ metadata, annot.)    │   │
│                                               └──────────────────────┘   │
│  ┌──────────────────┐                                                    │
│  │ Input Capture    │──── QUIC datagrams (fire-and-forget) ─────────▶    │
│  │ (PointerEvents,  │    + QUIC reliable stream (tool cmds)              │
│  │  KeyboardEvents) │                                                    │
│  └──────────────────┘                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. GPU Hardware Encoding — Per-Platform Analysis

### 2.1 Platform Encoder Matrix (Priority Order)

| Priority | Platform | HW Encoder | API | Codec Support | Rust Integration Path |
|----------|----------|-----------|-----|---------------|----------------------|
| **P0** | **Apple M4** | Apple Media Engine | VideoToolbox | H.264, H.265, AV1 | Direct `objc2` FFI to `VTCompressionSession` + IOSurface zero-copy |
| **P0** | **NVIDIA RTX 30/40/50** | NVENC (dedicated ASIC) | Video Codec SDK | H.264, H.265, AV1 | `nvidia-video-codec-rs` with CUDA texture interop |
| **P1** | **AMD RDNA3+** (nice-to-have) | VCN 4.0 | AMF/VA-API | H.264, H.265, AV1 | `gstreamer-rs` with `amfenc`/`vaapi` plugin |
| **Future** | **Cross-vendor** | Vulkan Video Encode | `VK_KHR_video_encode_queue` | H.264, H.265 | `ash` crate — when mature, unifies NVIDIA+AMD |

### 2.2 Recommended Encoding Strategy: Direct APIs First

Since we're optimizing for absolute best latency, **skip GStreamer** as the primary path. Go direct to each platform's native encoder API to eliminate middleware overhead:

```
┌───────────────────────────────────────────────────┐
│         Encoding Trait (Rust)                      │
│  trait FrameEncoder {                              │
│    fn encode_frame(&self,                          │
│      texture: &NativeTextureHandle,                │
│      params: &EncodeParams)                        │
│      -> Result<EncodedNALUs>;                      │
│                                                    │
│    fn encode_lossless(&self,                       │
│      texture: &NativeTextureHandle)                │
│      -> Result<EncodedNALUs>;                      │
│  }                                                 │
├───────────────────────────────────────────────────┤
│  Platform Backends (compile-time selected):        │
│  ┌───────────────────┐  ┌────────────────────┐    │
│  │ VideoToolbox       │  │ NVENC (direct)     │    │
│  │ (objc2 FFI,        │  │ (nvidia-video-     │    │
│  │  IOSurface input)  │  │  codec-rs, CUDA)   │    │
│  └───────────────────┘  └────────────────────┘    │
│           ┌────────────────────────┐               │
│           │ GStreamer (fallback,   │               │
│           │  AMD + other GPUs)    │               │
│           └────────────────────────┘               │
└───────────────────────────────────────────────────┘
```

**Why direct over GStreamer for P0:**
- Eliminates ~0.5–2ms per frame of pipeline overhead
- Direct control over encoder hardware queues and synchronization
- True zero-copy from wgpu render texture → encoder input
- Precise control over lossless mode toggling per-frame

**GStreamer retained as P1 fallback** for AMD and any future GPU vendors.

### 2.3 Codec Selection — LAN-Optimized

On a high-speed LAN (1–10 Gbps), bandwidth is abundant. This changes the calculus:

| Scenario | Codec | Bitrate (LAN) | Rationale |
|----------|-------|---------------|-----------|
| **Interactive navigation** | H.265 Main10 (or AV1 on RTX 40+/M4) | 50–150 Mbps | Extremely high bitrate → near-lossless even during motion. LAN can handle it. |
| **Settled view (lossless)** | H.265 Lossless (QP=0) | 200–500 Mbps burst | True lossless I-frame on idle. At LAN speeds, a 4K lossless frame (~5–15 MB) transfers in <2ms. |
| **10-bit HDR / wide dynamic range** | H.265 Main10 profile | Same | 10-bit encoding preserves wider dynamic range (HDR, scientific data) better than 8-bit H.264 |

**Key LAN insight:** We can afford **much higher bitrates** than internet streaming. A 4K@60fps stream at 100 Mbps uses only 1% of a 10GbE link. This means we can push quality to near-lossless during interaction and true lossless on settle, with negligible network impact.

### 2.4 Hybrid Lossy+Lossless Protocol

```
PHASE 1 — INTERACTIVE (near-lossless, low-latency)
├── User is actively scrolling/rotating/zooming
├── H.265 Main10 at very high bitrate (50–150 Mbps), QP ≈ 10–14
├── Target: 60fps, <16ms encode latency
├── On LAN, this is perceptually lossless even during motion
└── NVENC: P4 preset + ultra-low-latency tuning
    VideoToolbox: kVTCompressionPropertyKey_RealTime + high bitrate

PHASE 2 — LOSSLESS SETTLE (true lossless)
├── User stops interacting (150–300ms idle timeout)
├── Send lossless H.265 I-frame (QP=0) or lossless tile
├── At LAN speeds: 4K lossless frame ≈ 5–15 MB → <2ms transfer
├── Browser replaces lossy canvas with lossless frame
└── Checksum verification for audit trail

PHASE 3 — IDLE (bandwidth conservation)
├── User reading for >2s
├── Drop to 5fps or stop encoding entirely
├── Send delta updates only if viewport changes (e.g., cursor overlay)
└── Resume 60fps instantly on any input event
```

---

## 3. Frame Capture — Zero-Copy from wgpu to Encoder

### 3.1 Strategy: Zero-Copy First, Staging as Fallback

On LAN with best-latency goals, zero-copy is worth the unsafe complexity. Each frame saved is ~2–5ms at 4K.

#### Primary: Zero-Copy via wgpu-hal Native Interop

##### Apple M4 (Metal → VideoToolbox)
```
wgpu render → MTLTexture (via wgpu-hal) → IOSurface backing → VTCompressionSession
```
1. `texture.as_hal::<hal::api::Metal>()` → get `MTLTexture` reference
2. Extract backing `IOSurfaceRef` (Metal textures on Apple Silicon are IOSurface-backed)
3. Create `CVPixelBuffer` wrapping the IOSurface
4. Feed directly to `VTCompressionSessionEncodeFrame()`
5. **Zero CPU copies. Zero GPU copies. Encode happens on dedicated Media Engine.**

##### NVIDIA RTX (Vulkan → CUDA → NVENC)
```
wgpu render → VkImage (via wgpu-hal) → CUDA external memory import → NVENC
```
1. `texture.as_hal::<hal::api::Vulkan>()` → get `VkImage`
2. Export `VkDeviceMemory` via `vkGetMemoryFdKHR` (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`)
3. Import into CUDA via `cuExternalMemoryGetMappedBuffer` or `cuExternalMemoryGetMappedMipmappedArray`
4. Register CUDA pointer as NVENC input resource (`NV_ENC_REGISTER_RESOURCE`)
5. Encode directly — NVENC reads from GPU memory, writes NALUs to mapped output buffer
6. **Zero CPU copies. One GPU-internal copy (Vulkan→CUDA domain, hardware-fast).**

Alternatively, on Windows with DX12 backend:
```
wgpu render → ID3D12Resource (via wgpu-hal) → CreateSharedHandle → NVENC DX12 input
```

##### AMD (Vulkan → VA-API) — Nice-to-have
```
wgpu render → VkImage → Vulkan Video Encode (future) or VA-API surface import
```

#### Fallback: Triple-Buffered Staging (safe Rust, any platform)
```
wgpu Texture → copy_texture_to_buffer → map_async → CPU memory → encoder
```
- 3 rotating `MAP_READ | COPY_DST` buffers
- Encode frame N while capturing frame N+1 while rendering frame N+2
- ~2–5ms overhead at 4K, ~1ms at 1080p
- Still viable for LAN (well within latency budget)

### 3.2 Recommended Phase-In

**Phase 1:** Start with zero-copy on the platform you're developing on (likely Mac M4 — the Metal/IOSurface path is cleanest). Use staging fallback on other platforms.

**Phase 2:** Add NVENC zero-copy via Vulkan external memory.

**Phase 3:** Staging buffer becomes AMD-only fallback (or replaced by Vulkan Video Encode when mature).

---

## 4. Transport Protocol — WebTransport + WebCodecs (Primary)

### 4.1 Why WebTransport is Primary (not WebRTC)

On an internal LAN with mandated Chrome/Chromium, we can choose the lowest-latency option:

| Factor | WebTransport + WebCodecs | WebRTC |
|--------|--------------------------|--------|
| **Latency** | **30–100µs framing** (QUIC datagrams, no RTP/SRTP overhead) | 1–5ms (RTP packetize + jitter buffer) |
| **Codec control** | **Full** — choose codec, tune every parameter, feed raw NALUs | Black-box encoder/decoder |
| **NAT traversal** | Not needed (LAN!) | Built-in but unnecessary here |
| **Complexity** | More DIY, but we control everything | Simpler but opaque |
| **Browser support** | Chrome/Edge/Chromium ✅ (Safari partial, Firefox WIP) | All browsers |
| **Frame delivery** | **Unreliable datagrams** — newest frame wins, no buffering stalls | Jitter buffer adds latency |

**Verdict:** WebTransport + WebCodecs gives us **direct control** and **lower latency** at the cost of Chrome-only — acceptable on an internal LAN.

### 4.2 Transport Architecture

```
Server (Rust)                              Browser (Chrome)
─────────────                              ────────────────
quinn HTTP/3 server                        WebTransport API
  │                                          │
  ├── QUIC Datagram ──── video frames ────▶  ├── WebCodecs VideoDecoder
  │   (unreliable,                           │   (HW-accelerated decode)
  │    newest-wins)                          │
  │                                          │
  ├── QUIC Stream 0 ──── control msgs ───▶  ├── Session control
  │   (reliable,         (codec config,      │   (quality negotiation)
  │    bidirectional)     quality params)     │
  │                                          │
  ◀── QUIC Datagram ──── mouse/pointer ────  ├── Input capture
  │   (unreliable,       events              │   (PointerEvent, wheel)
  │    fire-and-forget)                      │
  │                                          │
  ◀── QUIC Stream 1 ──── keyboard/tool ────  ├── Reliable input
      (reliable)          commands               (key, tool select)
```

### 4.3 Frame Delivery — Unreliable Datagrams

This is the key performance advantage. Unlike WebRTC which retransmits lost video packets:

- **Each encoded frame is one or more QUIC datagrams** (up to ~1200 bytes each)
- If a datagram is lost, **skip it** — display the next frame instead
- On LAN with <0.01% packet loss, almost no frames are ever lost
- Eliminates jitter buffer entirely — decode and display immediately
- For frames larger than one datagram: use a short sequence number + "frame complete" marker

```rust
// Server-side frame packetization
struct FramePacket {
    frame_id: u32,       // monotonic frame counter
    fragment_idx: u16,   // 0..N for fragmented frames
    fragment_count: u16, // total fragments
    timestamp_us: u64,   // capture timestamp (microseconds)
    is_keyframe: bool,
    is_lossless: bool,   // lossless refinement frame
    payload: Vec<u8>,    // H.265 NALU fragment
}
```

### 4.4 Input Delivery — Unreliable + Reliable Channels

```javascript
// Browser → Server input protocol
// Unreliable datagrams for continuous input (mouse, scroll)
const mouseMsg = new Uint8Array(16); // compact binary
// [type:1][buttons:1][x:f32][y:f32][timestamp:u32]
transport.datagrams.writable.getWriter().write(mouseMsg);

// Reliable stream for discrete actions (key press, tool select)
const writer = reliableStream.writable.getWriter();
writer.write(new Uint8Array([CMD_KEY_DOWN, keyCode]));
```

### 4.5 Rust Server Crates

| Crate | Purpose | Notes |
|-------|---------|-------|
| `quinn` | QUIC transport foundation | Mature QUIC implementation for Rust |
| `wtransport` | WebTransport server/session layer | Wraps the HTTP/3 + WebTransport handshake on top of quinn |
| `tokio` | Async runtime | Mature async runtime for Rust |
| `axum` | HTTPS server for initial page load | Serves HTML/JS client |

### 4.6 Fallback: WebSocket + WebCodecs

For environments where WebTransport isn't available (older Chrome, other browsers):
- WebSocket carries encoded frames as binary messages
- Still uses WebCodecs for decode (not `<video>` tag)
- Higher latency (~5–10ms added from TCP head-of-line blocking) but still good on LAN
- Same codec pipeline, just different transport wrapper

---

## 5. Browser Client Architecture

### 5.1 Video Decode & Display — WebCodecs (Primary)

```javascript
// WebCodecs decoder — direct HW-accelerated decode, no jitter buffer
const decoder = new VideoDecoder({
  output: (frame) => {
    // Option A: Direct to canvas (simplest)
    ctx.drawImage(frame, 0, 0);
    frame.close();
    
    // Option B: requestVideoFrameCallback for vsync-aligned display
    // videoElement.requestVideoFrameCallback(() => { ... });
  },
  error: (e) => console.error('Decode error:', e),
});

// Configure for H.265 with hardware acceleration
decoder.configure({
  codec: 'hev1.1.6.L153.B0',       // H.265 Main10 Level 5.1
  hardwareAcceleration: 'prefer-hardware',
  optimizeForLatency: true,          // critical: disables decode buffering
});

// Receive frames from WebTransport datagrams
const reader = transport.datagrams.readable.getReader();
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  
  const packet = parseFramePacket(value);
  reassembleFrame(packet); // collect fragments → full NALU
  
  if (isFrameComplete(packet.frame_id)) {
    const chunk = new EncodedVideoChunk({
      type: packet.is_keyframe ? 'key' : 'delta',
      timestamp: packet.timestamp_us,
      data: getAssembledFrame(packet.frame_id),
    });
    decoder.decode(chunk);
  }
}
```

**Key setting: `optimizeForLatency: true`** — This tells Chrome's decoder to output frames immediately rather than buffering. Combined with unreliable datagrams, this eliminates the two biggest sources of display latency.

### 5.2 Input Capture & Forwarding — Compact Binary via QUIC Datagrams

```javascript
const canvas = document.getElementById('viewer');
const dgWriter = transport.datagrams.writable.getWriter();
const relWriter = reliableStream.writable.getWriter();

// Compact binary encoding — no JSON overhead
const INPUT_MOUSE_MOVE = 0x01, INPUT_MOUSE_DOWN = 0x02, INPUT_MOUSE_UP = 0x03;
const INPUT_SCROLL = 0x04, INPUT_KEY_DOWN = 0x10, INPUT_KEY_UP = 0x11;

function sendUnreliable(buf) { dgWriter.write(buf); }
function sendReliable(buf) { relWriter.write(buf); }

// Use PointerEvents (unified mouse/pen/touch) for future stylus support
canvas.addEventListener('pointermove', (e) => {
  const buf = new ArrayBuffer(14);
  const dv = new DataView(buf);
  dv.setUint8(0, INPUT_MOUSE_MOVE);
  dv.setUint8(1, e.buttons);
  dv.setFloat32(2, e.offsetX / canvas.width, true);   // normalized coords
  dv.setFloat32(6, e.offsetY / canvas.height, true);
  dv.setUint32(10, performance.now() | 0, true);      // timestamp for RTT measurement
  sendUnreliable(new Uint8Array(buf));
});

canvas.addEventListener('pointerdown', (e) => {
  const buf = new Uint8Array([INPUT_MOUSE_DOWN, e.button,
    ...f32ToBytes(e.offsetX / canvas.width),
    ...f32ToBytes(e.offsetY / canvas.height)]);
  sendReliable(buf);  // reliable — button state must not be lost
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const buf = new ArrayBuffer(10);
  const dv = new DataView(buf);
  dv.setUint8(0, INPUT_SCROLL);
  dv.setFloat32(1, e.deltaX, true);
  dv.setFloat32(5, e.deltaY, true);
  dv.setUint8(9, e.deltaMode);
  sendUnreliable(buf);  // scroll is continuous, loss OK
}, { passive: false });

document.addEventListener('keydown', (e) => {
  if (canvas.matches(':focus-within')) {
    e.preventDefault();
    sendReliable(new Uint8Array([INPUT_KEY_DOWN, ...encodeKeyCode(e.code)]));
  }
});
```

### 5.3 Overlay UI in Browser

For interactive applications, some UI elements should be browser-native (not streamed):
- **Toolbars, menus, settings panels** → HTML/CSS (always crisp, no encoding artifacts)
- **Status text, metadata overlays** → HTML positioned over canvas
- **Annotations / vector overlays** → SVG or Canvas 2D overlay (crisp vector rendering)
- **3D/2D viewport content** → streamed from GPU (the expensive part)

This **hybrid approach** reduces encoded area, improves perceived quality, and lowers bandwidth.

---

## 6. Lossless Refinement Protocol

### 6.1 The Hybrid Lossy/Lossless Strategy

Some applications demand pixel-perfect rendering when the user pauses to examine content. The solution is a **two-phase streaming protocol:**

```
PHASE 1 — INTERACTIVE (lossy, low-latency)
├── User is actively scrolling/rotating/zooming
├── Stream H.265/AV1 at adaptive bitrate (8–50 Mbps)
├── Target: 60fps, <50ms latency
├── Quality: "visually lossless" during motion (high QP=18–22)
└── Purpose: fluid interaction

PHASE 2 — LOSSLESS REFINEMENT (pixel-perfect)
├── User stops interacting (200–500ms idle timeout)
├── Send lossless refinement:
│   Option A: Lossless H.264/H.265 I-frame (supported on NVENC/VTB)
│   Option B: PNG/WebP lossless tile of viewport
├── Target: <500ms for full lossless update
├── Can be ROI-focused: lossless center, lossy periphery
└── Purpose: pixel-perfect for detailed examination
```

### 6.2 ROI-Aware Encoding

For maximum efficiency, use **region-of-interest encoding:**
- The application knows which region the user is examining (cursor position, zoom level)
- Allocate more bits to the ROI, fewer to periphery
- NVENC and VideoToolbox support per-region QP adjustment
- On settle, send lossless only for the ROI tile

### 6.3 Validation Requirements

- The lossless refinement frame must be **bit-exact** with what the native app renders
- Implement a verification mode: compute checksum of rendered frame, send with encoded data, browser verifies after decode
- Log compression ratios and any lossy→lossless transitions for audit trail

---

## 7. Adaptive Quality & Bitrate Control

### 7.1 Resolution/Framerate Tiers

| Tier | Resolution | FPS | Bitrate Range | Use Case | HW Requirement |
|------|-----------|-----|---------------|----------|----------------|
| **Interactive** | 1080p | 60 | 8–20 Mbps | Scrolling, rotating, zooming | Any modern GPU |
| **High-Res Interactive** | 4K | 30 | 20–50 Mbps | Detailed examination during navigation | M4 / RTX 3060+ / RX 6800+ |
| **Lossless Settle** | 4K | N/A (single frame) | Lossless (~50–200 MB uncompressed, ~5–20 MB lossless) | Pixel-perfect examination | Any GPU (CPU fallback OK) |
| **Low Bandwidth** | 720p | 30 | 2–8 Mbps | Remote/WAN access | Any GPU |

### 7.2 Adaptive Algorithm

```
on_network_stats_update(rtt, bandwidth, packet_loss):
  if bandwidth > 40 Mbps AND rtt < 20ms:
    set_tier(4K@60fps, bitrate=40Mbps)
  elif bandwidth > 15 Mbps AND rtt < 50ms:
    set_tier(1080p@60fps, bitrate=15Mbps)
  elif bandwidth > 5 Mbps:
    set_tier(1080p@30fps, bitrate=8Mbps)
  else:
    set_tier(720p@30fps, bitrate=3Mbps)

on_interaction_idle(idle_duration):
  if idle_duration > 300ms:
    send_lossless_refinement(current_viewport)
  if idle_duration > 1000ms:
    reduce_fps_to(5)  // save bandwidth when user is reading
```

### 7.3 Smart Frame Skipping

- Track **dirty regions** — if only a small toolbar animates, don't re-encode the entire frame
- Use **content-adaptive encoding** — images with large uniform areas (CAD, visualization, dashboards) compress extremely well
- During idle, use **delta encoding** — only send changed pixels (common in remote desktop)

---

## 8. Integration with Any wgpu Application

### 8.1 Architecture — How a Consumer Uses ultra-streamer

```
Your wgpu Application                          ultra-streamer-rs
─────────────────────                          ──────────────────
┌─────────────────────┐
│ Your render loop:   │
│  - scene graph      │
│  - UI (egui, etc.)  │
│  - custom shaders   │
│                     │
│ Renders to a wgpu   │──── shared render target ───▶ FrameCapture
│ Texture each frame  │                                   │
└─────────────────────┘                              FrameEncoder
                                                          │
┌─────────────────────┐                              StreamTransport
│ Your input handler  │◀── AppAction events ──────── InputBridge
│  - camera control   │                                   │
│  - tool selection   │                              QualityController
│  - app state        │                              (adaptive bitrate +
└─────────────────────┘                               lossless refinement)
```

### 8.2 Minimal Integration Example (Pseudocode)

```rust
use ustreamer_capture::staging::StagingCapture;
use ustreamer_encode::FrameEncoder;
use ustreamer_quality::QualityController;

// Your existing wgpu app
let (device, queue) = /* your wgpu setup */;
let render_texture = /* your render target */;

// Add ultra-streamer
let mut capture = StagingCapture::new(3);
let mut encoder = /* platform encoder */;
let mut quality = QualityController::new(Default::default());
let transport = /* WebTransport session */;

// Your render loop — unchanged
loop {
    your_app.render(&device, &queue, &render_texture);

    // Capture + encode + send (can run on separate thread)
    let params = quality.frame_params();
    let frame = capture.capture(&device, &queue, &render_texture)?;
    let encoded = encoder.encode(&frame, &params)?;
    transport.send_frame(&encoded).await?;

    // Process browser input
    for event in transport.recv_input() {
        let actions = input_mapper.process(&event);
        your_app.handle_actions(&actions);
        quality.on_input();
    }
}
```

### 8.3 Design Principles

- **Zero changes to your renderer** — ultra-streamer reads from your existing render target
- **No framework lock-in** — works with any wgpu app (winit, egui, custom)
- **Consumer owns the event loop** — ultra-streamer is a library, not a framework
- **Opt-in features** — use only the crates you need (e.g., capture + encode without transport)

---

## 9. Latency Budget Analysis — LAN Optimized

### 9.1 Target: ≤ 16ms Local, ≤ 30ms LAN

```
Component                    Local (ms)    LAN (ms)    Notes
─────────────────────────────────────────────────────────────────
Input capture (browser JS)      0.3          0.3       PointerEvent timestamp
Input transport (QUIC dg)       0.1          0.3–0.5   UDP datagram, no TCP HOL
Input processing (Rust)         0.2          0.2       Decode + map to event
Render (wgpu)                   2–8*         2–8*      Scene-dependent
Frame capture                   0.1**        0.1**     Zero-copy (Approach A)
                                1–3***       1–3***    Staging buffer (fallback)
Encode (HW)                     1–3          1–3       NVENC ultra-low-lat preset
Packetize + send (QUIC dg)      0.1          0.1       No RTP, no encryption needed on LAN
Network transit                  <0.1         0.3–1.0   Switched ethernet
Decode (WebCodecs HW)           1–2          1–2       optimizeForLatency: true
Canvas drawImage                 0.5          0.5       
─────────────────────────────────────────────────────────────────
TOTAL (zero-copy)              4.4–14ms     5–14ms     ✅ Within 16ms budget
TOTAL (staging fallback)       5.4–18ms     6–18ms     ✅ Within 30ms budget

*  2ms for simple slice, 8ms for complex volume render
** Zero-copy via IOSurface/CUDA interop
*** Triple-buffered staging buffer path
```

### 9.2 Key Optimization Levers (LAN-specific)

1. **No encryption overhead** — internal LAN can skip TLS for video datagrams (or use fast symmetric cipher)
2. **Skip jitter buffer entirely** — QUIC datagrams + WebCodecs `optimizeForLatency` = immediate decode
3. **No congestion control needed** — dedicated LAN with known bandwidth, send at full rate
4. **Pipeline overlap** — encode frame N while rendering frame N+1 (pipelining hides encode latency)
5. **Skip vsync on server** — render as fast as possible, not tied to 60Hz display
6. **Jumbo datagrams** — on LAN, QUIC datagrams can be larger (MTU 9000 with jumbo frames), reducing fragmentation
7. **Input coalescing** — browser sends at most 1 mouse event per rAF (16ms), reducing redundant input

---

## 10. Hardware-Specific Encoding Configuration

### 10.1 Apple M4 (VideoToolbox) — P0

```
Encoder: VTCompressionSession (direct objc2 FFI)
Codec: H.265 Main10 — best quality on Apple Silicon, native 10-bit
Input: IOSurface from MTLTexture (zero-copy)
Settings:
  kVTCompressionPropertyKey_RealTime = true
  kVTCompressionPropertyKey_AllowFrameReordering = false  // no B-frames
  kVTCompressionPropertyKey_MaxKeyFrameInterval = 60      // 1 sec at 60fps
  kVTCompressionPropertyKey_AverageBitRate = 80_000_000   // 80 Mbps (LAN)
  kVTCompressionPropertyKey_DataRateLimits = [150_000_000, 1.0]  // burst 150 Mbps
  kVTCompressionPropertyKey_ProfileLevel = kVTProfileLevel_HEVC_Main10_AutoLevel
  
Lossless mode (on settle):
  kVTCompressionPropertyKey_Lossless = true
  (single I-frame, then resume lossy)
  
AV1 alternative (M4 Pro/Max):
  Codec: AV1 Main
  Better compression → lower bitrate for same quality
  Check availability: VTIsHardwareDecodeSupported(kCMVideoCodecType_AV1)
```

### 10.2 NVIDIA RTX 30/40/50 (NVENC) — P0

```
Encoder: NvEncodeAPI via nvidia-video-codec-rs
Codec: H.265 Main10 (RTX 30+), AV1 (RTX 40+ Ada, RTX 50 Blackwell)
Input: CUDA mapped resource from Vulkan external memory (zero-copy)
Settings:
  Preset: P4 (balanced speed/quality)
  Tuning: NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY
  Rate Control: NV_ENC_PARAMS_RC_VBR with high maxBitRate
    averageBitRate = 80_000_000   // 80 Mbps
    maxBitRate     = 200_000_000  // 200 Mbps burst
  B-frames: 0 (zero latency)
  GOP: IPPP... with keyframe every 60 frames
  Async mode: encode on separate CUDA stream

Lossless mode (on settle):
  NV_ENC_PARAMS_RC_CONSTQP, qpInterP = 0, qpIntra = 0
  encodeConfig.frameIntervalP = 0  // I-frame only
  
AV1 (RTX 40+):
  Preferred over H.265 for better quality/bitrate
  Use NV_ENC_CODEC_AV1_GUID with similar settings
  Dual AV1 encoders on Ada → can encode two streams simultaneously
```

### 10.3 AMD RDNA3+ (AMF/VA-API) — P1 Nice-to-Have

```
Encoder: GStreamer with amfh265enc (Windows) or vaapih265enc (Linux)
Codec: H.265 Main, AV1 on RDNA3+
Input: Via GStreamer appsrc (staging buffer path, not zero-copy initially)
Settings: Similar bitrate/quality targets as above
Note: Lower priority — implement after Mac + NVIDIA are solid
```

---

## 11. Implementation Phases / Todos

### Phase 1: Zero-Copy Encode + WebTransport MVP (Mac M4 first)
- **capture-metal-zerocopy** *(done)*: wgpu-hal Metal texture → IOSurface extraction → CVPixelBuffer wrapper
- **encoder-videotoolbox** *(done)*: Direct VTCompressionSession FFI via objc2 — HEVC real-time encode, keyframe forcing, and `hvcC` decoder-config extraction
- **transport-webtransport** *(done)*: WebTransport session handling via `wtransport`/`quinn`, datagram send/receive, reliable stream control path
- **frame-packetizer** *(done)*: Encode NALU fragmentation into QUIC datagrams with frame ID, fragment index, keyframe flag
- **browser-client-webcodecs** *(done)*: HTML/JS client with WebTransport connect, datagram reassembly, WebCodecs decode path, metrics
- **input-capture-binary** *(done)*: Browser-side PointerEvent/KeyboardEvent → compact binary encoding → QUIC datagrams (unreliable) + stream (reliable)
- **input-bridge-rust** *(done)*: Rust-side binary input decode → map to abstract AppAction events for the consumer application

### Phase 2: NVIDIA + Quality + Lossless
- **capture-nvenc-zerocopy**: feature-gated `VulkanExternalCapture` now allocates exportable Vulkan images, re-wraps them as `wgpu::Texture`s, copies render targets into them with standard `wgpu` commands, exports Linux `OPAQUE_FD` plus Windows `OPAQUE_WIN32` handles, defaults to `HostSynchronized`, and also supports an opt-in exported timeline semaphore sync mode; the Windows path now requests explicit read/write access rights on exported Win32 handles, `ustreamer-nvenc-probe` has validated the conservative host-sync path on a real RTX 2070, and remaining work is replacing the conservative host wait with a true GPU-driven handoff
- **encoder-nvenc-direct**: feature-gated `NvencEncoder` module is now in place, can translate exported Vulkan frames into explicit direct-NVENC input/rate-control/sync descriptors, import Linux `OPAQUE_FD` plus Windows `OPAQUE_WIN32` exports into CUDA device memory with explicit dedicated-allocation handling, and validate that boundary through `ustreamer-nvenc-probe`; real Windows hardware now confirms the host-sync CUDA import path is alive, and remaining work is consuming future GPU sync handles in a real NVENC session, resource registration, and bitstream output
- **lossless-settle** *(done)*: Idle detection → forced high-bitrate refine keyframe, explicit `refine` vs `lossless` signaling on the wire; VideoToolbox remains visually-lossless only, while future backends can mark true lossless frames
- **lossless-checksum** *(done)*: Server computes diagnostic RGBA checksums for CPU-readable frames, sends them as typed control messages, and the browser verifies decoded output in the HUD; this already proves current VideoToolbox refine frames are not bit-exact, while future true-lossless backends can validate cleanly
- **adaptive-quality** *(done)*: Monitor QUIC RTT + loss → cap bitrate/resolution tier with downgrade/upgrade hysteresis, adjust framerate on idle
- **browser-overlay-hybrid**: Move toolbars/metadata/annotations to native HTML/SVG overlay, reducing encoded viewport

### Phase 3: Polish + AMD Fallback
- **encoder-gstreamer-fallback**: GStreamer-based encoder for AMD and unsupported GPUs (H.265 via amfenc/vaapih265enc)
- **capture-staging-fallback** *(done)*: Triple-buffered copy_texture_to_buffer for platforms without zero-copy interop
- **transport-ws-fallback** *(done)*: WebSocket fallback transport for non-Chrome browsers (same WebCodecs decode)
- **headless-server** *(done)*: Run viewer without window for dedicated server deployment (wgpu headless device)
- **multi-client** *(done)*: Multiple simultaneous browser sessions from one server are now supported in the demo via a shared encoded stream broadcast with per-viewer decoder initialization; fully independent per-client encoder instances remain a future scaling refinement
- **av1-support**: Enable AV1 encoding on RTX 40+/M4 Pro+ with codec negotiation at session start

### Phase 4: Advanced Optimization
- **roi-encoding**: Per-region QP: high quality at cursor/ROI, lower at periphery
- **delta-encoding**: Dirty-region tracking in renderer → only re-encode changed tiles
- **input-prediction**: Client-side scroll/rotate prediction with server correction
- **perf-dashboard** *(done)*: Real-time latency/quality metrics overlay (encode time, network RTT, decode time, frame drop rate)

---

## 12. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| wgpu-hal Metal → IOSurface extraction is undocumented | Blocks Mac zero-copy | Prototype early; Metal textures on Apple Silicon are typically IOSurface-backed. Fallback to staging buffer. |
| NVENC CUDA interop from Vulkan external memory is complex | Delays NVIDIA zero-copy | Start with staging buffer on NVIDIA; JRF63/desktop-streaming repo has working Rust reference code |
| Chrome WebTransport API still evolving | Browser API changes | Pin Chrome version on LAN workstations; WebSocket fallback ready |
| H.265 decode in Chrome requires `--enable-features=PlatformHEVCDecoderSupport` | Browser config needed | IT can set Chrome policy. Alternative: use AV1 (natively supported in Chrome) |
| WebCodecs `optimizeForLatency` behavior varies | Decode latency unpredictable | Benchmark on target hardware; worst case add manual frame pacing |
| Lossless H.265 frame size at 4K can be 5–15 MB | Momentary bandwidth spike | LAN handles it (5 MB @ 1 Gbps = 40µs). Rate-limit to one lossless frame per settle. |
| objc2 FFI to VideoToolbox is verbose/unsafe | Dev effort on Mac | Create a thin safe wrapper crate; reference apple-sys/core-video bindings |
| Rust WebTransport support is still maturing | Transport reliability | Use `wtransport` today; keep raw QUIC/WebSocket fallback options available if browser/runtime quirks appear |

---

## 13. Reference Projects & Resources

- **WebRTC-For-Desktop** (Rust): DXGI capture → NVENC → WebRTC to browser ([GitHub](https://github.com/JRF63/desktop-streaming))
- **Sunshine** (C++): Most mature game streaming server, multi-GPU encode ([GitHub](https://github.com/LizardByte/Sunshine))
- **Selkies-GStreamer**: GStreamer + WebRTC for cloud Linux desktops ([GitHub](https://github.com/selkies-project/selkies-gstreamer))
- **webrtc-rs**: Pure Rust WebRTC stack ([GitHub](https://github.com/webrtc-rs/webrtc))
- **gstreamer-rs**: GStreamer Rust bindings ([GitHub](https://github.com/sdroege/gstreamer-rs))
- **nvidia-video-codec-rs**: NVENC/NVDEC Rust bindings ([GitHub](https://github.com/rust-av/nvidia-video-codec-rs))
