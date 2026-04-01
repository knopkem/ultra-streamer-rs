//! Adaptive quality controller for the ultra-streamer pipeline.
//!
//! Handles:
//! - Interaction idle detection → lossless refinement on settle
//! - Network quality monitoring → tier switching (720p → 1080p → 4K)
//! - Framerate reduction during idle periods

use ustreamer_proto::quality::{EncodeMode, EncodeParams, QualityTier};
use std::time::{Duration, Instant};

/// Thresholds for idle detection and quality transitions.
pub struct QualityConfig {
    /// Time after last input before sending lossless frame.
    pub settle_timeout: Duration,
    /// Time after last input before reducing framerate.
    pub idle_timeout: Duration,
    /// Minimum FPS during idle.
    pub idle_fps: u32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            settle_timeout: Duration::from_millis(200),
            idle_timeout: Duration::from_secs(2),
            idle_fps: 5,
        }
    }
}

/// Tracks interaction state and determines encoding parameters each frame.
pub struct QualityController {
    config: QualityConfig,
    last_input_time: Instant,
    current_tier: QualityTier,
    lossless_sent: bool,
}

impl QualityController {
    pub fn new(config: QualityConfig) -> Self {
        Self {
            config,
            last_input_time: Instant::now(),
            current_tier: QualityTier::Standard,
            lossless_sent: false,
        }
    }

    /// Call when any input event is received.
    pub fn on_input(&mut self) {
        self.last_input_time = Instant::now();
        self.lossless_sent = false;
    }

    /// Get the encode parameters for the current frame.
    pub fn frame_params(&mut self) -> EncodeParams {
        let idle_duration = self.last_input_time.elapsed();

        let mode = if idle_duration >= self.config.settle_timeout && !self.lossless_sent {
            self.lossless_sent = true;
            EncodeMode::LosslessRefine
        } else if idle_duration >= self.config.idle_timeout {
            EncodeMode::IdleLowFps
        } else {
            EncodeMode::Interactive
        };

        let (width, height, fps, bitrate, max_bitrate) = match self.current_tier {
            QualityTier::Low => (1280, 720, 30, 5_000_000, 10_000_000),
            QualityTier::Standard => (1920, 1080, 60, 15_000_000, 30_000_000),
            QualityTier::HighRes => (3840, 2160, 30, 40_000_000, 80_000_000),
            QualityTier::Ultra => (3840, 2160, 60, 80_000_000, 150_000_000),
        };

        let target_fps = match mode {
            EncodeMode::IdleLowFps => self.config.idle_fps,
            EncodeMode::LosslessRefine => fps, // single frame, then idle
            EncodeMode::Interactive => fps,
        };

        EncodeParams {
            width,
            height,
            target_fps,
            bitrate_bps: bitrate,
            max_bitrate_bps: max_bitrate,
            mode,
            force_keyframe: mode == EncodeMode::LosslessRefine,
        }
    }

    pub fn set_tier(&mut self, tier: QualityTier) {
        self.current_tier = tier;
    }
}
