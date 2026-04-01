//! Adaptive quality controller for the ultra-streamer pipeline.
//!
//! Handles:
//! - Interaction idle detection → lossless refinement on settle
//! - Network quality monitoring → bitrate/resolution tier switching
//! - Framerate reduction during idle periods

use std::cmp::Ordering;
use std::time::{Duration, Instant};
use ustreamer_proto::quality::{EncodeMode, EncodeParams, QualityTier};

/// Network feedback used to cap the currently requested quality tier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkMetrics {
    /// Current smoothed round-trip time for the transport session.
    pub rtt: Duration,
    /// Estimated packet loss ratio in the range `[0.0, 1.0]`, if available.
    pub packet_loss_ratio: Option<f32>,
}

impl NetworkMetrics {
    /// Create a new metrics sample with RTT only.
    pub fn new(rtt: Duration) -> Self {
        Self {
            rtt,
            packet_loss_ratio: None,
        }
    }

    /// Attach an estimated packet loss ratio to the sample.
    pub fn with_packet_loss(mut self, packet_loss_ratio: f32) -> Self {
        self.packet_loss_ratio = Some(packet_loss_ratio.clamp(0.0, 1.0));
        self
    }
}

/// Thresholds for idle detection and quality transitions.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Time after last input before sending a single refine frame.
    pub settle_timeout: Duration,
    /// Time after last input before reducing framerate.
    pub idle_timeout: Duration,
    /// Minimum FPS during idle.
    pub idle_fps: u32,
    /// Maximum RTT that still permits the `Ultra` tier.
    pub ultra_rtt_limit: Duration,
    /// Maximum RTT that still permits the `HighRes` tier.
    pub high_res_rtt_limit: Duration,
    /// Maximum RTT that still permits the `Standard` tier.
    pub standard_rtt_limit: Duration,
    /// Maximum loss ratio that still permits the `Ultra` tier.
    pub ultra_loss_limit: f32,
    /// Maximum loss ratio that still permits the `HighRes` tier.
    pub high_res_loss_limit: f32,
    /// Maximum loss ratio that still permits the `Standard` tier.
    pub standard_loss_limit: f32,
    /// Consecutive degraded samples required before capping quality downward.
    pub downgrade_hysteresis: u32,
    /// Consecutive recovered samples required before allowing one tier of recovery.
    pub upgrade_hysteresis: u32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            settle_timeout: Duration::from_millis(200),
            idle_timeout: Duration::from_secs(2),
            idle_fps: 5,
            ultra_rtt_limit: Duration::from_millis(8),
            high_res_rtt_limit: Duration::from_millis(18),
            standard_rtt_limit: Duration::from_millis(40),
            ultra_loss_limit: 0.005,
            high_res_loss_limit: 0.02,
            standard_loss_limit: 0.05,
            downgrade_hysteresis: 2,
            upgrade_hysteresis: 6,
        }
    }
}

/// Tracks interaction state and determines encoding parameters each frame.
pub struct QualityController {
    config: QualityConfig,
    last_input_time: Instant,
    requested_tier: QualityTier,
    network_cap_tier: QualityTier,
    last_network_metrics: Option<NetworkMetrics>,
    lossless_sent: bool,
    consecutive_degraded_samples: u32,
    consecutive_recovered_samples: u32,
}

impl QualityController {
    pub fn new(config: QualityConfig) -> Self {
        Self {
            config,
            last_input_time: Instant::now(),
            requested_tier: QualityTier::Standard,
            network_cap_tier: QualityTier::Ultra,
            last_network_metrics: None,
            lossless_sent: false,
            consecutive_degraded_samples: 0,
            consecutive_recovered_samples: 0,
        }
    }

    /// Call when any input event is received.
    pub fn on_input(&mut self) {
        self.last_input_time = Instant::now();
        self.lossless_sent = false;
    }

    /// Feed a transport RTT sample into the adaptive quality controller.
    pub fn on_transport_rtt(&mut self, rtt: Duration) {
        self.on_network_metrics(NetworkMetrics::new(rtt));
    }

    /// Feed a full network metrics sample into the adaptive quality controller.
    ///
    /// The controller treats `requested_tier` as the application's preferred
    /// maximum quality, and `network_cap_tier` as the highest quality the
    /// current transport conditions can safely sustain.
    pub fn on_network_metrics(&mut self, metrics: NetworkMetrics) {
        let sampled_tier = self.sampled_network_tier(metrics);
        self.last_network_metrics = Some(metrics);

        match tier_rank(sampled_tier).cmp(&tier_rank(self.network_cap_tier)) {
            Ordering::Less => {
                self.consecutive_recovered_samples = 0;
                self.consecutive_degraded_samples += 1;
                if self.consecutive_degraded_samples >= self.config.downgrade_hysteresis.max(1) {
                    self.network_cap_tier = sampled_tier;
                    self.consecutive_degraded_samples = 0;
                }
            }
            Ordering::Greater => {
                self.consecutive_degraded_samples = 0;
                self.consecutive_recovered_samples += 1;
                if self.consecutive_recovered_samples >= self.config.upgrade_hysteresis.max(1) {
                    self.network_cap_tier =
                        min_tier(step_up_tier(self.network_cap_tier), sampled_tier);
                    self.consecutive_recovered_samples = 0;
                }
            }
            Ordering::Equal => {
                self.consecutive_degraded_samples = 0;
                self.consecutive_recovered_samples = 0;
            }
        }
    }

    /// The currently requested application tier.
    pub fn requested_tier(&self) -> QualityTier {
        self.requested_tier
    }

    /// The highest tier currently allowed by recent network samples.
    pub fn network_cap_tier(&self) -> QualityTier {
        self.network_cap_tier
    }

    /// The effective encode tier after combining app preference and network cap.
    pub fn current_tier(&self) -> QualityTier {
        min_tier(self.requested_tier, self.network_cap_tier)
    }

    /// The most recent network metrics sample, if one has been provided.
    pub fn last_network_metrics(&self) -> Option<NetworkMetrics> {
        self.last_network_metrics
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

        let (width, height, fps, bitrate, max_bitrate) = match self.current_tier() {
            QualityTier::Low => (1280, 720, 30, 5_000_000, 10_000_000),
            QualityTier::Standard => (1920, 1080, 60, 15_000_000, 30_000_000),
            QualityTier::HighRes => (3840, 2160, 30, 40_000_000, 80_000_000),
            QualityTier::Ultra => (3840, 2160, 60, 80_000_000, 150_000_000),
        };

        let target_fps = match mode {
            EncodeMode::IdleLowFps => self.config.idle_fps,
            EncodeMode::LosslessRefine => fps,
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

    /// Set the application's preferred maximum tier.
    pub fn set_tier(&mut self, tier: QualityTier) {
        self.requested_tier = tier;
    }

    fn sampled_network_tier(&self, metrics: NetworkMetrics) -> QualityTier {
        let loss = metrics.packet_loss_ratio.unwrap_or(0.0).clamp(0.0, 1.0);

        if metrics.rtt > self.config.standard_rtt_limit || loss > self.config.standard_loss_limit {
            QualityTier::Low
        } else if metrics.rtt > self.config.high_res_rtt_limit
            || loss > self.config.high_res_loss_limit
        {
            QualityTier::Standard
        } else if metrics.rtt > self.config.ultra_rtt_limit
            || loss > self.config.ultra_loss_limit
        {
            QualityTier::HighRes
        } else {
            QualityTier::Ultra
        }
    }
}

fn tier_rank(tier: QualityTier) -> u8 {
    match tier {
        QualityTier::Low => 0,
        QualityTier::Standard => 1,
        QualityTier::HighRes => 2,
        QualityTier::Ultra => 3,
    }
}

fn min_tier(a: QualityTier, b: QualityTier) -> QualityTier {
    if tier_rank(a) <= tier_rank(b) { a } else { b }
}

fn step_up_tier(tier: QualityTier) -> QualityTier {
    match tier {
        QualityTier::Low => QualityTier::Standard,
        QualityTier::Standard => QualityTier::HighRes,
        QualityTier::HighRes => QualityTier::Ultra,
        QualityTier::Ultra => QualityTier::Ultra,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_standard_without_network_feedback() {
        let mut controller = QualityController::new(Default::default());

        let params = controller.frame_params();

        assert_eq!(controller.requested_tier(), QualityTier::Standard);
        assert_eq!(controller.network_cap_tier(), QualityTier::Ultra);
        assert_eq!(controller.current_tier(), QualityTier::Standard);
        assert_eq!((params.width, params.height, params.target_fps), (1920, 1080, 60));
    }

    #[test]
    fn degraded_rtt_caps_requested_ultra_tier() {
        let mut controller = QualityController::new(Default::default());
        controller.set_tier(QualityTier::Ultra);

        controller.on_transport_rtt(Duration::from_millis(20));
        assert_eq!(controller.current_tier(), QualityTier::Ultra);

        controller.on_transport_rtt(Duration::from_millis(20));
        assert_eq!(controller.network_cap_tier(), QualityTier::Standard);
        assert_eq!(controller.current_tier(), QualityTier::Standard);

        let params = controller.frame_params();
        assert_eq!((params.width, params.height, params.target_fps), (1920, 1080, 60));
    }

    #[test]
    fn recovery_happens_one_tier_at_a_time() {
        let mut controller = QualityController::new(Default::default());
        controller.set_tier(QualityTier::Ultra);

        for _ in 0..controller.config.downgrade_hysteresis {
            controller.on_network_metrics(
                NetworkMetrics::new(Duration::from_millis(1)).with_packet_loss(0.06),
            );
        }
        assert_eq!(controller.network_cap_tier(), QualityTier::Low);

        for _ in 0..controller.config.upgrade_hysteresis {
            controller.on_transport_rtt(Duration::from_millis(1));
        }
        assert_eq!(controller.network_cap_tier(), QualityTier::Standard);

        for _ in 0..controller.config.upgrade_hysteresis {
            controller.on_transport_rtt(Duration::from_millis(1));
        }
        assert_eq!(controller.network_cap_tier(), QualityTier::HighRes);
    }

    #[test]
    fn requested_tier_remains_a_hard_upper_bound() {
        let mut controller = QualityController::new(Default::default());
        controller.set_tier(QualityTier::HighRes);

        for _ in 0..controller.config.upgrade_hysteresis {
            controller.on_transport_rtt(Duration::from_millis(1));
        }

        assert_eq!(controller.network_cap_tier(), QualityTier::Ultra);
        assert_eq!(controller.current_tier(), QualityTier::HighRes);
    }

    #[test]
    fn settle_refine_forces_one_keyframe_until_next_input() {
        let mut controller = QualityController::new(Default::default());
        controller.last_input_time = Instant::now() - controller.config.settle_timeout - Duration::from_millis(1);

        let refine = controller.frame_params();
        assert_eq!(refine.mode, EncodeMode::LosslessRefine);
        assert!(refine.force_keyframe);

        let idle = controller.frame_params();
        assert_eq!(idle.mode, EncodeMode::Interactive);
        assert!(!idle.force_keyframe);

        controller.on_input();
        assert_eq!(controller.frame_params().mode, EncodeMode::Interactive);
    }
}
