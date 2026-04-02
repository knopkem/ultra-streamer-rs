/// Quality tier for adaptive streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityTier {
    /// 720p @ 30fps, 2–8 Mbps
    Low,
    /// 1080p @ 60fps, 8–20 Mbps
    Standard,
    /// 4K @ 30fps, 20–50 Mbps
    HighRes,
    /// 4K @ 60fps, 50–150 Mbps
    Ultra,
}

/// Encoding mode for the current frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeMode {
    /// Normal lossy encoding during interaction.
    Interactive,
    /// Lossless I-frame after idle settle (pixel-perfect refinement).
    LosslessRefine,
    /// Reduced framerate during idle periods.
    IdleLowFps,
}

/// Parameters for the encoder, derived from quality controller decisions.
#[derive(Debug, Clone)]
pub struct EncodeParams {
    pub width: u32,
    pub height: u32,
    pub target_fps: u32,
    pub bitrate_bps: u64,
    pub max_bitrate_bps: u64,
    pub mode: EncodeMode,
    pub force_keyframe: bool,
}

impl Default for EncodeParams {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            target_fps: 60,
            bitrate_bps: 15_000_000,
            max_bitrate_bps: 30_000_000,
            mode: EncodeMode::Interactive,
            force_keyframe: false,
        }
    }
}
