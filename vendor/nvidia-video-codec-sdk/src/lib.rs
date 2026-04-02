//! Bindings for the [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk).
//!
//! The raw bindings can be found in [`sys`].
//!
//! Feel free to contribute!
//!
//! ---
//!
//! # Encoding
//!
//! See [NVIDIA Video Codec SDK - Video Encoder API Programming Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvenc-video-encoder-api-prog-guide/index.html).
//!
#![warn(
    missing_docs,
    clippy::pedantic,
    clippy::style,
    clippy::unwrap_used,
    missing_debug_implementations,
    missing_copy_implementations
)]
pub mod sys;
