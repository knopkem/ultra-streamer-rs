//! Auto-generated bindings to NVIDIA Video Codec SDK.
//!
//! The bindings were generated using [bindgen](https://github.com/rust-lang/rust-bindgen)
//! using the scripts `sys/linux_sys/bindgen.sh` and
//! `sys/windows_sys/bindgen.ps1` for the respective operating system.

#[cfg(any(target_os = "linux", target_os = "windows"))]
mod guid;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod version;

#[allow(warnings)]
#[rustfmt::skip]
#[cfg(target_os = "linux")]
mod linux_sys;
#[cfg(target_os = "linux")]
pub use linux_sys::*;

#[allow(warnings)]
#[rustfmt::skip]
#[cfg(target_os = "windows")]
mod windows_sys;
#[cfg(target_os = "windows")]
pub use windows_sys::*;
