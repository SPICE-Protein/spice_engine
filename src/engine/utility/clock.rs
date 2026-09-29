//! Monotonic-time shim (v1.3.9 web port).
//!
//! `std::time::Instant` and `SystemTime` are panic stubs on
//! `wasm32-unknown-unknown` — the platform ABI exposes no clock, so
//! `Instant::now()` traps with "time not implemented on this platform" on
//! its first call, which for us was inside `step_borrowed` (every step).
//! Every timing site in this crate is pure bookkeeping (the
//! `computation_time` buckets, build progress durations, RNG seeds); no
//! physics reads a clock mid-step, so replacing `Instant` with this two
//! method shim changes no numeric output on native and unblocks wasm.
//!
//! Native: thin pass-through over `std::time`. Wasm: one imported JS
//! function `env.now_ms` (bound to `performance.now()` by the loader, see
//! `src/web.rs`), monotonic since page origin.

use std::time::Duration;

/// A monotonic instant mirroring the two `std::time::Instant` operations
/// this crate actually uses: `now()` and `elapsed()`.
#[derive(Copy, Clone)]
pub struct Mono {
    #[cfg(not(target_arch = "wasm32"))]
    inner: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    millis: f64,
}

impl Mono {
    /// Current monotonic instant.
    #[inline]
    #[must_use]
    pub fn now() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                inner: std::time::Instant::now(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self {
                millis: js_now_ms(),
            }
        }
    }

    /// Time since this instant was taken. Clamped at zero on wasm where the
    /// JS clock could in principle be re-based.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.elapsed()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let delta_ms = js_now_ms() - self.millis;
            Duration::from_micros((delta_ms * 1_000.0).max(0.0) as u64)
        }
    }
}

/// Nanoseconds since the Unix epoch, for seed generation. The consumers
/// (barostat `generate_unique_seed`) need uniqueness across builds, not
/// true entropy — the JS wall clock plus our counter/SplitMix64 mixer
/// preserves that contract on wasm.
#[must_use]
pub fn unix_ns() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        (js_now_ms() * 1_000_000.0).max(0.0) as u64
    }
}

/// The JS-provided millisecond clock; supplied by the browser loader as
/// `env.now_ms = () => performance.now()`.
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn now_ms() -> f64;
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn js_now_ms() -> f64 {
    // Safety: `now_ms` is a pure JS binding; it neither aliases Rust memory
    // nor mutates anything the engine observes.
    unsafe { now_ms() }
}
