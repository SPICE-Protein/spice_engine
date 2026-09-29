//! Progress-bar shim (v1.3.9 web port).
//!
//! On native this is a thin re-export of `indicatif`, so the three call sites
//! (`equilibrate`, `domain` grid/radial scans, `minimize_energy` relaxation)
//! draw the same bars they always did.
//!
//! On `wasm32-unknown-unknown` it is a no-op stand-in with an identical surface.
//! Two reasons the real crate cannot ride into the browser:
//!
//! 1. `indicatif`'s clock on wasm is `web-time`, which pulls `wasm-bindgen` +
//!    `js-sys`. Our wasm artifact is a raw `extern "C"` module (there is no
//!    wasm-bindgen CLI offline to post-process its custom-section import glue),
//!    and the `cargo tree` structural gate for the web build must stay free of
//!    wasm-bindgen. Keeping `indicatif` out of the graph is what the
//!    native-only `[target..'cfg(not(wasm32))'.dependencies]` entry enforces.
//! 2. Even the timing calls would trap: `ProgressBar::new` reads
//!    `web_time::Instant::now()` → a `__wbg_*` import our hand-written loader
//!    does not provide. The only path that actually runs in the browser is the
//!    build-time relaxation in `minimize_energy`, and a progress bar there draws
//!    nothing a headless demo can see anyway.
//!
//! The methods mirror indicatif's signatures (`&self` interior-mutable style,
//! builder `ProgressStyle` consuming `self`) so the shared call sites compile
//! unchanged against either arm.

#[cfg(not(target_arch = "wasm32"))]
pub use indicatif::{ProgressBar, ProgressStyle};

#[cfg(target_arch = "wasm32")]
/// No-op progress bar; the browser build never draws one.
pub struct ProgressBar {
    _len: u64,
}

#[cfg(target_arch = "wasm32")]
impl ProgressBar {
    #[must_use]
    pub fn new(len: u64) -> Self {
        Self { _len: len }
    }
    pub fn set_style(&self, _style: ProgressStyle) {}
    pub fn set_message<M: std::fmt::Display>(&self, _msg: M) {}
    pub fn set_position(&self, _n: u64) {}
    pub fn inc(&self, _n: u64) {}
    pub fn finish(&self) {}
    pub fn finish_with_message<M: std::fmt::Display>(&self, _msg: M) {}
    pub fn println<M: std::fmt::Display>(&self, _msg: M) {}
    #[must_use]
    pub fn is_hidden(&self) -> bool {
        true
    }
}

#[cfg(target_arch = "wasm32")]
/// No-op bar style; the template/char builders accept and discard input.
#[derive(Clone)]
pub struct ProgressStyle {}

#[cfg(target_arch = "wasm32")]
impl ProgressStyle {
    #[must_use]
    pub fn default_bar() -> Self {
        Self {}
    }
    #[must_use]
    pub fn template(self, _tmpl: &str) -> Result<Self, String> {
        Ok(self)
    }
    #[must_use]
    pub fn progress_chars(self, _chars: &str) -> Self {
        self
    }
}
