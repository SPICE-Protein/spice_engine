//! Session-RNG seam (v1.3.9 web port).
//!
//! Three solvent-build sites want a fresh OS-entropy RNG per call
//! (`rand::rng()` → getrandom). `getrandom` deliberately refuses to compile
//! for `wasm32-unknown-unknown` unless a JS glue crate enables its
//! `wasm_js` backend, and we ship without wasm-bindgen. The sites only use
//! the stream for water-orientation jitter during packing — decorrelation
//! matters, cryptographic entropy does not — so on wasm we seed a
//! `StdRng` from the clock shim and an avalanche counter instead.
//!
//! Native keeps `ThreadRng` verbatim: identical streams, identical behavior.
//! Per-physical-step thermostat/barostat noise does NOT route here; it
//! rides the barostat's own `StdRng`.

/// The RNG type the solvent packing sites thread around.
#[cfg(not(target_arch = "wasm32"))]
pub type SessionRng = rand::rngs::ThreadRng;

/// Wasm stand-in: seeded, not entropic. See module docs.
#[cfg(target_arch = "wasm32")]
pub type SessionRng = rand::rngs::StdRng;

/// A fresh session RNG, OS-entropy where the platform has it.
#[must_use]
pub fn session_rng() -> SessionRng {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rand::rng()
    }
    #[cfg(target_arch = "wasm32")]
    {
        use rand::SeedableRng;
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);
        let ticks = super::clock::unix_ns();
        let count = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        // Same SplitMix64 avalanche the barostat seed uses, so consecutive
        // builds in one page never share a stream.
        let mut x = ticks
            .wrapping_add(count)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        SessionRng::seed_from_u64(x ^ (x >> 31))
    }
}
