# spice_engine on the web (WebAssembly) — v1.3.9

This is the **authoritative in-tree reference** for the browser build. The runnable
artifact + JS loader + a minimal reference page live in this repo's `web/`
directory: `web/loader.mjs` (ergonomic JS wrapper), `web/index.html` (canvas
reference page), `web/dist/*.wasm.gz` (prebuilt scalar + simd128), `web/mini.cif`
(fast fixture), `web/smoke.mjs` / `web/selftest.mjs` (node correctness gates).
Serve it locally with `make web-serve` (then open `http://localhost:8080`).

## Why this works at all

The engine core (`src/engine/**`) has **always been pyo3-free** — `src/ffi.rs` is the
single Python seam and it sits behind the non-default `python` feature. So `cargo
check` without that feature already only built the pure-Rust dynamics engine. The wasm
port therefore is **not** a rewrite of the physics; it is (a) dependency-graph surgery
to drop the crates that cannot build or load on `wasm32-unknown-unknown`, and (b) two
tiny portability shims (clock + entropy) that leave native output bit-identical.

## Build

```bash
# scalar baseline (runs in every wasm engine since 2017)
make wasm
#   → target/wasm32-unknown-unknown/release/spice_engine.wasm      (~5 MB / 1.77 MB gz)

# simd128 variant (Chrome 101+/FF 100+/Safari 16+; non-bit-exact)
make wasm-simd
#   → …/spice_engine_simd.wasm

# end-to-end correctness gate (node): instantiate → build 2LYZ → step → read back
make web-smoke      # scalar
make web-verify     # both variants

# type-check only
make web-check
```

The command behind them:

```bash
RUSTFLAGS='--cfg getrandom_backend="unsupported"' \
  cargo build --offline --release --no-default-features --features web \
  --target wasm32-unknown-unknown
```

`RUSTFLAGS` is scoped **per Makefile recipe** and never written to `.cargo/config.toml`,
so it cannot leak into a native/wheel build.

## The `web` feature graph (what is compiled out)

| dep | native | web | how |
|---|---|---|---|
| pyo3 / numpy | on `python` | — | not a web feature |
| candle (charge GNN) | `inference` | off | `web = []`; `partial_charge_inference` cfg-gated; the bincode half moved to always-on `pci_files` (the water template loads through it) |
| bio_apis / bio_files network | `network` | off | fork `SPICE-Protein/bio_files` makes `bio_apis` optional; `network` forwards `bio_files/network` |
| ring / rustls / ureq | via network | off | dropped with the above |
| indicatif (progress) | native | off | `web-time` pulls `wasm-bindgen`/`js-sys`; replaced by `crate::progress` no-op bar |
| CHARMM36m / Martini3 assets (~28 MB) | `heavy-forcefields` (default on) | off | `include_str!` gated; requesting them in `web` returns a clean error |
| ewald → statrs rand | — | off | fork `SPICE-Protein/ewald` `statrs = default-features=false` (v0.1.16) |

Structural gate (`.github/workflows/web-wasm.yml`) asserts the web graph contains
**none** of `wasm-bindgen js-sys web-sys ring rustls web-time ureq indicatif`.

## ABI design (why hand-rolled, no wasm-bindgen)

There is **no wasm-bindgen CLI offline** to post-process its custom-section import glue,
so the wasm exposes a raw `extern "C"` surface (`src/web.rs`). Exactly two JS imports —
`env.now_ms` (the clock) and `env.log` (console sink). Everything else:

- integer **handles** into a thread-local engine registry (`handle = index + 1`);
- JS→Rust bytes: `spice_alloc(n) → ptr`, fill, pass `(ptr, n)`, `spice_free(ptr, n)`;
- Rust→JS JSON: written to a module OUT buffer, read via `spice_out_ptr()/len()` before
  the next call;
- positions/roles in a separate persistent scratch that must be **re-sliced every frame**
  (wasm `memory.grow` detaches old typed-array views);
- every entry point is `catch_unwind`-guarded (release profile keeps `unwind`, **not**
  `panic = "abort"`): a Rust panic returns a negative code and writes `{"error": …}`,
  never a wasm trap.

## API (see the demo `loader.mjs` for the ergonomic JS wrapper)

Exports: `spice_init`, `spice_version`, `spice_alloc`/`spice_free`, `spice_out_ptr`/
`spice_out_len`, `spice_last_error`, `spice_build_mmcif`, `spice_step`,
`spice_step_action`, `spice_observables`, `spice_positions` (+ `_ptr`/`_len` for positions
and roles), `spice_set_temperature`, `spice_set_gamma`, `spice_set_timestep`,
`spice_set_pressure`, `spice_set_force_overrides`, `spice_reset_velocities`,
`spice_free_engine`.

`spice_build_mmcif(cif, params_json)` accepts the same build knobs as the native
`Engine.build` (ph / temp_k / pressure_bar / ionic_strength_m / relax_iters / tolerance /
strict_incomplete / box_pad_a / divalent / redox / cosolvents_json / salts_json).

**Positions site order:** solute atoms, then per water `O, H0, H1` (the OPC EP/charge site
`M` is not emitted). `roles`: `0` solute heavy, `1` solute H, `2` water O, `3` water H.

`set_temperature(k, soluteK)` — pass a real `soluteK` to engage the **dual bath**; the v1.3.8
folding protocol rule still holds (single-bath leaves the solute ~0.72-0.78× setpoint).

## Measured budget (single-thread scalar wasm, node/V8 ≈ browser)

| system | sites | build | step |
|---|---|---|---|
| mini (5 res) | 1.5 k | ~3 s | ~12 ms |
| 2LYZ | 12.6 k | ~64 s | ~50-90 ms |

That is roughly **10-17 ns/day** for the small system on one thread; interactive rendering
means step a few per animation frame, not full-length production MD. Multithreading
(SharedArrayBuffer + worker pool) and GPU (WebGPU) are explicit non-goals. HSFA2-class
systems (277 k sites) are far too large for a tab build and are out of scope for the demo.

## Determinism caveat

wasm entropy is seeded from `env.now_ms` + a counter (`engine::utility::entropy`), so water
placement / redox / ion seeds differ per page load exactly as native builds are deliberately
non-reproducible. Same-config single-run trajectories are reproducible; cross-build bit
equality is not asserted anywhere (native or web).
