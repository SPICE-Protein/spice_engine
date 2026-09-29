# spice_engine on the web (WebAssembly) — v1.3.10

This is the **authoritative in-tree reference** for the browser build. The runnable
artifact + JS loader + a minimal reference page live in this repo's `web/`
directory: `web/loader.mjs` (ergonomic JS wrapper), `web/index.html` (canvas
reference page), `web/mini.cif` (fast fixture), `web/smoke.mjs` / `web/selftest.mjs`
(node correctness gates; `web/api.test.mjs` drives the whole Web API end-to-end). **Wasm blobs are NOT committed** — `web/dist/` is
gitignored; populate it with `make web-dist` (local build) or `make web-nightly`
— a tokenless curl of the latest CI artifact via **[nightly.link](https://nightly.link)**:
`https://nightly.link/SPICE-Protein/spice_engine/workflows/web-wasm/main/spice_engine-wasm.zip`
(GitHub's own artifact endpoint answers 404 to anonymous users even on public
repos; nightly.link re-serves the newest successful main-branch run for free —
GET works, HEAD does not). Serve with `make web-serve` → `http://localhost:8080`.

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

# stage web/dist/{,simd.}wasm(.gz) for the demo (or make web-nightly = tokenless
# curl of the latest CI artifact through nightly.link)
make web-dist

# end-to-end correctness gate (node): instantiate → build 2LYZ → step → read back
make web-smoke      # scalar
make web-verify     # both variants

# type-check only
make web-check

# Web API facade end-to-end (spins its own http server; needs web/dist staged)
make web-apitest
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

## API

**Recommended surface: `web/api.mjs`** — a zero-dependency facade (`Sim.load(cif, {env})` →
`step/stepAction/set/setForcesOff/snapshot/animate/dispose`, camelCase env keys, rAF render-loop
helper, exact metric keys typed in `web/api.d.ts`; end-to-end test `node web/api.test.mjs`).
`loader.mjs` underneath keeps the direct ABI mapping; everything below is the raw export list.

**v1.3.10 ports the Python FFI's analysis + control surface.** The engine core was already
pyo3-free for every probe in `ffi.rs`, so the wasm exports are the same `&SpiceEngine` calls,
same JSON key spellings — scripts translate 1:1 between Python and JS. Numeric probes (ESP,
field, SASA, PME positions, Ca/pseudo-labels, per-residue forces) write their float arrays
into a second OUT channel — read `spice_res_ptr`/`spice_res_len` (f64 elements) before the
next numeric call (views detach on `memory.grow`, so copy). JSON probes (metrics m1-m5,
energy terms, species temperatures + KE/DOF, thermo info, water rigid split, env info,
exclusion diagnostics, computation time, atom names/labels, sequence, debug state dump,
clash/force reports, rigid-scale probe) come through the usual OUT buffer. Selection-style
calls (select_atoms / contact_count / bottleneck / set_trend / equilibrate / efield with
time-averaged positions) take a JSON argument. Control additions: distance restraints
(add/retarget/clear — SMD pulls are expressible in the tab), trend monitor (arm via preset
`rl_fail_fast`/`default` or knobs; fires `crashed=true` + `trend_alarm`), integrator switch
(`langevin_middle`/`langevin_strong`/`nve`), skip-water-thermostat bisect knob,
`equilibrate` (synchronous ramp+hold — at mini scale ~15 steps block the tab for a fraction
of a second, budget accordingly), pseudo-label reset, and `efield_json`/`efield_omega` build
knobs (external field, static or oscillating). NOT ported: `mutate_with_solvent_reuse`
(RL build optimization), the RL 16-dim action pre-mapping (JS bias arrays are already
per-atom), `mask_fraction` (client-side trivial), and NumPy interop (typed arrays are native).

`spice_pme_positions` is the engine's charge-site packing: solute atoms **wrapped to [0,L)**,
then per water **M, H0, H1** — the uncharged O is excluded — so its length is
`3 * n_sites` but not the same site composition as `spice_positions` (O/H0/H1). Feed it to
`spice_efield`'s optional positions argument for time-averaged structure queries.

Exports: `spice_init`, `spice_version`, `spice_alloc`/`spice_free`, `spice_out_ptr`/
`spice_out_len`, `spice_res_ptr`/`spice_res_len`, `spice_last_error`, `spice_build_mmcif`,
`spice_step`, `spice_step_action`, `spice_observables`, `spice_positions` (+ `_ptr`/`_len`
for positions and roles), `spice_set_temperature`, `spice_set_gamma`, `spice_set_timestep`,
`spice_set_pressure`, `spice_set_force_overrides`, `spice_reset_velocities`,
`spice_free_engine`, and the v1.3.10 parity set: `spice_esp`, `spice_efield`,
`spice_pme_positions`, `spice_atom_sasa`, `spice_coords_ca`, `spice_pseudo_labels`,
`spice_reset_pseudo_labels`, `spice_per_residue_max_force`, `spice_metrics`,
`spice_energy_terms`, `spice_species_temperatures`, `spice_thermo_info`,
`spice_water_rigid_split`, `spice_env_info`, `spice_exclusion_diagnostics`,
`spice_debug_state_dump`, `spice_atom_names`, `spice_atom_labels`, `spice_sequence`,
`spice_select_atoms`, `spice_contact_count`, `spice_bottleneck`, `spice_clash_report`,
`spice_force_report`, `spice_rigid_scale_probe`, `spice_computation_time`,
`spice_set_integrator`, `spice_set_trend`, `spice_reset_trend`, `spice_clear_trend`,
`spice_has_trend`, `spice_set_skip_water_thermostat`, `spice_add_restraint`,
`spice_update_restraint`, `spice_clear_restraints`, `spice_equilibrate`.

`spice_build_mmcif(cif, params_json)` accepts the same build knobs as the native
`Engine.build` (ph / temp_k / pressure_bar / ionic_strength_m / relax_iters / tolerance /
strict_incomplete / box_pad_a / divalent / redox / cosolvents_json / salts_json; v1.3.10
adds efield_json / efield_omega).

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
