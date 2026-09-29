# SPICE Engine: capability reference (v1.3.9)

For Python users and integrators. The "why / how it works" behind each row lives in code comments and the named integration tests.

- Package `spice_engine` (maturin wheel; `import spice_engine as se`).
- Execution units: CPU (x86_64 SIMD / arm64 kernels), rayon-parallel, deterministic placement.
- Python ≥ 3.9; requires numpy.

---

## 1. Top-level objects

| Object / function | What it is |
|---|---|
| `se.Structure` | in-memory atom array (the engine's only input shape) |
| `se.Env` **(new v1.3.7)** | reusable build-recipe object (environment + build knobs) |
| `se.Engine` | one built system + MD stepping + observability |
| `se.scan_stability` / `se.scan_stability_ranges` / `se.scan_radial` | environment-grid / range stability scans (phase maps) |
| `se.mutate_sequence(seq, pos, aa)` / `se.validate_sequence(seq)` | sequence tools (`pos` is 0-based) |
| `se.version()` | crate version string |

## 2. Structure

```python
s = se.Structure.from_atoms(atom_names, elements, res_seq, res_names, coords, occupancy=None)
#   all arrays length N; coords [N,3] f32 (Å); all-keyword calls also supported
#   ⚠ structures containing altlocs MUST pass occupancy: prepare() takes the
#     highest-occupancy conformer by it; without it conformers superpose → hard clashes
s = se.Structure.from_mmcif("path.cif")   # convenience path: skips waters/non-aa residues
s.sequence(); s.residue_count()
```

Production path: `from_atoms` (Parquet → numpy). `from_mmcif` is for debugging.

## 3. Env (v1.3.7)

Define once, share between `build`/`mutate`; this kills the two-call-site parameter drift that motivated it:

```python
e = se.Env(ph=7.0, temp_k=310.0, pressure_bar=1.0, ionic_strength_m=0.15,
           mg_cl2_m=0.0, ca_cl2_m=0.0, sr_cl2_m=0.0, ba_cl2_m=0.0,
           redox_reducing=0.0, cosolvents_json="", salts_json="",
           relax_iters=2000, tolerance=2.0, strict_incomplete=True,
           box_pad_a=0.0)
e.temp_k = 350.0            # all fields readable/writable; `pressure` aliases `pressure_bar`
```

| Field | Meaning | Notes |
|---|---|---|
| `ph` | build-time protonation input (ladder acts on **internal residues**; termini/His special-cased) | changing pH = rebuild |
| `temp_k` | target temperature K (clamped 250 to 400) | runtime-adjustable via `set_temperature` |
| `pressure_bar` | target pressure, bar; **0 = NVT** (barostat off) | RL production segments use 0 |
| `ionic_strength_m` | **NaCl background molarity**, NOT total I; total I via `engine.effective_ionic_strength_m()` | - |
| `mg/ca/sr/ba_cl2_m` | divalent-salt formula-unit molarity (12-6-4 Li-Merz/OPC parameters) | the mutate path re-inserts them only with `repack_ions=True` (v1.3.8) |
| `redox_reducing` | ∈[0,1]: deterministically (seeded) reduces a fraction of disulfides to CYS | - |
| `cosolvents_json` | urea/TMAO/GdmCl presets or custom multi-site molecules | presets vendored from CHARMM CGenFF v4.6 |
| `salts_json` | generic electrolyte `[{"cation":"K+","anion":"Cl-","molarity":0.15}]`; ion names resolve through a **closed registry** | stoichiometry auto-balanced |
| `relax_iters/tolerance` | builder-tail L-BFGS iteration cap / convergence threshold | |
| `box_pad_a` | solvation box padding, Å; **0 = builder default (10 Å)** | v1.3.8; the big-box knob for extended-chain/PFDE protocols. Do NOT instead rescale starting coordinates: uniform scaling >1.2× corrupts build-time protonation (the heavy-neighbor test reads stretched sidechains as isolated residues) |
| `strict_incomplete` | True: residues missing any charge-lib sidechain heavy atom **reject the build**, naming the missing atoms (the 4LPX class); False: degrade-build + warn | - |

## 4. Engine: building

```python
# v1.3.7 recommended (Env object; no more hand-typed parameter blocks):
eng = se.Engine.build(s, env=e)
mut = eng.mutate_with_solvent_reuse(s2, env=e)

# Fully backward-compatible legacy (spice_rl style; the 7-slot positional
# prefix contract is permanent):
eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.15, 2000, 2.0)                  # positional
eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.15, 2000, 2.0, strict_incomplete=False)
```
- `env=` is **mutually exclusive** with any individual scalar (collision → ValueError naming the conflicting fields; no silent "who overrides whom").
- Passing the trailing `mg_molar/ca_molar/sr_molar/ba_molar/redox_reducing/ cosolvents_json/salts_json` directly to `build/mutate` still works but raises a **DeprecationWarning** (fold them into `Env`); `strict_incomplete/ relax_iters/tolerance` are first-class parameters and never warn.
- Build-time warnings are part of the contract: `[solvent_reuse]`, `H clash resolution:`, `net charge` etc. go straight to stderr.

`mutate_with_solvent_reuse` semantics: deep-copy the parent box's water/ions/cosolvents + swap the solute; **unchanged residues ride the parent's relaxed coordinates** (index-aligned copy, element-checked). Measured: env-only change ~0.2 s, real single-point mutation ~1 s (vs cold build ~30 s). By default it does **not** repack ions for changed `ionic`/divalent/salt fields (it warns when you try); environment changes go through full build / scans.

**Opt-in salt repack (v1.3.8):** `mutate_with_solvent_reuse(..., repack_ions=True)` keeps the solvent box fixed but strips the monatomic ions, remaps every index-keyed structure, re-inserts the requested ion population (same helpers, same 6 Å exclusion, divalent and `salts_json` included) and relaxes the new sites. The advertised contract: **ion counts and effective ionic strength match a fresh build at that salt to integer rounding** (measured: 2LYZ 0.15 to 0.30 M, both I_eff = 0.3330 M, in ~2.4 s vs ~30 s cold), while **positions deliberately do not**: stripped sites are not healed back to water, so the box carries an accepted density drift and the result is **not bitwise-comparable with a fresh build**. Use it for cheap salt-condition evaluation; canonical phase-map points still come from full builds.

## 5. Engine: stepping & runtime knobs

| Method | Semantics |
|---|---|
| `step(action=None)` | advance + full metrics. `action` is an `f32[M=16]` biased-force coefficient vector on the low-rank basis (the RL action); **length must be exactly 16** (wrong → ValueError) |
| `step_md(action=None)` | advance only, no metrics (the hot path; RL loops use this) |
| `step_fast(action=None, metrics_every=0)` | advance; metrics on the absolute-step cadence; the `metrics_available` flag tells you when they're in the dict |
| `equilibrate(ramp_steps=300, t_start_k=100.0, k_restraint=0.0, hold_steps=100, restrain_hydrogens=False, friction_gamma=10.0)` | optional NVT warm-up ramp (barostat frozen throughout); RL callers handle settling themselves |
| `set_temperature(k, solute_k=None)` | change the bath setpoint at runtime; `solute_k` gives the solute its own setpoint (dual bath, v1.3.8) while water keeps `k` |
| `set_dual_bath(solute_k)` | shorthand: solute setpoint + solute friction γ = 2.0 in one call (v1.3.8) |
| `set_langevin_gamma(gamma)` | runtime friction for the water bath |
| `reset_velocities()` / `reset_pseudo_labels()` / `reset_trend()` | redravel velocities / clear time-averaged Cα / clear trend windows |
| `set_integrator(mode)` | `"langevin_middle"`(γ0.5, default) \| `"langevin_strong"`(γ10) \| `"nve"` (conservation tests) |
| `set_force_overrides(bonded, coulomb, lj, long_range)` | diagnostic bisection: `True` **disables** that force class |
| `set_skip_water_thermostat(bool)` | diagnostic: rigid water ignores thermostat noise |
| `add_distance_restraint(i, j, r0, k)` | add a harmonic distance restraint (½kΔr², into the bonded virial) |

### The step dictionary (keys are a stable contract)

| Key | Type | Meaning |
|---|---|---|
| `u_t_kcal` / `u_t_kj` | f64 | potential energy (both unit systems, always present) |
| `coords_ca` | f32[L,3] | Cα coordinates |
| `step_count` / `time_ps` | int/float | absolute step number / simulation time |
| `crashed` / `crash_reason` | bool/str\|None | crash verdict (U≥1e8 kcal or trend alarm); `trend_alarm:<sig>` prefix = trend early-stop |
| `trend_alarm` | str\|None | `"energy_rise"｜"rg_expand"｜"ss_loss"` |
| `n_clamped` / `max_accel_clamped` | int/float | atoms cut by the acceleration cap (1e5 Å/ps²) this step / max reduction; **persistent >0 = force spikes being beaten down** |
| `t_kin` | float | instantaneous kinetic temperature, K |
| `m1..m5, rg, stability_margin, rmsf` | f64 | via `step`, `step_fast` or `metrics()`; definitions in `metrics.rs` |

`step_fast` adds `metrics_available: bool`. **Keys are only ever added, never removed.**

### The trend monitor (fail-fast, the RL lifeline)

```python
eng.set_trend_monitor(preset="rl_fail_fast")   # 0.6 ps window, z=8, high floors; RL default
eng.set_trend_monitor(window=60, check_every=5, z_threshold=3.0,
                      energy_floor_ps=50.0, rg_floor_ps=0.1, ss_floor_ps=0.01,
                      skip_steps=0)            # or fully manual
eng.has_trend_monitor(); eng.reset_trend(); eng.clear_trend_monitor()
```
Three signals (energy rise / Rg expansion / H-bond SS loss) get OLS-slope z-tests; **≥2 of 3 must cross** their floors to alarm (the floors are deliberately high).

## 6. Engine: observability & diagnostics

| Method | Returns |
|---|---|
| `metrics()` | dict: m1..m5, rg, u_t_kcal, n_ss_ref/kept, n_surface_charged, stability_margin, rmsf |
| `pseudo_labels()` / `coords_ca()` / `coords_ca_flat()` | time-averaged Cα [L,3] / live Cα / flat [3L] |
| `sequence()` / `n_residues()` / `per_residue_max_force()` / `atom_labels()` | sequence / residue count / max force per residue / (element, aa, seq, serial) table |
| `thermo_info()` / `species_temperatures()` / `water_rigid_split()` | DOF / kinetic-energy decomposition dicts |
| `kinetic_energy_kcal()` / `step_count()` / `time_ps()` / `computation_time()` | scalars + timing buckets (bonded/nonbonded/ewald/neighbor) |
| `effective_ionic_strength_m()` **(v1.3.7)** | the box's **actual** I = ½Σcᵢzᵢ² (counterions + divalent included); the input `ionic_strength_m` is only the NaCl background |
| `pressure_bar()` **(v1.3.7)** | last measured instantaneous pressure, bar (pure diagnostic under NVT; a large positive number on a cold box is normal) |
| `env_info()` **(v1.3.7)** | echo of the actually-effective (post-clamp) environment vector, efield included |
| `exclusion_diagnostics()` **(v1.3.7)** | 1-2/1-3, 1-4, bond/angle/dihedral table and pair counts; catches "bonded pairs leaked into nonbonded" |
| `electrostatic_potential(points)` **(new v1.3.8)** | φ in kcal/mol/e at arbitrary probe points `[[x,y,z],...]`, through the engine's own PME split (real-space + reciprocal, same charge packing as the production forces). The PME self/background constant cancels **only in differences**: report relative quantities (site potential minus a bulk reference), never an absolute φ |
| `atom_sasa(probe_radius=1.4, n_sphere=200)` **(new v1.3.8)** | per-atom Shrake-Rupley SASA, Å². Computed over solute + ion atoms only: explicit waters are **not** occluders (the 1.4 Å probe *is* the solvent model; letting hydration water occlude collapses the solute SASA by three orders of magnitude). Pocket volume, MM/PBSA nonpolar term, burial analysis |
| `atom_names()` **(new v1.3.8)** | true per-atom names (`CA`, `CB`, `OD1`, `OXT`, added-H names), from the input mmCIF `type_in_res` chain. **Not** `atom_labels()`'s Amber ff types (`2C`/`O2` are element types, not PDB names) |
| `select_atoms(res_seq, names=None, sidechain_heavy=False)` **(new v1.3.8)** | atom-index query by residue number (single int or list) + optional name list + optional sidechain-heavy filter (drops backbone N/CA/C/O/OXT and all H). Feeds `contact_count`, `bottleneck_radius`, `electrostatic_potential` |
| `contact_count(set_a, set_b, cutoff)` **(new v1.3.8)** | pairs across two index sets within `cutoff` (min-image); the Q_site catalytic-atom contact-retention metric |
| `bottleneck_radius(path, spacing=0.5, exclude=None, include_water=False)` **(new v1.3.8)** | `(profile, min_clearance)` in Å: at each point of a `spacing`-resampled polyline through `path` (≥3 points), the min-image distance to the nearest vdW surface. Negative = the path runs through matter; `exclude` drops blocking atom indices (e.g. a roaming ion parked in a corner lane); `include_water=True` adds explicit waters as occluders (default: solute + ions only, same policy as `atom_sasa`) |
| `energy_terms()` **(new v1.3.8)** | `{total, nonbonded, bonded}` kcal/mol split of the current potential energy |
| `clash_report(min_force)` / `force_report(min_force)` | high-force atom lists (H-only / all elements), O(N²) diagnostic |
| `mask_fraction()` | active fraction of the current action mask |

**v1.3.8 physics note: the cold-box pressure closed as a four-layer bookkeeping story.** (1) The `ewald` fork never removed the erf-side of 1-2/1-3 and intra-rigid-water pairs from $S(\mathbf k)$; every water leaked a frozen $\approx -134$ kcal/mol into the reciprocal energy and excluded solute pairs felt phantom reciprocal forces. (2) The same crate’s k-space virial carried an inverted sign: correct under grid-scales-with-box is $W_{\mathrm{lr}} = \sum_{\mathbf k} E_{\mathbf k}(k^2/2\alpha^2 - 1)$, fixed upstream at rev 22cc623 and shipped as pinned 0.1.15. (3) The SETTLE constraint virial had been computed every step since day one and discarded (`let _ =`); the billing contract now returns and books it (each impulse ÷ its own drift dt; LangevinMiddle halves its two half-drifts). (4) The production water template carried TIP3P H geometry under OPC charges; intake now rebuilds every site at canonical OPC internals (`WaterMolOpc::from_axes`). Water-water *direct*-space pair energy is also booked now (it is what makes the energy ledger audit-grade: $U$ settles near $-12.3$ kcal/mol per water, and batch-split invariance holds to $\sim 4\times10^{-12}$). Measured after all four: 2LYZ default build, NVT 300 K, $P = +12.2$ kbar stable (was the +43 kbar class); the residue is build-time density calibration (open work). Plus the dual-bath thermostat API above (`solute_k`), without which annealing ladders ran the solute at 0.72 to 0.78× their label. **The dynamics changed for every run**: results produced before v1.3.8 (RL episodes, PFDE ladders) need revalidation (implementation in `pme.rs`/`integrate.rs`, evidence in the named tests).

## 7. Stability scans (phase maps)

```python
pts = se.scan_stability(s, temps=[...], phs=[...])                # explicit axes
pts = se.scan_stability_ranges(s, (280,400,10), (5,9,0.5),
                               pressure_range=None, ionic_range=(0,0.3,0.05),
                               n_steps=20, equil_steps=10, repeats=3,
                               relax_iters=None, tolerance=2.0,
                               prune_crashed=True, adaptive_repeats=True,
                               trend_detector=True, trend_window=100, trend_z_threshold=3.0)
rays = se.scan_radial(s, anchor_ph=7.0, anchor_temp=310.0, ...)   # boundary radial search
```
- Each point is an independent full build (ionic/pH change = rebuild; the canonical channel for environment changes).
- Point-dict keys are always present (NaN-filled where a metric wasn't computed): `temp,ph,pressure,ionic,mg,ca,sr,ba,stable,crashed,build_failed, terminated_reason,m1..m5,rg` (since v1.3.7 `scan_radial` points follow the same convention).
- Threshold / prune semantics (`is_stable`, adaptive repeats, two-stage grid) in `domain.rs`; the organizing ideas (fail fast, prune); metric definitions in `metrics.rs`.

## 8. Environment-knob reachability matrix

| Knob | Env/build | Runtime | Scans |
|---|---|---|---|
| pH / ionic / divalent / salts / cosolvents / redox | ✅ (build-time semantics) | ✗ (rebuild required), except salt via mutate `repack_ions=True` (v1.3.8): counts-exact, positions drifted | grid axes (sr/ba/salts/cosolvents **not** in the grid) |
| temperature | ✅ | ✅ `set_temperature` | ✅ axis |
| pressure | ✅ (0 = NVT) | mode switch: no | pressure axis (`stability`'s `pressures=` / ranges' `pressure_range=` / radial) |
| efield (+ω) | Rust `BuildOptions` only; **no FFI entry yet** | - | - |
| custom_protonation (per-residue variants) | Rust `BuildOptions` only; **no FFI entry yet** | - | - |

## 9. Limitations & explicit non-goals (each is a decision, not an unfinished task)

1. **RL production segments must be NVT (`pressure=0`)**: cold-build box + NPT position teleporting = continuous external work P·ΔV̇ → detonation. NPT is for already-relaxed systems.
2. **The mutate path does not repack salt/ions by default**: ions were inserted by *displacing* waters, so in-place concentration changes drift density and cannot bit-align with a fresh build; the default path warns and environment change → rebuild. v1.3.8 adds the opt-in `repack_ions=True`: the solvent box stays fixed, monatomic ions are stripped and the requested population (background NaCl, divalent, `salts_json`) re-inserted; **counts and effective I match a fresh build to integer rounding, positions do not** (accepted density drift, stripped sites not healed back to water; the ion-layout golden never runs this path). A density-conserving variant is backlog.
3. **The pH ladder drives internal residues only**; termini have fixed protonation (fractional terminal charge at extreme pH is by design).
4. **Closed ion registry**: Na/K/Cl/Mg/Ca/Sr/Ba (provenance-locked); polyatomic ions (SO₄²⁻ etc.) await vetted parameter provenance.
5. **12-6-4 C₄ pairs through the one-sided water channel only** (Panteva model); custom sites need c4 on both partners to combine.
6. **Build RNGs are deliberately non-reproducible** (thermostat/barostat entropy fresh per build); only pre-MD *placement* is seeded (ions/redox). Never assert bit-equality across full builds.
7. **CPU-only numeric path** (GPU feature compile-gated off); wheels are CI-built for x86_64 only.
8. **No enhanced-sampling / free-energy methodology** (the alchemical module is an experimental lane, not exposed to FFI).

## 10. Compatibility contract (what spice_rl / test scripts may rely on)

- The 7-slot positional prefix of `build/mutate`, `(structure, ph, temp, pressure, ionic_strength_m, relax_iters, tolerance)`, never reorders.
- Step-dict and scan-point keys are only added, never removed; `u_t_kcal` + `u_t_kj` + `crashed` are permanently hard-present.
- All failures → `ValueError`; the `"Incomplete structure"` substring is a tested contract.
- Feature detection via `hasattr(Engine, "mutate_with_solvent_reuse")` etc. is safe (methods only ever added).
- DeprecationWarnings on trailing scalars are notices, not blocks; removal keeps at least one major-version window and is announced in these docs.

## 11. Browser (WebAssembly) build — v1.3.10

The engine compiles to `wasm32-unknown-unknown` and runs a full solvated MD in the tab. This is a **new feature-gated build target**, not a new runtime model — the pure-Rust engine core was always pyo3-free (`ffi.rs` is the only Python seam), so the port is dependency + shim surgery, zero hot-path physics changes. Native numbers are bit-identical (the `web` build never compiles the touched native code paths; the default-feature native build is unchanged and its full test suite is green at 1.3.9).

**API surface + demo** live in this repo's `web/` directory (`web/api.mjs` application facade + `web/loader.mjs` ABI wrapper + `web/index.html` reference page; `web/dist/` is gitignored — stage it with `make web-dist`, or fetch the CI artifact token-free via `make web-nightly`; run `make web-serve` to open it in a browser tab). Authoritative doc: `docs/web_demo.md`.

| capability | status |
|---|---|
| build from pasted mmCIF (`spice_build_mmcif`) | ✅ env params (pH/salt/divalent/redox), padding, relaxation |
| step / step-with-bias-force | ✅ `spice_step` / `spice_step_action` |
| dual-bath thermostat, timestep, friction, NPT on/off | ✅ `spice_set_temperature{,solute}` / `set_timestep` / `set_gamma` / `set_pressure` |
| observables (energy split, pressure, species T, Rg, ionic, net charge) | ✅ mirrors the FFI step dict |
| positions/roles snapshot (solute + OPC O/H0/H1) | ✅ re-sliced views survive `memory.grow` |
| toy mode (per-term force disable) | ✅ `spice_set_force_overrides` |
| **FFI-parity analysis probes** (v1.3.10) | ✅ ESP / field / pme_positions / SASA / select / contacts / bottleneck / metrics(m1-m5) / per-residue force / clash & force reports / thermo & species & env & exclusion info / water rigid split / computation time / debug state dump / rigid-scale probe |
| **FFI-parity control** (v1.3.10) | ✅ trend monitor (arm/reset/clear), integrator switch (incl. NVE), distance restraints (add/retarget/clear), skip-water-thermostat, pseudo-label reset, `equilibrate` ramp, external-field build knobs (`efield_json`/`efield_omega`) |
| solvent-reuse mutation / RL action pre-mapping / NumPy interop | ❌ not ported (RL-build machinery; JS passes per-atom bias arrays directly and gets typed arrays natively) |
| rayon parallelism | ⚠ single-thread (wasm has no threads without SharedArrayBuffer+workers — out of scope) |
| CHARMM36m / Martini3 / candle charge-GNN / network downloads | ❌ compiled out (`web` feature); native-only |

Measured (single-thread scalar wasm, node/V8 — representative of the browser): tiny system (1.5 k sites) ≈ **12 ms/step**; 2LYZ build+hydrate+relax ≈ **60 s** then step. SIMD128 build runs without trapping but gave no win at that size; it is non-bit-exact (FTZ). Build with `make wasm` / `make wasm-simd`, gate with `make web-smoke`; the wasm graph is held free of `wasm-bindgen`/`js-sys`/`ring`/`rustls`/`web-time`/`ureq`/`indicatif` by `.github/workflows/web-wasm.yml`.

Three portable-Rust blockers were solved with shims that do not change native output: (1) `std::time::Instant`/`SystemTime` are panic stubs on wasm → `engine::utility::clock::Mono` (native = pass-through; wasm = a 2-import JS clock); (2) OS-entropy `rand::rng()` in the solvent path → `engine::utility::entropy::session_rng` (native = identical `ThreadRng`; wasm = `StdRng` seeded from clock+counter); (3) `indicatif` pulls `web-time`/wasm-bindgen → `crate::progress` no-op bar on wasm. See `docs/web_demo.md`.
