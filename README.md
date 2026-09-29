# spice_engine — an all-atom molecular-dynamics engine (Rust · Python · WebAssembly)

> A **general-purpose all-atom MD engine**: ff19SB protein · Li/Merz **OPC** water (rigid) · monovalent **and divalent** ions (12-6 / 12-6-4) · GAFF2 small molecules · OL24 nucleic acids · SPME/PME electrostatics · Langevin + velocity-rescale thermostats + optional Monte-Carlo barostat (NVT/NPT) · a **conditional-environment** knob set (pH / ionic strength / divalent / redox / external electric field / cosolvents) · an RL-action (bias-force) interface · an observability suite · mutation with solvent-reuse · parallel pool · **PyO3 Python bindings** · and a **1.8 MB WebAssembly build that runs MD in the browser tab**.
>
> Protein adaptive evolution was the application that drove development; the engine itself is general-purpose and stands on its own — integrator, force-field plumbing, environment physics, browser port. **It is not a structure predictor**: no ML runs inside this crate.

The engine core (v1.3.9) lives in `src/engine/` (the migrated `md_core` dynamics engine — this is the hot path), with a newer SE-owned force-field tree in `src/forcefield/`. It takes atoms in, builds a fully solvated, charge- and protonation-consistent system, integrates it, and streams structure / energy / health signals out.

## Core concepts

```
                 ┌──────────────────────────────────────────────┐
  sequence+struct │  SPICE Rust engine (spice_engine)            │
  (from Python) ─▶│  build → MD step → 5 metrics M → RL loop     │
                 │  actions: low-rank bias forces / ΔT / ΔpH     │
                 └──────────────────────────────────────────────┘
                              │ pseudo-labels: time-averaged Cα
                              ▼
              stability-domain search / adaptive evolution (SAC)
```

- **Conditional environment** (v1.3 knob set): `EnvParams` drives build-time protonation (pH), thermostat/barostat setpoints, background NaCl ionic strength, **divalent salts** (Mg²⁺/Ca²⁺/Sr²⁺/Ba²⁺, 12-6-4), **redox** (disulfide reduction, CYX→CYS seeding), **external electric field** (static or oscillating), and **cosolvents** (urea / TMAO / GdmCl, CHARMM 2020 parameters). ΔT / Δγ / ΔT(dual-bath) / pressure hot-switch mid-run.
- **Observability** (read-only, `analysis.rs`): probe **electrostatic potential** and its analytic **field**, per-atom **SASA**, PDB-name atom selection, contact counts and channel **bottleneck clearance**, per-term energies + virial buckets, species (solute vs water) temperatures, Rg, net charge, effective ionic strength, a `debug_rigid_scale_probe` for same-config audits, and an optional sliding-window **trend detector** for RL fail-fast.
- **Three frontiers, one core**: the same engine compiles to a **PyO3 extension** (`python` feature), an **rlib** for Rust consumers (`EnginePool`), and a **WebAssembly** module (`web` feature) driven by a hand-rolled `extern "C"` ABI — full PME MD in the browser, no server. See `docs/web_demo.md` and the in-repo demo under `web/` (`api.mjs` Web API + `loader.mjs` ABI wrapper + `index.html` reference page; wasm blobs come from `make web-dist` or a tokenless nightly.link download via `make web-nightly` — never committed; `make web-serve` to try it in a tab).
- **RL actions** (`actions.rs`): `a ∈ R¹⁶` × low-rank basis `W[L,3]×16` → per-residue Cα bias forces (tanh-clamped to ±0.5 kcal/(mol·Å)); `ActionMask` re-randomises a residue subset every 20 steps; `EnvDelta{ΔT, ΔpH}`.
- **Stability-domain search** (`domain.rs`): scans a (T, pH) grid, using M to judge whether the protein keeps its native fold at each environmental point, and outputs the stability domain.

**Five physical metrics M** (`metrics.rs` — the RL state/reward vector):

| metric | meaning |
|---|---|
| m1 | Var(U)/(k_B·T) — potential-energy fluctuation (normalised by thermal noise) |
| m2 | \|Rg − Rg_ref\|/Rg_ref — radius-of-gyration drift |
| m3 | 1 − SS_kept/SS_ref — secondary-structure loss (DSSP-lite) |
| m4 | fraction of heavy-atom pairs with distance/(vdW sum) < 0.6 — clash score |
| m5 | surface ionizable residue actual charge vs pH-ideal charge — surface charge mismatch |

## Repository layout

```
spice_engine/  (crate: spice_engine)
├── src/
│   ├── env.rs        EnvParams: pH / T / P / ionic / divalent / redox / efield / cosolvent
│   ├── topology.rs   protein topology (sequence, Cα/backbone/heavy indices, residue→atom map)
│   ├── builder.rs    system build (pH protonation → H placement → solvation → salt → L-BFGS relax; + mutant solvent-reuse)
│   ├── engine/       ★ the MD hot path (migrated `md_core`: integrator / nonbonded+SPME /
│   │                   constraints / solvent / analysis / utility shims; #[path] chain
│   │                   under engine/core/mod.rs) + SpiceEngine facade (engine.rs)
│   ├── forcefield/   SE-owned force-field tree (SIMD backend, Amber/CHARMM/Martini types)
│   ├── metrics.rs    five physical metrics M (+ Rg / backbone-H-bond proxies)
│   ├── actions.rs    RL actions (force basis + ActionMask + EnvDelta)
│   ├── structure.rs  external structure ingestion (Python in-memory atoms → build)
│   ├── mutate.rs     sequence validation / point mutations
│   ├── pool.rs       EnginePool (parallel workers, MdState is Send)
│   ├── domain.rs     stability-domain grid scan (+ radial scans)
│   ├── progress.rs   indicatif shim: real bars native / no-op on wasm (v1.3.9)
│   ├── web.rs        WebAssembly extern-"C" API (feature `web`, wasm32 only; v1.3.9)
│   └── ffi.rs        PyO3 Python bindings (feature `python`)
├── docs/             capabilities.md (API reference) · web_demo.md (browser port)
├── web/              browser demo — api.mjs (Web API) + api.d.ts · loader.mjs · index.html ·
│                     smoke.mjs / selftest.mjs / api.test.mjs ·
│                     mini.cif fixture · dist/ gitignored (make web-dist / web-nightly)
├── tests/            Rust integration tests (md_smoke / npt_virial / dual_bath /
│                     topology_regression / ion_layout_golden / repack_ions / …)
├── pyproject.toml    maturin packaging config
├── Makefile          native + web-* wasm targets (RUSTFLAGS stays per-recipe)
└── Cargo.toml        cdylib + rlib; features python / inference / network / heavy-forcefields / web
```

Build-dependency forks (`../`):
- `ewald/` (SPICE-Protein fork) — SPME: arm64 gating, molecular exclusions, k-space virial (v0.1.16 `statrs` slimmed for wasm)
- `bio_files` (SPICE-Protein fork) — parser, `network` feature optional so the web build drops ureq/rustls/ring
- `dynamics` crate — upstream lineage, now migrated into `src/engine/`

## Quick start

```bash
# Rust engine + tests (release — MD must run in release)
cd spice_engine
cargo test --release

# Python bindings (conda env spice)
cd spice_engine
CONDA_PREFIX=/path/to/envs/spice VIRTUAL_ENV=/path/to/envs/spice \
  python -m maturin develop --release

# Browser (WebAssembly) — same engine core, no server
make wasm        # → …/wasm32-unknown-unknown/release/spice_engine.wasm (~5 MB / 1.8 MB gz)
make web-smoke   # node correctness gate: instantiate → build → step → read back
make web-dist    # stage web/dist/ for the demo (wasm blobs are NOT committed)
make web-nightly # …or pull the latest CI artifact via nightly.link (no token, no toolchain)
make web-serve   # serve web/ → http://localhost:8080 (api.mjs + loader.mjs + reference page)
make web-apitest # Web API end-to-end incl. all FFI-parity probes
```

### Python usage

```python
import spice_engine as se
import numpy as np

# 1) Structure (production: pass numpy arrays via from_atoms; from_mmcif is the debug path)
struct = se.Structure.from_mmcif("data/test/2LYZ.cif")

# 2) Build the engine (pH → protonation, T → thermostat, ionic strength → salt)
eng = se.Engine.build(struct, ph=7.0, temp=310.0, pressure=1.0,
                      ionic_strength_m=0.0, relax_iters=2000, tolerance=2.0)

# 3) Step once (optional action vector [16]; None = no bias)
out = eng.step(None)          # dict: u_t_kcal, coords_ca, step_count, m1..m5, crashed, ...
out = eng.step(np.zeros(16))  # with bias action

# 4) State / labels
print(eng.metrics())                    # five metrics
print(np.asarray(eng.pseudo_labels()))  # time-averaged Cα (pseudo-labels)
print(np.asarray(eng.coords_ca()))      # current Cα
eng.set_temperature(320.0)        # environment ΔT hot-switch
eng.reset_pseudo_labels()
```

### Browser JS API

The wasm build ships a zero-dependency ESM facade — `web/api.mjs` (typed by
`web/api.d.ts`, guarded end-to-end by `web/api.test.mjs` / `make web-apitest`):

```js
import { Sim } from "./web/api.mjs";

const sim = await Sim.load(cifText, {                       // mmCIF text, same build knobs as Python
  env: { tempK: 300, soluteK: 300, ionicStrengthM: 0.15, boxPadA: 6 },
  wasm: "dist/spice_engine.wasm.gz",                        // or make web-nightly to fetch the CI artifact
});
console.log(sim.info);                                      // { nAtoms, nSites, nWater, netChargeE, ... }

const stop = sim.animate((s, m) => {                        // requestAnimationFrame loop
  const { coords, roles } = s.snapshot();                   // Float32Array/Int32Array copies for your renderer
  draw(coords, roles);
  label.textContent = `${m.u_total_kcal.toFixed(1)} kcal/mol`;
}, { stepsPerFrame: 2 });

sim.set({ tempK: 320, soluteK: 320 });                      // hot dual-bath switch (pass soluteK!)
sim.setForcesOff({ longRange: true });                      // toy-force experiments
stop(); sim.dispose();
```

Runs in browsers and node ≥ 18, no server. Protocol rule inherited from the
engine: always pair `soluteK` with `tempK` — a single bath leaves the solute at
~0.72-0.78× setpoint. From v1.3.10 the wasm also carries the Python FFI's whole
analysis/control surface (ESP & field probes, SASA, selections/contacts/bottleneck,
the five metrics, restraints + SMD ramps, trend fail-fast, integrator switch,
equilibrate, debug dumps — same JSON keys as Python). Raw `extern "C"` ABI and
full contract: `docs/web_demo.md`.

### Stability-domain search

```rust
use spice_engine::{domain::{EnvGrid, StabilityConfig, scan_stability}, ...};

let grid = EnvGrid {
    temps: (280.0..=360.0).step_by(10.0).collect(),
    phs: vec![6.0, 7.0, 8.0],
    ..Default::default()
};
let stab = StabilityConfig::default(); // n_steps, M thresholds
let pts = scan_stability(&dev, &param_set, &structure, &grid, &opts, &stab)?;
// pts: Vec<StabilityPoint { env, stable, metrics }> — parallel stability scan
```

On the Python side you can also drive each point with `Engine.build` + `step` + `metrics` (for SAC); the Rust `domain.rs` grid scan serves as ground truth / batch data generation.

## Current status (v1.3.9)

| phase | content | status |
|---|---|---|
| P0 | dynamics fork compiles + arm64 + salt | ✅ |
| P1 | env/topology/builder/engine | ✅ |
| P2 | metrics (5 dims) + actions bias forces | ✅ |
| P3 | external structure + mutate + EnginePool + pseudo-labels | ✅ |
| P4 | PyO3 FFI + maturin packaging | ✅ |
| — | stability-domain search (domain.rs, parallel grid + radial scans) | ✅ |
| v1.3.x | environment knob set (divalent 12-6-4 / redox / efield / cosolvents) · L-BFGS relaxation · ion-strength audit + genion-style placement · solvent-reuse with index-copy | ✅ |
| v1.3.8 | **pressure closure** (PME molecular exclusions, k-space virial sign, SETTLE constraint virial billed, water-water energy fully booked) · dual-bath thermostat with exact OU per step · trend fail-fast for RL | ✅ |
| v1.3.9 | **WebAssembly browser port** (`web` feature, hand-rolled extern-"C" ABI, dependency-graph surgery — native output bit-identical) | ✅ |
| v1.3.10 | **Web API + Python-FFI parity on wasm** — `web/api.mjs` facade; ESP/field/SASA/metrics/restraints/trend/equilibrate & the rest of the analysis + control surface in the browser tab | ✅ |

**Known limitations**: builder starts are density-calibrated, not perfect — the 2LYZ reference build sits at a roughly constant +12 kbar offset (a flat build-time signature, not a drift; the calibration pass is open work), and very long runs from crystal geometries still want an equilibration ramp before production. Results produced before v1.3.8 predate the fixed force/virial bookkeeping and need revalidation. The wasm tab build is single-threaded and budgeted for small systems (~10–17 ns/day at 1.5 k sites); SharedArrayBuffer threading and WebGPU are explicit non-goals.
