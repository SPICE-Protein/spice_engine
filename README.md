# SPICE — Sequence-Protein Interaction under Conditional Environments

> Protein adaptive-evolution simulation platform: uses **reinforcement learning
> (SAC) + all-atom MD** to explore a protein's **stability domain** and adaptive
> evolution paths across a multidimensional environment (temperature / pH /
> ionic strength / pressure).

The Rust engine (this repo) builds on the `dynamics` crate
(Amber ff19SB force field + OPC water + PME) and provides environment-parameterised
MD, five physical metrics, RL action (bias-force) interfaces, external structure
ingestion, a parallel engine pool, and PyO3 Python bindings. The data pipeline
(PDB download/clean → Parquet → HF) lives in the sibling `spice_protein/`
directory.

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

- **Conditional environment**: `EnvParams { ph, temp_k, pressure_bar, ionic_strength_m }`
  sets protonation states / thermostat / salt concentration at build time; ΔT
  supports hot-switching temperature mid-run.
- **Five physical metrics M** (`metrics.rs`, the RL state/reward vector):
  | metric | meaning |
  |---|---|
  | m1 | Var(U)/(k_B·T) — potential-energy fluctuation (normalised by thermal noise) |
  | m2 | \|Rg − Rg_ref\|/Rg_ref — radius-of-gyration drift |
  | m3 | 1 − SS_kept/SS_ref — secondary-structure loss (DSSP-lite) |
  | m4 | fraction of heavy-atom pairs with distance/(vdW sum) < 0.6 — clash score |
  | m5 | surface ionizable residue actual charge vs pH-ideal charge — surface charge mismatch |
- **RL actions** (`actions.rs`): `a ∈ R¹⁶` × low-rank basis `W[L,3]×16` → per-residue
  Cα bias forces (tanh-clamped to ±0.5 kcal/(mol·Å)); `ActionMask` re-randomises a
  residue subset every 20 steps; `EnvDelta{ΔT, ΔpH}`.
- **Stability-domain search** (`domain.rs`): scans a (T, pH) grid, using M to judge
  whether the protein keeps its native fold at each environmental point, and
  outputs the stability domain.

## Repository layout

```
spice_engine/
├── src/
│   ├── env.rs        EnvParams environment parameters
│   ├── topology.rs   protein topology (sequence, Cα/backbone/heavy indices, residue→atom map)
│   ├── builder.rs    system build (pH protonation → H placement → solvation → salt → minimize)
│   ├── engine.rs     SpiceEngine (step / U / pseudo-labels / temperature hot-switch)
│   ├── metrics.rs    five physical metrics M
│   ├── actions.rs    RL actions (force basis + ActionMask + EnvDelta)
│   ├── structure.rs  external structure ingestion (Python in-memory atoms → build)
│   ├── mutate.rs     sequence validation / point mutations
│   ├── pool.rs       EnginePool (parallel workers, MdState is Send)
│   ├── domain.rs     stability-domain grid scan
│   └── ffi.rs        PyO3 Python bindings (feature `python`)
├── tests/            Rust integration tests (md_smoke / p2 / p3) + Python smoke
├── pyproject.toml    maturin packaging config
└── Cargo.toml        cdylib + rlib; [patch] ewald (arm64)
```

Upstream (`../`):
- `dynamics/` — MD engine fork (patched: salt, H-clash, neighbour list, full-field
  minimization, Barostat RNG → Send)
- `ewald/` — SPME fork (arm64 SIMD gating)
- `spice_protein/` — Python data pipeline (PDB → Parquet → HF)

## Quick start

```bash
# Rust engine + tests (release — MD must run in release)
cd spice_engine
cargo test --release

# Python bindings (conda env spice)
cd spice_engine
CONDA_PREFIX=/path/to/envs/spice VIRTUAL_ENV=/path/to/envs/spice \
  python -m maturin develop --release
```

### Python usage

```python
import spice_engine as se
import numpy as np

# 1) Structure (production: numpy arrays from the Python pipeline; mmCIF convenience here)
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

On the Python side you can also drive each point with `Engine.build` + `step` +
`metrics` (for SAC); the Rust `domain.rs` grid scan serves as ground truth /
batch data generation.

## Current status

| phase | content | status |
|---|---|---|
| P0 | dynamics fork compiles + arm64 + salt | ✅ |
| P1 | env/topology/builder/engine | ✅ |
| P2 | metrics (5 dims) + actions bias forces | ✅ |
| P3 | external structure + mutate + EnginePool + pseudo-labels | ✅ |
| P4 | PyO3 FFI + maturin packaging | ✅ |
| — | stability-domain search (domain.rs) | in progress |

**Known limitations**: steepest-descent minimization still leaves ~70–90
kcal/mol/Å residual forces on crystal-strain hotspots, so very long MD runs can
randomly blow up (tens-of-steps scale); the formal fix (positional restraints +
NVT ramp equilibration) is future work. Tests therefore assert short, reliable
stability windows.
