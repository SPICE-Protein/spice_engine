#!/usr/bin/env python
"""P4 smoke test: drive the Rust engine from Python via PyO3.

Run:  /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python tests/py_smoke.py
"""
import numpy as np
import spice_engine

print("version:", spice_engine.version())

# --- structure input (production: Python pipeline feeds from_atoms) ---
s = spice_engine.Structure.from_mmcif("data/test/2LYZ.cif")
print(f"Structure: {s.residue_count()} residues, seq len {len(s.sequence())}")

# --- build engine (solvate + minimize; takes ~1 min in release) ---
e = spice_engine.Engine.build(s, ph=7.0, temp=310.0, pressure=1.0,
                              ionic_strength_m=0.0, relax_iters=2000, tolerance=2.0)
print(f"Engine: n_residues={e.n_residues()}, seq[:30]={e.sequence()[:30]}")

# --- sequence tools ---
m = spice_engine.mutate_sequence(e.sequence(), 0, "A")
print("mutate seq[0] ->", m[0])

# --- drive a few steps, collecting metrics ---
for i in range(5):
    out = e.step(None)  # unbiased step (None action)
    print(f"  step {out['step_count']}: U={out['u_t_kcal']:.1f} kcal/mol "
          f"crashed={out['crashed']} m4={out['m4']:.5f} m5={out['m5']:.3f}")
    if out["crashed"]:
        break

print("metrics dict:", e.metrics())
print("pseudo_labels (time-avg Cα):", np.asarray(e.pseudo_labels()).shape)
print("coords_ca:", np.asarray(e.coords_ca()).shape)
print("mask_fraction:", e.mask_fraction())
print("OK")
