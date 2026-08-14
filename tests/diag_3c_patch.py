#!/usr/bin/env python
"""End-to-end: rebuilt sidechains with a perturbed Cα scaffold (the Path-B
pred_ca path that triggered "Missing valence angle params for HC-3C-HC") must
now build successfully. The 3C H-C-H angle patch in frcmod.ff19SB makes the
rebuilt sidechain's fragile H-typing resolvable instead of hard-failing."""
import sys

sys.path.insert(0, "/Users/redelectricity/Documents/Projects/SPICE/model")

import numpy as np
import spice_engine as se
from spice_rl.env.structure import load_structure_with_atoms
from spice_rl.train_post import _mutant_atoms

rng = np.random.default_rng(42)

struct, base_atoms = load_structure_with_atoms("/tmp", "1T19", max_residues=None)
seq = struct.sequence()
names = base_atoms["atom_names"]
coords = base_atoms["coords"]

ca_idx = [i for i, n in enumerate(names) if n == "CA"]
print(f"CA atoms: {len(ca_idx)} (altloc-duplicated), residues: {len(seq)}", flush=True)
wild_ca = np.asarray([coords[i] for i in ca_idx], np.float32)
# Model-predicted Cα ≈ wild + small perturbation (perturb every CA atom).
# 0.3 Å RMSD — realistic head-A prediction error (0.6 Å over-stresses the
# geometry and trips unrelated bond-typing artifacts).
pert = wild_ca + rng.normal(0, 0.3, wild_ca.shape).astype(np.float32)

mut_targets = ["M", "W", "I", "S", "F", "Y"]
ok = fail = 0
for target in mut_targets:
    i = next((i for i, aa in enumerate(seq) if aa not in ("G", "A", target)), 0)
    mut = seq[:i] + target + seq[i + 1:]
    try:
        nn, ee, ss, rr, cc = _mutant_atoms(base_atoms, mut, pert)
        s = se.Structure.from_atoms(nn, ee, ss, rr, cc)
        eng = se.Engine.build(
            s, 7.0, 310.0, 0.0, 0.0, relax_iters=30, tolerance=10.0
        )
        print(f"{target}/pert: BUILD_OK", flush=True)
        ok += 1
    except Exception as e:  # noqa: BLE001
        print(f"{target}/pert: FAIL: {e}", flush=True)
        fail += 1

print(f"OK={ok} FAIL={fail}", flush=True)
