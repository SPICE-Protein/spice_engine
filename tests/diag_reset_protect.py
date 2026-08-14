#!/usr/bin/env python
"""Verify md_env.reset() now fails SAFELY for a structurally-broken mutant
parent: it raises a catchable RuntimeError (engine=None, _needs_rebuild=True)
instead of letting the raw spice_engine ValueError abort the RL loop."""
import sys

sys.path.insert(0, "/Users/redelectricity/Documents/Projects/SPICE/model")

import types

import numpy as np
import spice_engine as se
from spice_rl.env.md_env import MDSimulationEnv
from spice_rl.env.structure import load_structure_with_atoms
from spice_rl.train_post import _mutant_atoms

cfg = types.SimpleNamespace(
    force_dim=16,
    env_offset_dim=3,
    mutation_every=5,
    u_window=20,
    relax_iters=30,
    tolerance=10.0,
    ph_rebuild_threshold=0.5,
    strict_incomplete=True,
)

struct, base_atoms = load_structure_with_atoms("/tmp", "1T19")
seq = struct.sequence()
names = base_atoms["atom_names"]
coords = base_atoms["coords"]
ca_idx = [i for i, n in enumerate(names) if n == "CA"]
wild_ca = np.asarray([coords[i] for i in ca_idx], np.float32)
# 0.6 Å aggressive perturbation -> guaranteed FF-param failure (XC-HC / O-HP).
rng = np.random.default_rng(7)
pert = wild_ca + rng.normal(0, 0.6, wild_ca.shape).astype(np.float32)

i = next((i for i, aa in enumerate(seq) if aa not in ("G", "A", "M")), 0)
mut = seq[:i] + "M" + seq[i + 1:]
nn, ee, ss, rr, cc = _mutant_atoms(base_atoms, mut, pert)
s = se.Structure.from_atoms(nn, ee, ss, rr, cc)

env = MDSimulationEnv(
    s, cfg, ph=7.0, temp=310.0, ionic=0.0, pressure=0.0, reuse_engine=False
)
try:
    env.reset()
    print("RESET_OK (unexpected for broken structure)", flush=True)
except RuntimeError as e:  # noqa: BLE001
    print(f"RESET_RUNTIME_ERROR (expected, RL-safe): {str(e)[:120]}", flush=True)
    print(f"engine is None: {env.engine is None}", flush=True)
    print(f"_needs_rebuild: {env._needs_rebuild}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"RESET_OTHER_EXC (BAD): {type(e).__name__}: {str(e)[:120]}", flush=True)
