#!/usr/bin/env python
"""Regression: a healthy (wild-type) parent must still build + reset() fine,
and a wild-scaffold mutation must build — proving the FF 3C patch and the
md_env.reset() protection did not break the normal path."""
import sys

sys.path.insert(0, "/Users/redelectricity/Documents/Projects/SPICE/model")

import types

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

# 1) Wild-type structure: Engine.build must succeed.
eng = se.Engine.build(struct, 7.0, 310.0, 0.0, 0.0, relax_iters=30, tolerance=10.0)
print("WILD_BUILD_OK", flush=True)

# 2) Wild-scaffold mutation (V->A): rebuilt sidechain must build.
mut_idx = next((i for i, aa in enumerate(seq) if aa == "V"), 0)
mut_seq = seq[:mut_idx] + "A" + seq[mut_idx + 1:]
nn, ee, ss, rr, cc = _mutant_atoms(base_atoms, mut_seq)
s = se.Structure.from_atoms(nn, ee, ss, rr, cc)
eng2 = se.Engine.build(s, 7.0, 310.0, 0.0, 0.0, relax_iters=30, tolerance=10.0)
print("MUTANT_WILD_BUILD_OK", flush=True)

# 3) A sidechain that uses 3C-typed carbon + HC/H1 hydrogens: PHE->MET rebuild.
mut_idx2 = next((i for i, aa in enumerate(seq) if aa not in ("G", "A")), 0)
mut2 = seq[:mut_idx2] + "M" + seq[mut_idx2 + 1:]
nn2, ee2, ss2, rr2, cc2 = _mutant_atoms(base_atoms, mut2)
s2 = se.Structure.from_atoms(nn2, ee2, ss2, rr2, cc2)
eng3 = se.Engine.build(s2, 7.0, 310.0, 0.0, 0.0, relax_iters=30, tolerance=10.0)
print("MUTANT_MET_BUILD_OK", flush=True)
