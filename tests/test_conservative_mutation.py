#!/usr/bin/env python
"""End-to-end test of conservative-mutation filter on real 1T19 data."""
import sys
sys.path.insert(0, "/Users/redelectricity/Documents/Projects/SPICE/model")
import spice_engine as se
import polars as pl
from spice_rl.env.structure import structure_from_dataframe, load_structure_with_atoms
from spice_rl.train_post import _mutant_atoms

# Load real 1T19 base_atoms from the parquet shard
struct, base_atoms = load_structure_with_atoms("/tmp", "1T19", max_residues=None)
seq = struct.sequence()
print(f"1T19 sequence ({len(seq)}): {seq}", flush=True)

# Find a residue to mutate: pick a VAL (mutate to ALA = conservative)
mut_idx = next((i for i, aa in enumerate(seq) if aa == "V"), 0)
mut_seq = seq[:mut_idx] + "A" + seq[mut_idx + 1:]
print(f"conservative mutation at {mut_idx}: {seq[mut_idx]}->A", flush=True)

# Build the conservative mutant
try:
    names, elems, seqs, resnames, coords = _mutant_atoms(base_atoms, mut_seq)
    s = se.Structure.from_atoms(names, elems, seqs, resnames, coords)
    eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
    for _ in range(5):
        eng.step_md()
    print("CONSERVATIVE MUTANT: BUILD_OK + 5 steps", flush=True)
except Exception as e:
    print(f"CONSERVATIVE MUTANT FAILED: {e}", flush=True)

# Now a NON-conservative mutation: mutate a PHE to TRP (or any residue to PHE)
mut_idx2 = next((i for i, aa in enumerate(seq) if aa != "A"), 0)
mut_seq2 = seq[:mut_idx2] + "W" + seq[mut_idx2 + 1:]
try:
    _mutant_atoms(base_atoms, mut_seq2)
    print("NONCONSERVATIVE: did not raise (unexpected)", flush=True)
except ValueError as e:
    print(f"NONCONSERVATIVE correctly rejected: {e}", flush=True)
