#!/usr/bin/env python
"""Identify the persistently-clamped hot atoms in 2LYZ."""
import spice_engine as se

s = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
labels = eng.atom_labels()

# Atom indices observed in clamp warnings for 2LYZ across runs.
hot = {1002, 1067, 1113, 1158, 1254, 1283, 1349, 1350, 1416, 1431, 1459, 1520, 1521, 1522, 1524, 1525, 1545}
print("hot atom identities (index: element, res, seq_id, serial):")
for i in sorted(hot):
    if i < len(labels):
        e, res, seq, sn = labels[i]
        print(f"  {i:5d}: {e:12s} res {res} seq {seq:3d} serial {sn}")
