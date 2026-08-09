#!/usr/bin/env python
"""Diagnose 1T19's hard-clash atoms: are the step-0 clamp atoms added H's
(fixable by raising CLASH_DIST) or heavy atoms (crystal overlap)?"""
import sys
sys.path.insert(0, "/Users/redelectricity/Documents/Projects/SPICE/model")
import polars as pl
from spice_rl.env.structure import structure_from_dataframe
import spice_engine as se

df = pl.read_parquet("/tmp/atoms_shard_0040.parquet")
sub = df.filter(pl.col("pdb_id") == "1T19").sort(["chain_id", "res_seq"])
s = structure_from_dataframe(sub)
eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)

labels = eng.atom_labels()
rep = eng.clash_report(500.0)  # AFTER build (forces zeroed by finish_minimize? -> may be 0)
print(f"post-build clash_report (forces likely zeroed): {len(rep)}")

# Run one step so forces reflect step 0, then report
r = eng.step_md()
print(f"step0 crashed={r['crashed']} u={r['u_t_kcal']:.1f}")

rep = eng.clash_report(500.0)
print(f"\n{len(rep)} hot H's after step0 (|F|>500):")
for (e, res, seq, sn, fmag, min_d) in rep[:20]:
    print(f"  H res {res} seq {seq:3d} serial {sn:4d} |F|={fmag:9.1f} min_d={min_d:5.3f}")

# Identity of the atoms seen in clamp warnings (indices 296, 400, 432, 1137, 1268...)
print("\nclamped-index identities:")
for i in [296, 400, 432, 1137, 1268, 1342, 1334, 1335, 405, 1127, 1798]:
    if i < len(labels):
        e, res, seq, sn = labels[i]
        print(f"  {i:5d}: {e:12s} res {res} seq {seq:3d} serial {sn}")
