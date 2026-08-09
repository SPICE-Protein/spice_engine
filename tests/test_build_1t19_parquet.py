#!/usr/bin/env python
"""Reproduce 1T19 failure via the PRODUCTION parquet path (structure.py)."""
import sys
sys.path.insert(0, "/Users/redelectricity/Documents/Projects/SPICE/model")
import polars as pl
from spice_rl.env.structure import structure_from_dataframe

df = pl.read_parquet("/tmp/atoms_shard_0040.parquet")
sub = df.filter(pl.col("pdb_id") == "1T19").sort(["chain_id", "res_seq"])
print(f"rows: {sub.height}, residues: {sub.select(['chain_id','res_seq']).unique().height}", flush=True)

import spice_engine as se
try:
    s = structure_from_dataframe(sub)
    print(f"struct residues: {s.residue_count()}", flush=True)
    eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
    print("BUILD_OK", flush=True)
    for _ in range(5):
        eng.step_md()
    print("5_STEPS_OK u=", eng.u_t_kcal(), flush=True)
except Exception as e:
    print("BUILD_FAILED", flush=True)
    print(repr(e), flush=True)
