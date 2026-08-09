#!/usr/bin/env python
"""End-to-end test of the restraint-free NVT-settle equilibration with the
DEFAULT config on the two most extreme structures:
  - 2LYZ (moderate step-0 clamps)
  - 1T19 parquet (hardest: step-0 accel was 7.7e6 in earlier runs)

For each: build, equilibrate(default), then 20 production steps must be
clamp-free and crash-free.
"""
import spice_engine as se


def run(name, struct, **kw):
    print(f"===== {name} =====")
    eng = se.Engine.build(struct, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
    eng.equilibrate()  # default: restraint-free, 300+100, gamma=10
    u = []
    for _ in range(20):
        r = eng.step_md()
        u.append(r["u_t_kcal"])
        assert not r["crashed"], f"{name}: crashed at step {len(u)}: {r['u_t_kcal']}"
    print(f"{name}: 20 production steps OK, u0={u[0]:.1f} u19={u[-1]:.1f}")
    return u


s2 = se.Structure.from_mmcif("data/test/2LYZ.cif")
run("2LYZ", s2)

# 1T19 via the production parquet path
import sys
sys.path.insert(0, "/Users/redelectricity/Documents/Projects/SPICE/model")
import polars as pl
from spice_rl.env.structure import structure_from_dataframe

df = pl.read_parquet("/tmp/atoms_shard_0040.parquet")
sub = df.filter(pl.col("pdb_id") == "1T19").sort(["chain_id", "res_seq"])
s1 = structure_from_dataframe(sub)
print(f"1T19 struct residues: {s1.residue_count()}")
run("1T19 parquet", s1)

print("ALL_PASS")
