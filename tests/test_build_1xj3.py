#!/usr/bin/env python
"""Reproduce 1XJ3 'Invalid parent type SD' failure via mmCIF + parquet paths."""
import sys
sys.path.insert(0, "/Users/redelectricity/Documents/Projects/SPICE/model")
import spice_engine as se


def try_build(name, struct):
    try:
        eng = se.Engine.build(struct, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
        for _ in range(5):
            eng.step_md()
        print(f"{name}: BUILD_OK + 5 steps u={eng.u_t_kcal():.0f}", flush=True)
    except Exception as e:
        print(f"{name}: BUILD_FAILED: {e}", flush=True)


# mmCIF path
s = se.Structure.from_mmcif("data/test/1XJ3.cif")
print(f"mmCIF: parsed {s.residue_count()} res", flush=True)
try_build("mmCIF", s)

# parquet path (production) — need the shard; download if present
import polars as pl
import os
try:
    if not os.path.exists("/tmp/atoms_shard_0022.parquet"):
        import urllib.request
        print("downloading shard 0022...", flush=True)
        urllib.request.urlretrieve(
            "https://hf-mirror.com/datasets/SPICE-Protein/spice_protein/resolve/main/atoms_shard_0022.parquet",
            "/tmp/atoms_shard_0022.parquet",
        )
    df = pl.read_parquet("/tmp/atoms_shard_0022.parquet")
    sub = df.filter(pl.col("pdb_id") == "1XJ3").sort(["chain_id", "res_seq"])
    print(f"parquet: {sub.height} rows, {sub.select(['chain_id','res_seq']).unique().height} res", flush=True)
    from spice_rl.env.structure import structure_from_dataframe
    try_build("parquet", structure_from_dataframe(sub))
except Exception as e:
    print(f"parquet: SKIP ({e})", flush=True)
