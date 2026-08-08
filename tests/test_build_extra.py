#!/usr/bin/env python
"""Regression: build 1R2I / 5H3G / 4LPX from mmCIF through the same
Engine.build path that failed in batch parquet ingestion.
Prints per-structure success/failure to the log.
"""
import spice_engine as se

STRUCTURES = ["1R2I", "5H3G", "4LPX"]

for name in STRUCTURES:
    print(f"\n===== {name} =====", flush=True)
    try:
        s = se.Structure.from_mmcif(f"data/test/{name}.cif")
        print(f"parsed: {s.residue_count()} res", flush=True)
        eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
        print("BUILD_OK", flush=True)
        for _ in range(10):
            eng.step_md()
        print("10_STEPS_OK u=", eng.u_t_kcal(), flush=True)
    except Exception as e:
        print("BUILD_FAILED", flush=True)
        print(repr(e), flush=True)
