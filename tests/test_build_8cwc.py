#!/usr/bin/env python
"""Reproduce the quick-screen XC-N-XC build failure on a real mmCIF (8CWC).
Builds via the same Engine.build path; prints success/failure to a log file.
"""
import spice_engine as se

try:
    s = se.Structure.from_mmcif("data/test/8CWC.cif")
    print(f"parsed: {s.residue_count()} res", flush=True)
    eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
    print("BUILD_OK", flush=True)
    for _ in range(30):
        eng.step_md()
    print("30_STEPS_OK u=", eng.u_t_kcal(), flush=True)
except Exception as e:
    print("BUILD_FAILED", flush=True)
    print(repr(e), flush=True)
