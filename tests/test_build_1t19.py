#!/usr/bin/env python
"""Reproduce the 1T19 'Atom missing FF type' H(HD2) failure."""
import spice_engine as se

try:
    s = se.Structure.from_mmcif("data/test/1T19.cif")
    print(f"parsed: {s.residue_count()} res", flush=True)
    eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
    print("BUILD_OK", flush=True)
    for _ in range(5):
        eng.step_md()
    print("5_STEPS_OK u=", eng.u_t_kcal(), flush=True)
except Exception as e:
    print("BUILD_FAILED", flush=True)
    print(repr(e), flush=True)
