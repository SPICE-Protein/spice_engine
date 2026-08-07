#!/usr/bin/env python
"""Capture the exact equilibration failure message."""
import spice_engine as se

s = se.Structure.from_mmcif("data/test/2LYZ.cif")
try:
    eng = se.Engine.build(s, ph=7.0, temp=310.0, pressure=0.0, ionic_strength_m=0.0,
                          relax_iters=2000, tolerance=2.0)
    print("build OK; steps:", eng.step(None)["step_count"])
except Exception as e:
    print("BUILD FAILED:", repr(e))
