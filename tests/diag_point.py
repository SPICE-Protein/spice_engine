#!/usr/bin/env python
"""Diagnose a single anchor point after the equil_steps fix.

Builds lysozyme at (310 K, pH 7.0), runs equil_steps=10 + n_steps=20, and prints
the terminal metrics — used to calibrate the m1 energy-fluctuation threshold now
that the initial equilibration spike is excluded.
"""
import spice_engine as se

s = se.Structure.from_mmcif("data/test/2LYZ.cif")
print(f"repeat-segment majority vote: {s.residue_count()}-res lysozyme @ 310K/pH7 ...")
pts = se.scan_stability(s, [310.0], [7.0], n_steps=20, equil_steps=10,
                        repeats=3, relax_iters=5000, tolerance=0.5)
p = pts[0]
print(f"  verdict: stable={p['stable']} crashed={p['crashed']} "
      f"build_failed={p['build_failed']} m1={p['m1']:.3g} m2={p['m2']:.3f} "
      f"m3={p['m3']:.3f} m4={p['m4']:.4f} m5={p['m5']:.3f}")
