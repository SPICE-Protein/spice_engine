#!/usr/bin/env python
"""Stability-domain scan over a (temp × pH) grid — parallel in Rust.

Run:  /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python tests/py_domain.py
"""
import spice_engine as se

s = se.Structure.from_mmcif("data/test/2LYZ.cif")
print(f"scanning stability domain for {s.residue_count()}-residue lysozyme...")

# Each SPICE axis uses its own resolution: fine temperature steps (10 K) but
# coarse pH steps (1.0 — protonation states are discrete, no high precision
# needed). Here: temp 290–330 @ 10 K (5 pts) × pH 6.0–8.0 @ 1.0 (3 pts) = 15 pts.
pts = se.scan_stability_ranges(s, (290.0, 330.0, 10.0), (6.0, 8.0, 1.0),
                               n_steps=20, relax_iters=None, tolerance=2.0)

print(f"\n{'T(K)':>5} {'pH':>4} {'stable':>7} {'crash':>6} {'build':>6} {'m2 RgΔ':>8} "
      f"{'m3 SS':>7} {'m4 clash':>9} {'m5 surfQ':>8}")
for p in pts:
    m2 = "n/a" if p["crashed"] else f"{p['m2']:.3f}"
    m3 = "n/a" if p["crashed"] else f"{p['m3']:.3f}"
    m4 = "n/a" if p["crashed"] else f"{p['m4']:.5f}"
    m5 = "n/a" if p["crashed"] else f"{p['m5']:.3f}"
    print(f"{p['temp']:5.0f} {p['ph']:4.1f} {str(p['stable']):>7} "
          f"{str(p['crashed']):>6} {str(p['build_failed']):>6} {m2:>8} {m3:>7} {m4:>9} {m5:>8}")

n_stable = sum(1 for p in pts if p["stable"])
print(f"\nstable points: {n_stable}/{len(pts)}")
