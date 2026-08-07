#!/usr/bin/env python
"""Quick functional check of the v2 TrendDetector wiring: run a tiny grid scan
with trend on, verify the API accepts the new kwargs and that terminated_reason
is present in the output (None for short windows — the detector is inert here,
which is expected)."""
import spice_engine as se

s = se.Structure.from_mmcif("data/test/2LYZ.cif")
pts = se.scan_stability(
    s,
    temps=[310.0, 313.0],
    phs=[7.0],
    n_steps=20,
    equil_steps=10,
    repeats=1,
    relax_iters=2000,
    tolerance=2.0,
    trend_detector=True,
    trend_window=50,
    trend_z_threshold=3.0,
)
for p in pts:
    print(f"T={p['temp']:.1f} pH={p['ph']:.1f} stable={p['stable']} "
          f"crashed={p['crashed']} terminated_reason={p.get('terminated_reason')} "
          f"m3={p.get('m3')}")
print("OK: terminated_reason present:", "terminated_reason" in pts[0])
