#!/usr/bin/env python
"""Staged (hierarchical) stability-domain search.

Stage 1 — fix temperature (and other env params) at the reference, scan pH
         only → the viable pH range (stable + not build_failed).
Stage 2 — within the viable pH range, scan temperature → the temp stability
         window at each viable pH.

This establishes the range first and never wastes temperature scans on pH
values that are not viable at the reference temperature.

Run: /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python tests/py_domain_staged.py
"""
import spice_engine as se

s = se.Structure.from_mmcif("data/test/2LYZ.cif")
ANCHOR_T = 310.0

# ---------- Stage 1: pH screen @ anchor temperature ----------
print(f"### Stage 1: pH screen @ {ANCHOR_T:.0f} K (fix T, scan pH 0-14) ###")
phs = [float(i) for i in range(15)]
pts = se.scan_stability(s, [ANCHOR_T], phs, n_steps=20, equil_steps=10,
                        repeats=3, relax_iters=2000, tolerance=2.0)
by = {p["ph"]: p for p in pts}
viable = sorted(ph for ph in phs if by[ph]["stable"] and not by[ph]["build_failed"])
dead = sorted(ph for ph in phs if ph not in viable)
print(f"\nviable pH @ {ANCHOR_T:.0f} K: {viable}")
print(f"  range: {min(viable):.0f} – {max(viable):.0f}" if viable else "  (none)")
if dead:
    print(f"  non-viable: {dead}")
print("\nper-pH screen detail:")
for ph in phs:
    p = by[ph]
    print(f"  pH={ph:.0f}: stable={p['stable']} crashed={p['crashed']} "
          f"build_failed={p['build_failed']} m1={p['m1']:.3g} m3={p['m3']:.3f}")

if not viable:
    print("\nNo viable pH at the anchor temperature — nothing to scan in Stage 2.")
    raise SystemExit(0)

# ---------- Stage 2: temperature within the viable pH range ----------
ph_min, ph_max = min(viable), max(viable)
print(f"\n### Stage 2: temperature scan within viable pH {ph_min:.0f}–{ph_max:.0f} ###")
pts2 = se.scan_stability_ranges(
    s,
    (253.0, 393.0, 15.0),       # temps: 253..388 (-20..115 °C)
    (float(ph_min), float(ph_max), 1.0),
    n_steps=20,
    equil_steps=10,
    repeats=3,
    relax_iters=2000,
    tolerance=2.0,
)

temps = sorted({p["temp"] for p in pts2})
phs2 = sorted({p["ph"] for p in pts2})
by2 = {(p["temp"], p["ph"]): p for p in pts2}

print(f"\n{'T\\pH':>6} " + " ".join(f"{ph:>6.0f}" for ph in phs2))
for t in temps:
    row = []
    for ph in phs2:
        p = by2[(t, ph)]
        if p["build_failed"]:
            row.append("BF")
        else:
            row.append("S" if p["stable"] else f"U({p['m3']:.1f})")
    print(f"{t:>6.0f} " + " ".join(f"{c:>6}" for c in row))
print("\nlegend: S=stable  U(m3)=unstable  BF=build_failed")
