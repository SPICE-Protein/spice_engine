#!/usr/bin/env python
"""Full 2D stability-domain map (temperature × pH).

Unlike the radial scan (which takes slices through the anchor), this maps the
actual 2D region: for EVERY pH it scans all temperatures. Internally
`scan_stability` groups points by build-key (pH / pressure / ionic), so each pH
builds ONE solvated+minimized template and reuses it across all its
temperatures — the LAMMPS-style "velocity create + fix nvt" sweep.

Run: /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python tests/py_domain_grid.py
"""
import spice_engine as se

s = se.Structure.from_mmcif("data/test/2LYZ.cif")
print(f"2D stability domain: {s.residue_count()}-res lysozyme, "
      "temp × pH grid...")

# Full biologically sensible domain: temperature 253-393 K (-20..120 °C, covers
# psychrophile → hyperthermophile) × pH 0-14 (most protein acid/alkali limits).
# Protonation is discrete, so pH step 1.0; temp step 15 K.
pts = se.scan_stability_ranges(
    s,
    (253.0, 393.0, 15.0),   # temps: 253, 268, ..., 388
    (0.0, 14.0, 1.0),       # phs:   0, 1, ..., 14
    n_steps=20,
    equil_steps=10,
    repeats=3,
    relax_iters=2000,
    tolerance=2.0,
)

# Assemble into a (temp × ph) matrix.
temps = sorted({p["temp"] for p in pts})
phs = sorted({p["ph"] for p in pts})
by = {(p["temp"], p["ph"]): p for p in pts}

# pH below ~3.9 (Asp pKa) protonates ALL acidic sidechains: the model applies
# only discrete protonation (+ counterions) — it has no acid-hydrolysis /
# real-denaturation pathway, so extreme-pH "stability" is MODEL EXTRAPOLATION,
# not physical. Flag it instead of presenting it as real.
EXTRAP_PH_MIN, EXTRAP_PH_MAX = 0.0, 3.0

matrix_lines = [f"{'T\\pH':>6} " + " ".join(f"{ph:>6.1f}" for ph in phs)]
for t in temps:
    row = []
    for ph in phs:
        p = by[(t, ph)]
        if p["build_failed"]:
            row.append("   BF")
        elif EXTRAP_PH_MIN <= ph <= EXTRAP_PH_MAX:
            # Model extrapolation: protonated-catastrophe regime, untrustworthy.
            row.append("    E")
        elif p["stable"]:
            row.append("    S")
        else:
            row.append(f"U({p['m3']:.2f})")
    matrix_lines.append(f"{t:>6.0f} " + " ".join(f"{c:>6}" for c in row))
matrix_text = "\n".join(matrix_lines)
print("\n" + matrix_text)

# Persist the map so it survives terminal/grep filtering.
out_path = "/tmp/stability_domain.txt"
with open(out_path, "w") as f:
    f.write(matrix_text + "\n")
print(f"\nsaved stability-domain matrix to {out_path}")

print("\nlegend: S=stable  U=unstable(m3)  BF=build_failed")
print(f"        E=MODEL EXTRAPOLATION (pH {EXTRAP_PH_MIN:.0f}-{EXTRAP_PH_MAX:.0f}: "
      "all Asp/Glu protonated; coarse protonation has no acid-denaturation "
      "pathway — do NOT treat as real acid stability)")
print("\nper-point metrics (temp=300 / 320 / 380 rows):")
for t in (300.0, 320.0, 380.0):
    if t not in temps:
        continue
    for ph in phs:
        p = by[(t, ph)]
        print(f"  T={t:.0f} pH={ph:.1f}: stable={p['stable']} crashed={p['crashed']} "
              f"m1={p['m1']:.3g} m2={p['m2']:.3f} m3={p['m3']:.3f} "
              f"m4={p['m4']:.4f} m5={p['m5']:.3f}")
