#!/usr/bin/env python
"""Bidirectional stability-domain probe from an anchor environment.

Starts from the (assumed stable) anchor (e.g. 310 K, pH 7.0) and walks each
axis outward in both + and − directions until the system is judged unstable /
a build fails / max_steps. Each ray (axis × direction) runs in parallel in Rust.

This is the boundary-search view: it finds *where* the protein stops being
stable along each SPICE dimension, instead of sampling a full grid.

Run: /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python tests/py_domain_radial.py
"""
import spice_engine as se

s = se.Structure.from_mmcif("data/test/2LYZ.cif")
print(f"radial stability probe for {s.residue_count()}-residue lysozyme "
      "from (310 K, pH 7.0) anchor...")

# Phase 1 (cheap, template-reusable): temperature axis only. pH is skipped
# because changing pH re-protonates the system → every pH point needs a full
# rebuild (the expensive case), so it is run separately.
print("### Phase 1: temperature axis (template reuse, cheap) ###")
rays = se.scan_radial(
    s,
    anchor_temp=310.0, anchor_ph=7.0,
    temp_step=10.0, temp_max=12, temp_precision=5.0,  # coarse-doubling + bisect ±5 K
    ph_step=None,                                     # skip pH
    n_steps=20, tolerance=2.0,
)

print(f"\n{'axis':>8} {'dir':>2} {'n_stable':>8} {'last stable':>20} {'first unstable':>20}")


def fmt(e):
    return f"T={e['temp']:.0f} pH={e['ph']:.1f}" if e else "?"


for r in rays:
    bs, fu = r["boundary_stable"], r["first_unstable"]
    print(f"{r['axis']:>8} {r['direction']:>2} {r['n_stable']:>8} "
          f"{fmt(bs):>20} {fmt(fu):>20}")

print("\nper-point detail (first ray):")
for p in rays[0]["points"]:
    print("  ", p)
