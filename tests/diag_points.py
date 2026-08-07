#!/usr/bin/env python
"""Reusable stability single-point diagnostics.

Two modes:

* `--isolated` (default): each point is a FULLY INDEPENDENT build — no template
  reuse — so the verdict reflects the raw system, not scan/template state
  carryover. Use this to separate real stability boundaries from scan artifacts
  (e.g. cold-side template-reuse false crashes).

* `--reuse`: for each pH, all its temperatures are passed in ONE
  `scan_stability` call so the solvated+minimized template is reused across the
  temperature column (the LAMMPS-style sweep). Use this to verify template
  reuse itself (e.g. that restoring positions+box per point fixed cold false
  crashes).

Examples:
  # Isolated single points (fresh build each):
  python tests/diag_points.py
  python tests/diag_points.py --ph 7 --temp 283 --temp 298

  # Template-reuse column scan at pH 7 over the cold range:
  python tests/diag_points.py --reuse --ph 7 --temp 253 --temp 268 --temp 283 --temp 298
"""
import argparse
import spice_engine as se

# Cold-side diagnostic set: pH 7 (should be cold-stable) + very-alkaline pH.
DEFAULT_POINTS = [
    (7.0, 253.0), (7.0, 268.0), (7.0, 283.0), (7.0, 298.0),
    (12.0, 283.0), (13.0, 283.0), (13.0, 298.0), (14.0, 283.0), (14.0, 298.0),
]


def main():
    ap = argparse.ArgumentParser(description="Stability single-point diagnostics")
    ap.add_argument("--ph", type=float, action="append", help="pH value (repeatable)")
    ap.add_argument("--temp", type=float, action="append", help="Temp in K (repeatable)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--adaptive", action="store_true",
                    help="enable adaptive repeats (default: fixed 3-repeat majority)")
    ap.add_argument("--mode", choices=["isolated", "reuse"], default="isolated",
                    help="isolated=fresh build per point; reuse=template across temps per pH")
    ap.add_argument("--reuse", action="store_true",
                    help="shorthand for --mode reuse")
    ap.add_argument("--structure", default="data/test/2LYZ.cif")
    args = ap.parse_args()
    if args.reuse:
        args.mode = "reuse"

    if args.ph and args.temp:
        points = [(ph, t) for ph in args.ph for t in args.temp]
    else:
        points = DEFAULT_POINTS

    s = se.Structure.from_mmcif(args.structure)
    kw = dict(n_steps=20, equil_steps=10, repeats=args.repeats,
              relax_iters=2000, tolerance=2.0, adaptive_repeats=args.adaptive)

    print(f"=== {args.mode} mode, {len(points)} points, repeats={args.repeats}, "
          f"adaptive={args.adaptive} ===", flush=True)

    if args.mode == "reuse":
        # Group by pH so each pH column shares one template across its temps.
        by_ph = {}
        for ph, t in points:
            by_ph.setdefault(ph, []).append(t)
        for ph, temps in by_ph.items():
            pts = se.scan_stability(s, temps, [ph], **kw)
            for p in pts:
                print_pt(p, ph, p["temp"])
    else:
        for ph, t in points:
            p = se.scan_stability(s, [t], [ph], **kw)[0]
            print_pt(p, ph, t)

    print("=== done ===")


def print_pt(p, ph, t):
    print(f"pH={ph:.0f} T={t:.0f}: stable={p['stable']} crashed={p['crashed']} "
          f"build_failed={p['build_failed']} m1={p['m1']:.3g} m2={p['m2']:.3f} "
          f"m3={p['m3']:.3f}", flush=True)


if __name__ == "__main__":
    main()
