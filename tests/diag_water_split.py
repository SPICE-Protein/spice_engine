#!/usr/bin/env python
"""Decisive split: is the +67K NVT offset caused by the WATER thermostat?

Config A: per-atom force-based Langevin on solute + rigid water  (skip=False)
Config B: Langevin on solute ONLY                               (skip=True)

* A ~385K & B ~310-315K  -> water thermostat over-injects (9 noise DOF
                           vs 6 physical water DOF). Fix water treatment.
* A ~385K & B ~385K      -> not water-specific; something deeper.
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 600
EVERY = 100


def run(skip_water: bool, label: str):
    struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
    eng = se.Engine.build(struct, 7.0, TEMP, 0.0, 0.0, relax_iters=2000, tolerance=2.0)
    eng.set_integrator("langevin_strong")  # gamma=10
    if skip_water:
        eng.set_skip_water_thermostat(True)
    ts = []
    for i in range(N_STEPS):
        r = eng.step_md()
        ts.append(r["t_kin"])
        if i % EVERY == 0 or i == N_STEPS - 1:
            print(f"  [{label}] step {i:4d}: t_kin={r['t_kin']:6.1f} K", flush=True)
    print(f"  [{label}] equilibrium t_kin ~ {ts[-1]:.0f} K (target {TEMP})", flush=True)
    return ts[-1]


print("=== Config A: water thermostat ON (per-atom noise) ===", flush=True)
a = run(False, "A")
print("=== Config B: water thermostat OFF (solute-only) ===", flush=True)
b = run(True, "B")
print(f"RESULT: A={a:.0f}K B={b:.0f}K -> "
      f"{'WATER_CULPRIT' if abs(a - b) > 30 else 'NOT_WATER_SPECIFIC'}")
