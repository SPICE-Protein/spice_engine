#!/usr/bin/env python
"""CLEAN water split (strain released first): build -> equilibrate -> NVT.

If water per-atom noise over-injects (9 noise DOF vs 6 physical water DOF):
  A (water on)  ~ 385K
  B (water off) ~ 310K
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 500
EVERY = 100


def run(skip_water: bool, label: str):
    struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
    eng = se.Engine.build(struct, 7.0, TEMP, 0.0, 0.0, relax_iters=2000, tolerance=2.0)
    eng.equilibrate(ramp_steps=300, t_start_k=100.0, k_restraint=0.0,
                    hold_steps=100, restrain_hydrogens=False, friction_gamma=10.0)
    if skip_water:
        eng.set_skip_water_thermostat(True)
    eng.set_integrator("langevin_strong")
    ts = []
    for i in range(N_STEPS):
        r = eng.step_md()
        ts.append(r["t_kin"])
        if i % EVERY == 0 or i == N_STEPS - 1:
            print(f"  [{label}] step {i:4d}: t_kin={r['t_kin']:6.1f} K", flush=True)
    print(f"  [{label}] eq ~ {ts[-1]:.0f} K (target {TEMP})", flush=True)
    return ts[-1]


print("=== A: water thermostat ON ===", flush=True)
a = run(False, "A")
print("=== B: water thermostat OFF ===", flush=True)
b = run(True, "B")
print(f"RESULT: A={a:.0f}K B={b:.0f}K -> {'WATER_OVERINJECT' if abs(a-b) > 30 else 'NOT_WATER'}")
