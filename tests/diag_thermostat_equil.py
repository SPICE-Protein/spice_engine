#!/usr/bin/env python
"""Decisive: does the gamma=10 thermostat equilibrium == target?

Build 2LYZ @ 310, do NOT settle. Set integrator to LangevinMiddle(gamma=10)
and run 1000 steps from the (hot, ~420K) build state. Where does t_kin
converge?

  * ~310 -> thermostat equilibrium is correct; the +60K offset comes from the
           settle logic (ramp lag / insufficient hold). Fix the settle.
  * ~370 -> thermostat equilibrium itself is ~20% hot. Fix the thermostat.
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 1000
EVERY = 100

struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
eng.set_integrator("langevin_strong")  # gamma=10, no settle, run long

ts = []
for i in range(N_STEPS):
    r = eng.step_md()
    ts.append(r["t_kin"])
    if i % EVERY == 0 or i == N_STEPS - 1:
        print(f"  step {i:4d}: t_kin={r['t_kin']:6.1f} K  n_clamped={r['n_clamped']}  "
              f"u={r['u_t_kcal']:.0f}", flush=True)

t_last = ts[-1]
print(f"  -> gamma=10 equilibrium t_kin ≈ {t_last:.0f} K (target {TEMP}) | "
      f"{'EQUILIBRIUM_OK' if abs(t_last - TEMP) < 20.0 else 'EQUILIBRIUM_HOT'}")
