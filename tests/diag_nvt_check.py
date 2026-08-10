#!/usr/bin/env python
"""Decisive: is the +65-70K offset from the CRescale BAROSTAT (NPT) not the
thermostat? Build with pressure=0.0 (barostat_cfg=None => TRUE NVT) and run
gamma=10 to equilibrium.

  * equilibrium ~ target (310)  -> barostat was the culprit; production must
    run NVT (pressure=0). No thermostat rewrite needed.
  * equilibrium still ~370     -> thermostat itself, back to the drawing board.
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 800
EVERY = 100

# pressure=0.0 => builder sets barostat_cfg=None (true NVT)
struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 0.0, 0.0, relax_iters=2000, tolerance=2.0)
eng.set_integrator("langevin_strong")  # gamma=10, no settle, long run

ts = []
for i in range(N_STEPS):
    r = eng.step_md()
    ts.append(r["t_kin"])
    if i % EVERY == 0 or i == N_STEPS - 1:
        print(f"  step {i:4d}: t_kin={r['t_kin']:6.1f} K  u={r['u_t_kcal']:.0f}", flush=True)

t_last = ts[-1]
print(f"  -> NVT(gamma=10) equilibrium t_kin ≈ {t_last:.0f} K (target {TEMP}) | "
      f"{'NVT_OK' if abs(t_last - TEMP) < 20.0 else 'STILL_HOT'}")
