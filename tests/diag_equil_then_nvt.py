#!/usr/bin/env python
"""Decisive: is the +67K NVT offset a THERMOSTAT bug, or an artifact of
running production MD WITHOUT equilibrating first (huge initial strain ->
massive heat dump -> thermostat can't fully recover)?

  * build -> equilibrate() -> 800 NVT(gamma=10) -> ~310K : OFFSET WAS
    THE UN-EQUILIBRATED HOT START; production (which equilibrates) is fine.
  * still -> ~385K : real thermostat bug (would need noise/DOF audit).
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 800
EVERY = 100

struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 0.0, 0.0, relax_iters=2000, tolerance=2.0)

# Release the initial build strain via the restraint-free NVT settle ramp.
eng.equilibrate(ramp_steps=300, t_start_k=100.0, k_restraint=0.0,
                hold_steps=100, restrain_hydrogens=False, friction_gamma=10.0)
print("equilibrate done", flush=True)

eng.set_integrator("langevin_strong")  # gamma=10, production-like
ts = []
for i in range(N_STEPS):
    r = eng.step_md()
    ts.append(r["t_kin"])
    if i % EVERY == 0 or i == N_STEPS - 1:
        print(f"  step {i:4d}: t_kin={r['t_kin']:6.1f} K  u={r['u_t_kcal']:.0f}", flush=True)

t_last = ts[-1]
print(f"  -> post-equilibration NVT(gamma=10) t_kin ~ {t_last:.0f} K (target {TEMP}) | "
      f"{'NVT_OK' if abs(t_last - TEMP) < 20.0 else 'STILL_HOT'}")
