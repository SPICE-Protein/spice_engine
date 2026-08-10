#!/usr/bin/env python
"""Per-species temperature split (solute vs water) after equilibrate.

Tells us WHICH species the thermostat over-heats:
  * solute_t ~ 310 & water_t ~ 380+  -> water thermostat over-injects
  * solute_t ~ 380 & water_t ~ 310   -> solute thermostat over-injects
  * both ~ 380                       -> global calibration error
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 400
EVERY = 50

struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 0.0, 0.0, relax_iters=2000, tolerance=2.0)
eng.equilibrate(ramp_steps=300, t_start_k=100.0, k_restraint=0.0,
                hold_steps=100, restrain_hydrogens=False, friction_gamma=10.0)
eng.set_integrator("langevin_strong")
for i in range(N_STEPS):
    r = eng.step_md()
    if i % EVERY == 0 or i == N_STEPS - 1:
        sp = eng.species_temperatures()
        print(f"  step {i:4d}: total={r['t_kin']:6.1f}K  "
              f"solute={sp['solute_t_k']:6.1f}K(dof={sp['solute_dof']:.0f})  "
              f"water={sp['water_t_k']:6.1f}K(dof={sp['water_dof']:.0f})", flush=True)
