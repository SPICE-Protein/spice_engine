#!/usr/bin/env python
"""Water rigid-body vs internal KE split, post-equilibration NVT.

* internal_t_k ~ 310 (i.e. internal modes carry ~½kBT each) -> water has 9
  effective DOF, SETTLE is NOT projecting velocities -> water_t (6-dof) is
  inflated ~1.5x and the system is really COLD (thermostat under-injects).
* internal_t_k ~ small -> water is rigid (6 DOF), water really hot.
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 200
EVERY = 50

struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 0.0, 0.0, relax_iters=2000, tolerance=2.0)
eng.equilibrate(ramp_steps=300, t_start_k=100.0, k_restraint=0.0,
                hold_steps=100, restrain_hydrogens=False, friction_gamma=10.0)
eng.set_integrator("langevin_strong")
for i in range(N_STEPS):
    r = eng.step_md()
    if i % EVERY == 0 or i == N_STEPS - 1:
        ws = eng.water_rigid_split()
        sp = eng.species_temperatures()
        print(f"  step {i:4d}: total={r['t_kin']:6.1f}K | "
              f"water: rigid_t={ws['water_rigid_t_k']:6.1f} "
              f"9dof_t={ws['water_9dof_t_k']:6.1f} "
              f"internal_t={ws['water_internal_t_k']:6.1f} "
              f"(internal_ke={ws['water_internal_ke_kcal']:.0f}/{ws['water_total_ke_kcal']:.0f}) | "
              f"solute_t={sp['solute_t_k']:6.1f}", flush=True)
