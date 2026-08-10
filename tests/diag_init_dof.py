#!/usr/bin/env python
"""Three-way root-cause probe for the +67K NVT offset.

1. INIT:  what is t_kin immediately after build (before any step)?
          -> if ~473K, initialize_velocities(310) is wrong.
2. DOF:   thermo_dof vs dof_for_thermo_now (stale cache?).
3. NVE:   per-step energy drift over 200 steps from the SAME start
          -> sustained heat source if E rises monotonically.
"""
import spice_engine as se

TEMP = 310.0
struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 0.0, 0.0, relax_iters=2000, tolerance=2.0)

ti = eng.thermo_info()
for k in ["n_atoms", "n_static", "n_hydrogens", "n_water", "thermo_dof",
          "dof_for_thermo_now", "dof_water_6n", "dof_solute_3n",
          "kinetic_energy_kcal", "t_implied_k"]:
    print(f"{k:22s} = {ti.get(k)}")

# NVE drift probe
eng.set_integrator("nve")
prev = None
for i in range(200):
    r = eng.step_md()
    if i % 50 == 0 or i in (0, 1, 2, 5, 199):
        print(f"  NVE step {i:3d}: t_kin={r['t_kin']:6.1f}K u={r['u_t_kcal']:.1f}", flush=True)
