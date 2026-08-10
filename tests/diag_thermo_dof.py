#!/usr/bin/env python
"""Diagnose the +60-75 K offset between reported t_kin and the thermostat target.

Reads exact DOF bookkeeping from the engine: atom/water/H counts vs the cached
`thermo_dof`, and the temperature the cached DOF implies for the current KE.

Expected DOF for 2LYZ-type systems:
  dof = 6*n_water + 3*(n_atoms - n_static) - n_hydrogens(if SHAKE) - 3(if COM removal)
If thermo_dof != expected, t_kin is miscalibrated by that ratio.
"""
import spice_engine as se

TEMP = 310.0


def show(eng, label):
    ti = eng.thermo_info()
    exp_dof_shake_com = (
        ti["dof_water_6n"]
        + ti["dof_solute_3n"]
        - ti["n_hydrogens"]
        - 3  # COM removal (linear)
    )
    exp_dof_no_shake = ti["dof_water_6n"] + ti["dof_solute_3n"] - 3
    print(f"--- {label} ---")
    for k, v in ti.items():
        print(f"  {k:22s} = {v}")
    print(f"  expected_dof (SHAKE + COM)      = {exp_dof_shake_com}")
    print(f"  expected_dof (no SHAKE)         = {exp_dof_no_shake}")
    ratio = ti["thermo_dof"] / exp_dof_shake_com if exp_dof_shake_com else 0.0
    print(f"  thermo_dof / expected           = {ratio:.4f}")
    print(f"  t_kin now                       = {eng.step_md()['t_kin']:.1f} K (target {TEMP})")


struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
show(eng, "right after build")
eng.equilibrate()
show(eng, "after settle + 1 production step")
