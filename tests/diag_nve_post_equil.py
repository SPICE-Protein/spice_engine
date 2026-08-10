#!/usr/bin/env python
"""Decisive NVE post-equilibration energy drift test.

If water dynamics has a genuine heat source (SETTLE/OPC/nonbonded), total
E = U + KE will drift hugely in NVE (no thermostat to hide it).
If E is conserved, the thermostat/measurement is the issue.
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 300
EVERY = 50

struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 0.0, 0.0, relax_iters=2000, tolerance=2.0)
eng.equilibrate(ramp_steps=300, t_start_k=100.0, k_restraint=0.0,
                hold_steps=100, restrain_hydrogens=False, friction_gamma=10.0)
eng.set_integrator("nve")

prev_e = None
for i in range(N_STEPS):
    r = eng.step_md()
    ke = eng.kinetic_energy_kcal()
    e = r["u_t_kcal"] + ke
    if prev_e is None:
        prev_e = e
    drift = e - prev_e
    if i % EVERY == 0 or i == N_STEPS - 1:
        print(f"  step {i:4d}: t_kin={r['t_kin']:6.1f}K u={r['u_t_kcal']:10.1f} "
              f"E={e:10.1f} dE(step0)={drift:+10.1f}", flush=True)
    prev_e = e

# overall drift
r0 = None
print("NOTE: dE(step0) is drift vs step 0", flush=True)
