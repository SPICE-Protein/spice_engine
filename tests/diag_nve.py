#!/usr/bin/env python
"""NVE energy-conservation probe — decides WHERE the production heating comes
from.

Procedure: build 2LYZ @ 310 K, equilibrate (removes initial clamps so we probe
the steady-state integrator), then switch to NVE (VerletVelocity, NO
thermostat) and run 300 steps. Track total energy E = U + KE:

  * E conserved (flat within noise) -> the integrator/forces are fine; the
    heating seen under LangevinMiddle(gamma=0.5) is a thermostat-path issue.
  * E climbs monotonically         -> a NON-conservative force/integrator term
    is injecting energy (SPME-ratio caching, SETTLE/SHAKE, f32, etc.); a weak
    thermostat (gamma=0.5) simply can't remove it fast enough.
"""
import spice_engine as se

N_STEPS = 300
EVERY = 25
TEMP = 310.0

print(f"===== 2LYZ @ {TEMP:.0f} K, NVE probe (after settle) =====", flush=True)
struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
eng.equilibrate()  # default settle: remove initial strain/clamps
eng.set_integrator("nve")  # no thermostat from here on

e0 = None
prev = None
for i in range(N_STEPS):
    r = eng.step_md()
    ke = eng.kinetic_energy_kcal()
    etot = r["u_t_kcal"] + ke
    if e0 is None:
        e0 = etot
    dE = etot - e0
    if i % EVERY == 0 or i == N_STEPS - 1:
        print(
            f"  step {i:4d}: t_kin={r['t_kin']:6.1f} K  u={r['u_t_kcal']:.0f}  "
            f"ke={ke:.0f}  E={etot:.0f}  dE={dE:+.1f}  "
            f"clamps={r['n_clamped']}",
            flush=True,
        )
    prev = (etot, i)

etot, i = prev
dE = etot - e0
print(f"  -> dE over {i} steps = {dE:+.1f} kcal/mol ({dE / max(i, 1):+.2f} kcal/mol/step)")
if abs(dE) < 300.0:
    print("  -> NVE_CONSERVES (thermostat-path problem)")
else:
    print("  -> NVE_LEAKS (integrator/force energy injection)")
