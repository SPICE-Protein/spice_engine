#!/usr/bin/env python
"""Bisect the +70K thermostat offset: is the rigid-WATER per-atom noise the
culprit? Build NVT @310, gamma=10, equilibrate (expect ~380), then turn OFF
the water thermostat and see where temperature settles.

  * drops toward 310  -> water noise is the culprit (fix: rigid-body Langevin)
  * stays ~380        -> culprit is elsewhere (solute SHAKE / measurement)
"""
import spice_engine as se

TEMP = 310.0
N = 600
EVERY = 100

struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
# pressure=0 => true NVT (no barostat)
eng = se.Engine.build(struct, 7.0, TEMP, 0.0, 0.0, relax_iters=2000, tolerance=2.0)
eng.set_integrator("langevin_strong")  # gamma=10

print("=== phase 1: water thermostat ON (expect ~380) ===", flush=True)
ts1 = []
for i in range(N):
    r = eng.step_md()
    ts1.append(r["t_kin"])
    if i % EVERY == 0 or i == N - 1:
        print(f"  step {i:4d}: t_kin={r['t_kin']:6.1f} K", flush=True)
eq1 = sum(ts1[-200:]) / 200

print("=== phase 2: water thermostat OFF ===", flush=True)
eng.set_skip_water_thermostat(True)
ts2 = []
for i in range(N):
    r = eng.step_md()
    ts2.append(r["t_kin"])
    if i % EVERY == 0 or i == N - 1:
        print(f"  step {i:4d}: t_kin={r['t_kin']:6.1f} K", flush=True)
eq2 = sum(ts2[-200:]) / 200

print(f"  -> eq with water thermo ON  = {eq1:.0f} K")
print(f"  -> eq with water thermo OFF = {eq2:.0f} K (target {TEMP})")
if abs(eq2 - TEMP) < abs(eq1 - TEMP) - 20:
    print("  -> WATER_THERMOSTAT_IS_CULPRIT")
else:
    print("  -> WATER_THERMOSTAT_NOT_CULPRIT")
