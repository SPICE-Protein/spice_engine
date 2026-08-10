#!/usr/bin/env python
"""Characterize the thermostat equilibrium offset empirically.

One build, then sweep (gamma, target) live (set_temperature only changes the
thermostat target; the system re-equilibrates). Records the settled t_kin at
each corner. The SHAPE of the offset (additive / proportional / gamma-scaled)
points to the mechanism:

  * additive  (~ target + C)            -> constant energy injection
  * proportional (~ target * f)         -> noise/friction calibration factor
  * gamma-scaled (~ target + gamma*C')  -> constraint-thermostat interaction
"""
import spice_engine as se

N_SETTLE = 600      # steps to reach equilibrium at each corner
N_AVG = 200         # last N steps to average

struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
eng.equilibrate()  # clean, settled start

GAMMAS = [10.0, 2.0, 0.5]
TARGETS = [200.0, 310.0, 380.0, 450.0]

results = {}
for g in GAMMAS:
    mode = "langevin_strong" if g == 10.0 else "langevin_middle"
    eng.set_integrator(mode)
    row = {}
    for t in TARGETS:
        eng.set_temperature(t)
        ts = []
        for i in range(N_SETTLE):
            r = eng.step_md()
            ts.append(r["t_kin"])
        eq = sum(ts[-N_AVG:]) / N_AVG
        row[t] = eq
        print(f"  gamma={g:4.1f} target={t:5.0f} -> equil t_kin={eq:6.1f} K  "
              f"(offset {eq - t:+6.1f}, ratio {eq / t:.3f})", flush=True)
    results[g] = row
    print("", flush=True)

print("===== OFFSET SURFACE =====")
print(f"{'target':>8} | " + " | ".join(f"g={g:<5}" for g in GAMMAS))
for t in TARGETS:
    print(f"{t:8.0f} | " + " | ".join(f"{results[g][t]:6.1f}" for g in GAMMAS))
print("(offset = equil - target)")
for t in TARGETS:
    print(f"  offset@{t:4.0f}: " + " ".join(f"g={g}:{results[g][t]-t:+5.1f}" for g in GAMMAS))
