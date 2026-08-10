#!/usr/bin/env python
"""Quick verify after the water-reset fix: with the settle truly cold-starting
(all water velocities zeroed), does production t_kin now sit at the target?
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 100
EVERY = 20

struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 1.0, 0.0, relax_iters=2000, tolerance=2.0)

ti = eng.thermo_info()
print(f"right after build: ke={ti['kinetic_energy_kcal']:.0f} kcal/mol, "
      f"t_implied={ti['t_implied_k']:.0f} K (target {TEMP})", flush=True)

eng.equilibrate()

ti = eng.thermo_info()
print(f"after settle: ke={ti['kinetic_energy_kcal']:.0f}, t_implied={ti['t_implied_k']:.0f} K",
      flush=True)

ts = []
for i in range(N_STEPS):
    r = eng.step_md()
    ts.append(r["t_kin"])
    if i % EVERY == 0 or i == N_STEPS - 1:
        print(f"  step {i:3d}: t_kin={r['t_kin']:6.1f} K  n_clamped={r['n_clamped']}  "
              f"u={r['u_t_kcal']:.0f}", flush=True)

t_last = ts[-1]
print(f"  -> last t_kin={t_last:.1f} K (target {TEMP}) | "
      f"{'THERMOSTAT_OK' if abs(t_last - TEMP) < 20.0 else 'MISMATCH'}")
