#!/usr/bin/env python
"""Control experiment: build 2LYZ WITHOUT equilibrate and run production
directly. Separates two hypotheses for the t_kin drift seen after
equilibrate():

  (a) if step-0 t_kin ~ 310 K (not ~378) and stays flat -> the settle leaves
      the system hot; production thermostat is fine.
  (b) if step-0 t_kin ~ 310 K then climbs -> production LangevinMiddle
      (gamma=0.5) net-injects energy (thermostat/integrator bug).
  (c) if step-0 t_kin already ~ 378 K -> measurement offset (e.g. thermo_dof
      cached wrong) OR build's velocity init is off.
"""
import spice_engine as se

N_STEPS = 300
EVERY = 25
TEMP = 310.0

print(f"===== 2LYZ @ {TEMP:.0f} K, NO EQUILIBRATE (control) =====", flush=True)
struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
eng = se.Engine.build(struct, 7.0, TEMP, 1.0, 0.0, relax_iters=2000, tolerance=2.0)

t_k = []
for i in range(N_STEPS):
    r = eng.step_md()
    t_k.append(r["t_kin"])
    if r["crashed"]:
        print(f"  step {i}: CRASHED u={r['u_t_kcal']:.0f}", flush=True)
        break
    if i % EVERY == 0 or i == N_STEPS - 1:
        print(
            f"  step {i:4d}: t_kin={r['t_kin']:6.1f} K  n_clamped={r['n_clamped']:3d}  "
            f"max_accel={r['max_accel_clamped']:9.0f}  u={r['u_t_kcal']:.0f}",
            flush=True,
        )

t_avg = sum(t_k) / len(t_k) if t_k else 0.0
t_last = t_k[-1] if t_k else 0.0
print(f"  -> t_kin avg={t_avg:.1f} K, last={t_last:.1f} K (target {TEMP:.0f} K)")
verdict = "THERMOSTAT_OK" if abs(t_last - TEMP) < 15.0 else "THERMOSTAT_MISMATCH"
print(f"  -> {verdict}")
