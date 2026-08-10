#!/usr/bin/env python
"""Verify the thermostat actually reaches the target temperature, and that
the clamp metrics are observable.

This is the CRITICAL diagnostic from the instability analysis:
  T=380K -> bigger thermal kicks -> force spikes -> MAX_ACCEL clamp swallows
  the strain. We need to confirm t_kin really tracks the target (310 / 380 K),
  otherwise the Langevin coupling (gamma=0.5) is too weak and t_kin ~ 298 K is
  a more fundamental bug than the clamps.

Prints every `every` steps: step, t_kin, n_clamped, max_accel_clamped, u.
Ends with a verdict on whether the thermostat converged.
"""
import spice_engine as se

N_STEPS = 300
EVERY = 25
TARGETS = [310.0, 380.0]


def run(label, temp):
    print(f"\n===== 2LYZ @ {temp:.0f} K =====", flush=True)
    struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
    eng = se.Engine.build(struct, 7.0, temp, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
    eng.equilibrate()  # restraint-free NVT settle, restores target T + integrator

    t_k = []
    clamps = 0
    max_clamp = 0.0
    crashed = False
    for i in range(N_STEPS):
        r = eng.step_md()
        t_k.append(r["t_kin"])
        clamps += r["n_clamped"]
        max_clamp = max(max_clamp, r["max_accel_clamped"])
        if r["crashed"]:
            crashed = True
            print(f"  step {i}: CRASHED u={r['u_t_kcal']:.0f}", flush=True)
            break
        if i % EVERY == 0 or i == N_STEPS - 1:
            print(
                f"  step {i:4d}: t_kin={r['t_kin']:6.1f} K  "
                f"n_clamped={r['n_clamped']:3d}  max_accel={r['max_accel_clamped']:9.0f}  "
                f"u={r['u_t_kcal']:.0f}",
                flush=True,
            )

    t_avg = sum(t_k) / len(t_k) if t_k else 0.0
    t_last = t_k[-1] if t_k else 0.0
    verdict = "THERMOSTAT_OK" if abs(t_last - temp) < 15.0 else "THERMOSTAT_MISMATCH"
    print(
        f"  -> t_kin avg={t_avg:6.1f} K, last={t_last:6.1f} K (target {temp:.0f} K) | "
        f"{verdict} | total clamps={clamps}, max_accel={max_clamp:.0f} | crashed={crashed}",
        flush=True,
    )
    return verdict, t_avg, t_last, clamps, max_clamp, crashed


results = {}
for t in TARGETS:
    results[t] = run(f"2LYZ@{t:.0f}", t)

print("\n===== SUMMARY =====")
all_ok = True
for t, (verdict, t_avg, t_last, clamps, max_clamp, crashed) in results.items():
    ok = verdict == "THERMOSTAT_OK" and not crashed
    all_ok &= ok
    print(
        f"  {t:.0f}K: {verdict} t_avg={t_avg:.1f} t_last={t_last:.1f} "
        f"clamps={clamps} max_accel={max_clamp:.0f} crashed={crashed}",
        flush=True,
    )
print("ALL_OK" if all_ok else "CHECK_ABOVE")
