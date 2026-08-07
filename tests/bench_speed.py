#!/usr/bin/env python
"""Benchmark spice_engine MD throughput (steps/s) on 2LYZ.

Also probes whether a longer window produces T-dependent compactness/flexibility
metrics (m2 Rg-drift, m3 SS-loss), i.e. whether the scan CAN see thermal
destabilization given enough steps. Run in the background; tail the output.
"""
import time
import spice_engine as se


def print_profile(eng, label):
    """Print the engine's per-category timing (µs/step), sampled every 20 steps."""
    ct = eng.computation_time()
    steps = max(ct.pop("steps"), 1)
    total = ct.pop("total_us")
    line = "  ".join(f"{k.replace('_us',''):>16}: {v/steps:>9.1f}µs/step" for k, v in ct.items())
    print(f"[profile {label}] total {total/steps:>8.1f}µs/step | {line}", flush=True)


s = se.Structure.from_mmcif("data/test/2LYZ.cif")
print(f"building 2LYZ ({s.residue_count()} res) pH=7 T=310 ...", flush=True)
eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
print("built ok", flush=True)

WARM = 2000       # discard build strain
N = 20_000        # timed steps
STEP = 5000       # metrics checkpoint interval

t0 = time.perf_counter()
for i in range(WARM):
    eng.step_md()
t_warm = time.perf_counter() - t0
print(f"warm {WARM} steps: {t_warm:.1f}s ({WARM/t_warm:.0f} steps/s)", flush=True)
print_profile(eng, "after-warm")

t0 = time.perf_counter()
for i in range(1, N + 1):
    eng.step_md()
    if i % STEP == 0:
        m = eng.metrics()
        el = time.perf_counter() - t0
        rate = i / el
        print(f"  {i:>6} steps | {el:>6.1f}s | {rate:>7.0f} steps/s | "
              f"t={eng.time_ps():>8.3f}ps | rg={m['rg']:.3f} | m2={m['m2']:.4f} "
              f"m3={m['m3']:.3f} m1={m['m1']:.0f}", flush=True)
t_total = time.perf_counter() - t0
print(f"\ntimed {N} steps: {t_total:.1f}s total = {N/t_total:.0f} steps/s "
      f"= {N/t_total*0.002/1000:.3f} ns/s", flush=True)
print_profile(eng, "after-timed")

# Rough extrapolation table for the user
print("\nextrapolated wall-time (1 core):")
for ns in (0.05, 0.1, 0.5, 1.0, 5.0, 10.0):
    steps = ns * 1e3 / 0.002
    sec = steps / (N / t_total)
    print(f"  {ns:>5.1f} ns = {steps/1e3:>7.0f}k steps ~ {sec/60:>6.1f} min/point")
