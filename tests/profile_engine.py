#!/usr/bin/env python
"""Quick engine profile: build + 600 steps + print per-category timing.

This tells us WHERE a step's ~115 ms goes (bonded / nonbonded short-range /
ewald long-range / neighbor build / integrate / ambient). Run it, read the
breakdown, then port the LAMMPS technique that targets the dominant cost.
"""
import time
import spice_engine as se

s = se.Structure.from_mmcif("data/test/2LYZ.cif")
print(f"building 2LYZ ({s.residue_count()} res) pH=7 T=310 ...", flush=True)
t0 = time.perf_counter()
eng = se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
print(f"build ok in {time.perf_counter()-t0:.1f}s", flush=True)

# The engine samples timing every 20 steps, so 600 steps = 30 samples — plenty.
# Use step_md() so the measurement is the ENGINE only, not the Python metrics
# layer (the old step() spent ~90 ms/step in O(N²) metrics compute).
N = 600
t0 = time.perf_counter()
for i in range(N):
    eng.step_md()
wall = time.perf_counter() - t0
print(f"\n{N} steps in {wall:.1f}s = {N/wall:.1f} steps/s "
      f"(wall ~ {wall/N*1000:.0f} ms/step)", flush=True)

ct = eng.computation_time()
steps = max(ct.pop("steps"), 1)
total = ct.pop("total_us")
print(f"\nengine-internal profile (sampled every 20 steps, {steps} steps done):")
print(f"  {'total (MD phases only)':>28}: {total/steps:>9.1f} µs/step")
for k, v in ct.items():
    label = k.replace("_us", "")
    print(f"  {label:>28}: {v/steps:>9.1f} µs/step"
          f"  ({100*v/total:>5.1f}%)")
print(f"\n  wall-vs-engine gap: {wall/steps*1e6 - total/steps:>9.1f} µs/step "
      "(Python FFI + per-step metrics compute — if this is large, the Python "
      "layer, not the engine, is the bottleneck)")
