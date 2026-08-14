#!/usr/bin/env python
"""Reproduce the 'set=XK -> t_kin=YK' scan to check whether it is the
UN-EQUILIBRATED hot start + strain release dominating t_kin (vs the set temp).

Version A: build once, set_temperature(T), step a few, read t_kin (no equil).
Version B: build at T, equilibrate(), then step (proper).
"""
import spice_engine as se

TARGETS = [200.0, 250.0, 298.0, 380.0]


def build_eng(temp):
    struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
    return se.Engine.build(struct, 7.0, temp, 0.0, 0.0, relax_iters=2000, tolerance=2.0)


print("=== A: no equilibrate, set_temperature then 1 step ===", flush=True)
eng = build_eng(310.0)
for T in TARGETS:
    eng.set_temperature(T)
    r = eng.step_md()
    print(f"  set={T:.0f}K -> t_kin={r['t_kin']:.0f}K  u={r['u_t_kcal']:.0f}", flush=True)

print("=== A2: no equilibrate, set_temperature then 50 steps ===", flush=True)
eng = build_eng(310.0)
for T in TARGETS:
    eng.set_temperature(T)
    last = None
    for i in range(50):
        r = eng.step_md()
        last = r["t_kin"]
    print(f"  set={T:.0f}K -> t_kin(step50)={last:.0f}K  u={r['u_t_kcal']:.0f}", flush=True)

print("=== B: build at T + equilibrate, then 50 steps ===", flush=True)
for T in TARGETS:
    eng = build_eng(T)
    eng.equilibrate()
    last = None
    for i in range(50):
        r = eng.step_md()
        last = r["t_kin"]
    print(f"  target={T:.0f}K -> t_kin(step50)={last:.0f}K  u={r['u_t_kcal']:.0f}", flush=True)
