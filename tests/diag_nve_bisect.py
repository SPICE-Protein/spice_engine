#!/usr/bin/env python
"""NVE bisect: which force class is responsible for the ~7 kcal/mol/step energy
leak? Disable one class at a time (after settle) and measure the NVE total
energy drift dE/step over steps [20, 140].

Modes (each arg disables the class):
  full        : baseline (expect ~7)
  no_spme     : long_range_recip_disabled (SPME reciprocal-space cache?)
  no_lj       : lj_disabled
  no_coulomb  : coulomb_disabled (real + recip)
  no_bonded   : bonded_disabled (bonds/angles/dihedrals + H types)
"""
import spice_engine as se

TEMP = 310.0
N_STEPS = 140
SKIP = 20  # ignore the first SKIP steps (initial transient)


def leak_rate(overrides, label):
    print(f"\n===== {label} =====", flush=True)
    struct = se.Structure.from_mmcif("data/test/2LYZ.cif")
    eng = se.Engine.build(struct, 7.0, TEMP, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
    eng.equilibrate()
    eng.set_integrator("nve")
    eng.set_force_overrides(*overrides)

    e0 = None
    es = []
    for i in range(N_STEPS):
        r = eng.step_md()
        etot = r["u_t_kcal"] + eng.kinetic_energy_kcal()
        es.append(etot)
        if e0 is None:
            e0 = etot
        if i % 25 == 0:
            print(f"  step {i:4d}: t_kin={r['t_kin']:6.1f} K  E={etot:.0f}  dE={etot - e0:+.1f}", flush=True)

    seg = es[SKIP:]
    rate = (seg[-1] - seg[0]) / max(len(seg) - 1, 1)
    print(f"  -> dE/step (steps {SKIP}-{N_STEPS}) = {rate:+.3f} kcal/mol/step")
    return rate


MODES = [
    ((False, False, False, False), "full (baseline)"),
    ((False, False, False, True),  "no_spme (long-range recip OFF)"),
    ((False, False, True,  False), "no_lj"),
    ((False, True,  False, False), "no_coulomb"),
    ((True,  False, False, False), "no_bonded"),
]

rates = {}
for overrides, label in MODES:
    rates[label] = leak_rate(overrides, label)

print("\n===== BISECT RESULT =====")
for label, rate in rates.items():
    print(f"  {label:24s}: {rate:+.3f} kcal/mol/step")
