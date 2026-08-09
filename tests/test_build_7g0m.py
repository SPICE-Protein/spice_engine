#!/usr/bin/env python
"""Reproduce 7G0M N-term H->HB2 fallback (mmCIF + parquet paths)."""
import spice_engine as se

for label, struct in [
    ("mmCIF", se.Structure.from_mmcif("data/test/7G0M.cif")),
]:
    try:
        eng = se.Engine.build(struct, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
        print(f"{label}: BUILD_OK + 5 steps", flush=True)
        for _ in range(5):
            eng.step_md()
        print(f"{label}: u={eng.u_t_kcal():.0f}", flush=True)
    except Exception as e:
        print(f"{label}: BUILD_FAILED: {e}", flush=True)
