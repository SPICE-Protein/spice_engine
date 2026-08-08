#!/usr/bin/env python
"""Validate the incomplete-residue policy:
- 1R2I strict (default) must fail with a CLEAR aggregate error (was the
  misleading "Missing bond params for CX-HB2").
- 1R2I strict_incomplete=False must build truncated.
- 5H3G / 4LPX / 8CWC must still build OK in strict mode.
"""
import spice_engine as se


def build_ok(name, strict=True):
    try:
        s = se.Structure.from_mmcif(f"data/test/{name}.cif")
        eng = se.Engine.build(
            s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0,
            strict_incomplete=strict,
        )
        for _ in range(5):
            eng.step_md()
        print(f"{name} strict={strict}: BUILD_OK + 5 steps, u={eng.u_t_kcal():.0f}", flush=True)
        return True
    except Exception as e:
        print(f"{name} strict={strict}: BUILD_FAILED: {e}", flush=True)
        return False


print("== 1R2I strict (expect clear Incomplete structure error) ==", flush=True)
try:
    s = se.Structure.from_mmcif("data/test/1R2I.cif")
    se.Engine.build(s, 7.0, 310.0, 1.0, 0.0, relax_iters=2000, tolerance=2.0)
    print("1R2I strict: UNEXPECTEDLY BUILT", flush=True)
except Exception as e:
    msg = str(e)
    print("1R2I strict FAILED as expected.", flush=True)
    print("  has_clear_msg:", "Incomplete structure" in msg, flush=True)
    print("  message:", msg[:400], flush=True)

print("\n== 1R2I lenient (expect build OK) ==", flush=True)
build_ok("1R2I", strict=False)

print("\n== regression: 5H3G / 4LPX / 8CWC strict ==", flush=True)
for n in ("5H3G", "4LPX", "8CWC"):
    build_ok(n, strict=True)
