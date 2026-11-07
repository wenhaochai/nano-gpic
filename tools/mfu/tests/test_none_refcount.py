#!/usr/bin/env python3
"""None's refcount under Inductor's C++ wrapper (src/utils/patch_bugs.py, the Py_None fix): a compiled function whose
backward returns None for an input without a gradient, 1000 forward + backward calls, sys.getrefcount(None) before and
after, in one fresh process without the fix and one with it. torch's compiled path takes about 4 references to None per
call that it never releases, with either launcher (eager: 0; no gc-tracked object grows, so it is a count, not memory),
so the check is the difference: without the fix the count falls one short per call and per None output (here one), the
leak that crashed w1204 / w1239 once the None outputs per call outnumbered those 4; with the fix that difference is gone.
usage: python test_none_refcount.py   (CPU is enough: the wrapper template is the same)"""
import os, subprocess, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
CHILD = r'''
import os, sys
sys.path.insert(0, sys.argv[2])
import torch
if sys.argv[1] == "1":
    import src.utils.patch_bugs  # noqa: F401  (the fix)
import torch._inductor.config as cfg
cfg.cpp_wrapper = True
def f(x, y):
    return (x * y).sin().sum()
cf = torch.compile(f)
x, y = torch.randn(64, requires_grad=True), torch.randn(64)
for _ in range(5):
    cf(x, y).backward()
r0 = sys.getrefcount(None)
for _ in range(1000):
    cf(x, y).backward()
print(sys.getrefcount(None) - r0)
'''
out = {}
for fix in ("0", "1"):
    r = subprocess.run([sys.executable, "-c", CHILD, fix, SRC], capture_output=True, text=True,
                       env={**os.environ, "TORCHINDUCTOR_CPP_WRAPPER": "1", "CUDA_VISIBLE_DEVICES": ""})
    lines = [l for l in r.stdout.strip().splitlines() if l.lstrip("-").isdigit()]
    out[fix] = int(lines[-1]) if lines else None
    if out[fix] is None:
        print(r.stdout[-800:], r.stderr[-1500:])
print(f"None refcount change over 1000 compiled forward + backward calls: without the fix {out['0']}, with it {out['1']}")
ok = out["0"] is not None and out["1"] is not None and 900 <= out["1"] - out["0"] <= 1100
print(f"the fix keeps {out['1'] - out['0'] if ok or (out['0'] is not None and out['1'] is not None) else '?'} references to None that the "
      "unpatched wrapper lost over 1000 calls (one None output per call)")
print("NONE_REFCOUNT_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
