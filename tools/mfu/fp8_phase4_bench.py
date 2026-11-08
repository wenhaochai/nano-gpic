#!/usr/bin/env python3
"""Why FP8 phase 4 is slower (w1952: native3 -> native4 +18.9% at d1024, +10.4% at d1536): each phase 4 piece against the
phase 3 piece it replaces, at one image block's shapes (M = 128 x 256 tokens per GPU), CUDA events, median of 30 after 5
warm-up runs, with delayed-scaling history (the steady state):
  casts: dcast vs dcast_t (x [M, C] and dy [M, C]), swiglu_dcast vs swiglu_dcast_t (z [M, 2H]), swiglu_bwd_ds vs swiglu_bwd_dt;
  GEMMs: the four weight gradients bf16 (phase 3) vs FP8 (phase 4), and the qkv_x / proj dgrads bf16 vs FP8.
usage: python fp8_phase4_bench.py <width> [<M>]"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))
import src.models.layers.fp8_native as F  # noqa: E402

C = int(sys.argv[1])
M = int(sys.argv[2]) if len(sys.argv) > 2 else 128 * 256
H = 4 * C
dev, BF = "cuda", torch.bfloat16
torch.manual_seed(0)
F.DS.sites, F.DS.st = 16, None
st = F.ds_state(torch.device(dev))


def t(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


x = torch.randn(M, C, device=dev).to(BF)
z = torch.randn(M, 2 * H, device=dev).to(BF)
da = torch.randn(M, H, device=dev).to(BF)
# a history for every site (the steady state)
for site in range(16):
    F.dcast(x, st, site, False)
F.dcast(x, st, 0, True)
rows = []
rows.append(("cast x [M,C]", t(lambda: F.dcast(x, st, 1, False)), t(lambda: F.dcast_t(x, st, 1, False, False))))
rows.append(("cast dy [M,C]", t(lambda: F.dcast(x, st, 2, False)), t(lambda: F.dcast_t(x, st, 2, False, True))))
rows.append(("swiglu cast z [M,2H]", t(lambda: F.swiglu_dcast(z, st, 3)), t(lambda: F.swiglu_dcast_t(z, st, 3))))
rows.append(("swiglu backward", t(lambda: F.swiglu_bwd_ds(z, da, st, 4)), t(lambda: F.swiglu_bwd_dt(z, da, st, 4))))
xq, xqt, sx = F.dcast_t(x, st, 1, False, False)
aq, aqt, sa = F.swiglu_dcast_t(z, st, 3)
dyq, dyqt, sdy = F.dcast_t(x, st, 2, False, True)
dzq, dzqt, sdz = F.swiglu_bwd_dt(z, da, st, 4)
dz = torch.randn(M, 2 * H, device=dev).to(BF)
a_bf = torch.randn(M, H, device=dev).to(BF)
dy3 = torch.randn(M, 3 * C, device=dev).to(BF)
dy3q, dy3qt, s3 = F.dcast_t(dy3, st, 5, False, True)
for name, N, gb, ga, gq, gqt, sq, xin, xint, sxx in [
        ("wgrad qkv_x [3C,C]", 3 * C, dy3, x, dy3q, dy3qt, s3, None, xqt, sx),
        ("wgrad proj [C,C]", C, x, x, dyq, dyqt, sdy, None, xqt, sx),
        ("wgrad w12 [2H,C]", 2 * H, dz, x, dzq, dzqt, sdz, None, xqt, sx),
        ("wgrad w3 [C,H]", C, x, a_bf, dyq, dyqt, sdy, None, aqt, sa)]:
    rows.append((name, t(lambda: gb.t() @ ga), t(lambda: F._mmo(gqt, sq, xint, sxx, torch.bfloat16))))
w3 = torch.randn(3 * C, C, device=dev).to(BF)
wq, sw, wtq = F.tcast_t(w3)
rows.append(("dgrad qkv_x [M,C]", t(lambda: dy3 @ w3), t(lambda: F._mmo(dy3q, s3, wtq, sw, torch.bfloat16))))
w1 = torch.randn(C, C, device=dev).to(BF)
w1q, sw1, w1tq = F.tcast_t(w1)
rows.append(("dgrad proj [M,C]", t(lambda: x @ w1), t(lambda: F._mmo(dyq, sdy, w1tq, sw1, torch.bfloat16))))
print(f"d{C}, M {M}: ms per call, phase 3 piece / phase 4 piece")
tot3 = tot4 = 0.0
for name, a, b in rows:
    tot3 += a; tot4 += b
    print(f"  {name:24s} {a:7.3f} {b:7.3f}  {b - a:+7.3f}")
print(f"  {'sum (one block)':24s} {tot3:7.3f} {tot4:7.3f}  {tot4 - tot3:+7.3f}")
print("FP8_PHASE4_BENCH DONE")
