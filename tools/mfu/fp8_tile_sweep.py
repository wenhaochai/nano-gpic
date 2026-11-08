#!/usr/bin/env python3
"""FP8 phase 4's transposing casts ran 3-4x slower than bandwidth (w1959: d1536, M 32768: dcast_t 0.39 ms against 0.13 for
the row-major cast; swiglu_dcast_t 1.41 against 0.48; swiglu_bwd_dt 2.50 against 0.96). This times, at one width and
micro-batch, (1) each fused transposing kernel over tile shapes and warp counts (src/models/layers/fp8_native.TILES), and
(2) the two-pass alternative: phase 3's row-major cast, then an FP8 transpose (torch's copy of the uint8 view). Outputs of
every configuration are checked bit for bit against the 64 x 64 x 4 default. CUDA events, median of 30.
usage: python fp8_tile_sweep.py <width> [<M>]"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))
import src.models.layers.fp8_native as F  # noqa: E402

C = int(sys.argv[1])
M = int(sys.argv[2]) if len(sys.argv) > 2 else 128 * 256
H = 4 * C
dev, BF = "cuda", torch.bfloat16
torch.manual_seed(0)
F.DS.sites, F.DS.st = 8, None
st = F.ds_state(torch.device(dev))
x = torch.randn(M, C, device=dev).to(BF)
z = torch.randn(M, 2 * H, device=dev).to(BF)
da = torch.randn(M, H, device=dev).to(BF)
for site in range(8):
    F.dcast(x, st, site, False)
F.dcast(x, st, 0, True)   # every site has a history: the steady state


def t(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def bits(o):
    return [u.view(torch.uint8).clone() if u.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else u.clone() for u in o]


ops = {"cast": lambda: F.dcast_t(x, st, 1, False, False), "swiglu": lambda: F.swiglu_dcast_t(z, st, 2),
       "bwd": lambda: F.swiglu_bwd_dt(z, da, st, 3)}
print(f"d{C}, M {M}: ms per call")
best = {}
for k, fn in ops.items():
    F.TILES[k] = (64, 64, 4)
    ref = bits(fn())
    res = []
    for bm in (32, 64, 128):
        for bk in (32, 64, 128, 256):
            for nw in (4, 8):
                if bm * bk > 128 * 128:
                    continue
                F.TILES[k] = (bm, bk, nw)
                try:
                    same = all(torch.equal(a, b) for a, b in zip(bits(fn()), ref))
                    res.append((t(fn), bm, bk, nw, same))
                except Exception as e:   # a tile too large for shared memory
                    res.append((float("inf"), bm, bk, nw, type(e).__name__))
    res.sort()
    best[k] = res[0]
    print(f"  {k}: " + ", ".join(f"{bm}x{bk}x{nw} {ms:.3f}{'' if same is True else ' (' + str(same) + ')'}" for ms, bm, bk, nw, same in res[:6]))
    F.TILES[k] = res[0][1:4]
# two passes: the row-major cast of phase 3, then a transpose of the FP8 copy
two = {"cast": lambda: (lambda q: (q, q.view(torch.uint8).t().contiguous()))(F.dcast(x, st, 1, False)[0]),
       "swiglu": lambda: (lambda q: (q, q.view(torch.uint8).t().contiguous()))(F.swiglu_dcast(z, st, 2)[0]),
       "bwd": lambda: (lambda q: (q, q.view(torch.uint8).t().contiguous()))(F.swiglu_bwd_ds(z, da, st, 3)[1])}
for k, fn in two.items():
    print(f"  {k}: two passes (row cast + uint8 transpose) {t(fn):.3f}; best fused {best[k][0]:.3f} at {best[k][1]}x{best[k][2]}x{best[k][3]}")
print("FP8_TILE_SWEEP DONE")
