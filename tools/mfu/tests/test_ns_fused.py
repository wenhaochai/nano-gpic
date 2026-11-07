#!/usr/bin/env python3
"""MFU lever NS (GPIC_MFU_NS_FUSED=1, src/optim/muon.py msign): Newton-Schulz with the scalings and sums inside addmm /
baddbmm. In fp32 it equals the reference's polynomial to fp32 rounding; in bf16 its error against fp32 Newton-Schulz
must not exceed the reference's (it rounds once per GEMM instead of after every product and sum). Shapes: stacked units
(baddbmm), a tall matrix (transposed first), a square one. CPU or one GPU.
usage: python test_ns_fused.py"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.optim.muon import msign  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
big = dev == "cuda"
shapes = [(4, 1536, 6144), (12288, 1536), (1536, 1536)] if big else [(3, 192, 768), (768, 192), (256, 256)]


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


ok = True
for shp in shapes:
    G = torch.randn(*shp, device=dev) * torch.rand(*shp[:-1], 1, device=dev)   # uneven row scales, as momenta have
    ref = msign(G, dtype=torch.float32, fused=False)
    f32 = msign(G, dtype=torch.float32, fused=True)
    bf = msign(G, dtype=torch.bfloat16, fused=False)
    fu = msign(G, dtype=torch.bfloat16, fused=True)
    e32, ebf, efu = rel(f32, ref), rel(bf, ref), rel(fu, ref)
    good = e32 < 1e-4 and efu <= 1.05 * ebf
    ok &= good
    print(f"{dev} {tuple(shp)}: fused fp32 vs reference fp32 {e32:.1e}; bf16 vs fp32 Newton-Schulz: reference {ebf:.2e}, "
          f"fused {efu:.2e}" + ("" if good else "  FAIL"), flush=True)
print("NS_FUSED_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
