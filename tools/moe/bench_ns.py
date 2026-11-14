#!/usr/bin/env python3
"""Newton-Schulz speed for the MoE line's expert matrices (MoE operators, owner 2026-11-13: 「Infra上也要努力的 moe算子」). On 8 H100 the
experts' Newton-Schulz costs 9.4 ms per step at d12 with 16 experts of 2 active (w2682: 7.15 ms per layer on one GPU, at 362 TFLOPS), a
quarter of the MoE's extra step time under FSDP2. Each rank orthogonalizes its own units batched per shape (src/optim/muon.py, msign with
GPIC_MFU_NS_FUSED=1). This times one rank's batch, forward only, bf16, 5 steps, CUDA events (median of 20 after 5 warm-up calls), in these
forms (all compute the same polynomial; the compiled ones may round differently):
  ref        msign(G, fused=True): the training code
  contig     the same with the transposed input made contiguous first (a tall unit is orthogonalized as its transpose)
  compiled   torch.compile(msign) in the default mode
  autotune   torch.compile(msign, mode="max-autotune-no-cudagraphs"): Inductor times Triton bmm templates against cuBLAS per GEMM
and the largest difference of each form from ref (relative to ref's largest entry) and from an fp32 msign.
Batches (one rank's share on 8 ranks): d12 G2: 72 units [2048, 1024]; d11 G2: 66 units [1792, 896]; d12 G8: 288 units [512, 1024];
d11 G8: 264 units [448, 896]; d12 dense FFN: 4.5 units [4096, 1024] (5 here).
usage: bench_ns.py [--only d12g2,...]"""
import argparse, os, statistics, sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))
from src.optim.muon import msign   # noqa: E402

p = argparse.ArgumentParser(); p.add_argument("--only", default=""); a = p.parse_args()
dev = "cuda"
BATCHES = {"d12g2": (72, 2048, 1024), "d11g2": (66, 1792, 896), "d12g8": (288, 512, 1024), "d11g8": (264, 448, 896), "d12dense": (5, 4096, 1024)}
if a.only:
    BATCHES = {k: v for k, v in BATCHES.items() if k in a.only.split(",")}


def timed(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def ns_flops(b, r, c, steps=5):   # per step: A = X X^T, A A, B X on the wide form [m, n], m = min(r, c), n = max(r, c)
    m, n = min(r, c), max(r, c)
    return b * steps * (2 * m * m * n + 2 * m * m * m + 2 * m * m * n)


def contig(G, steps=5):
    X = G.mT.contiguous() if G.size(-2) > G.size(-1) else G
    return msign(X, steps=steps, dtype=torch.bfloat16, fused=True).mT if G.size(-2) > G.size(-1) else msign(X, steps=steps, dtype=torch.bfloat16, fused=True)


ref_fn = lambda G: msign(G, steps=5, dtype=torch.bfloat16, fused=True)
comp = torch.compile(lambda G: msign(G, steps=5, dtype=torch.bfloat16, fused=True))
auto = torch.compile(lambda G: msign(G, steps=5, dtype=torch.bfloat16, fused=True), mode="max-autotune-no-cudagraphs")
print(torch.cuda.get_device_name(), flush=True)
for name, (b, r, c) in BATCHES.items():
    torch.manual_seed(0)
    G = torch.randn(b, r, c, device=dev, dtype=torch.bfloat16)
    ref = ref_fn(G).float()
    f32 = msign(G.float(), steps=5, dtype=torch.float32, fused=True)
    big = float(ref.abs().max())
    out = {}
    for k, fn in (("ref", ref_fn), ("contig", contig), ("compiled", comp), ("autotune", auto)):
        ms = timed(lambda: fn(G))
        o = fn(G).float()
        out[k] = (ms, ns_flops(b, r, c) / ms / 1e9, float((o - ref).abs().max()) / big, float((o - f32).abs().max()) / big)
    print(f"{name}: {b} x [{r}, {c}]", flush=True)
    for k, (ms, tf, dr, df) in out.items():
        print(f"  {k:9s} {ms:7.2f} ms  {tf:5.0f} TFLOPS  x{ms / out['ref'][0]:.3f}  max diff vs ref {dr:.1e}, vs fp32 {df:.1e}", flush=True)
