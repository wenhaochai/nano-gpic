#!/usr/bin/env python3
"""The ceiling of native FP8 for the image blocks (owner 2026-11-07: 直接做b): per GEMM of one block at width C, M = 128 x 256
tokens, bf16 torch.mm against torch._scaled_mm (e4m3 activations and weights forward, e5m2 gradients backward, tensor
scales, fast accumulation forward only), forward / dgrad / wgrad; plus one cast pass (bf16 -> fp8 with scale, clamp and
amax, compiled) per GEMM input. CUDA-event timing, median of 50 after 10 warm-ups.
usage: python fp8_gemm_bench.py <C> [M]"""
import sys, statistics as st
import torch

C = int(sys.argv[1]); M = int(sys.argv[2]) if len(sys.argv) > 2 else 128 * 256
dev = "cuda"
E4, E5, BF = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16
one = torch.ones((), device=dev)


def t(fn, n=50, w=10):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)


@torch.compile
def cast(x, s):   # one fused pass: amax of the input and the scaled, clamped cast (delayed scaling: s from before)
    return (x.float() * s).clamp(-448, 448).to(E4), x.abs().amax()


shapes = {"qkv": (C, 3 * C), "proj": (C, C), "w12": (C, 8 * C), "w3": (4 * C, C)}
tot = {"bf16": 0.0, "fp8": 0.0, "cast": 0.0}
for name, (K, N) in shapes.items():
    x = torch.randn(M, K, device=dev, dtype=BF); w = torch.randn(N, K, device=dev, dtype=BF) * 0.02
    dy = torch.randn(M, N, device=dev, dtype=BF)
    xq, wq, wtq = x.to(E4), w.to(E4), w.t().contiguous().to(E4)
    dyq, dytq, xtq = dy.to(E5), dy.t().contiguous().to(E5), x.t().contiguous().to(E4)
    b = [t(lambda: x @ w.t()), t(lambda: dy @ w), t(lambda: dy.t() @ x)]
    f = [t(lambda: torch._scaled_mm(xq, wq.t(), scale_a=one, scale_b=one, out_dtype=BF, use_fast_accum=True)),
         t(lambda: torch._scaled_mm(dyq, wtq.t(), scale_a=one, scale_b=one, out_dtype=BF)),
         t(lambda: torch._scaled_mm(dytq, xtq.t(), scale_a=one, scale_b=one, out_dtype=BF))]
    c = t(lambda: cast(x, one))
    tot["bf16"] += sum(b); tot["fp8"] += sum(f); tot["cast"] += c
    print(f"C={C} {name} [M={M} K={K} N={N}]: bf16 fwd/dgrad/wgrad {b[0]:.3f}/{b[1]:.3f}/{b[2]:.3f} ms, fp8 {f[0]:.3f}/{f[1]:.3f}/{f[2]:.3f} ms "
          f"(x{sum(b) / sum(f):.2f}); one cast pass of the input {c:.3f} ms", flush=True)
print(f"C={C} block total: bf16 {tot['bf16']:.2f} ms, fp8 {tot['fp8']:.2f} ms (x{tot['bf16'] / tot['fp8']:.2f}), "
      f"one cast pass per GEMM input {tot['cast']:.2f} ms", flush=True)
