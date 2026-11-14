#!/usr/bin/env python3
"""Which form of the experts' GEMMs runs as fast as the dense layer's (MoE operators, owner 2026-11-13). Forward + backward of the expert
part (two GEMMs and SwiGLU, bf16) for n experts of cap tokens each, against the dense SwiGLU on n * cap / G tokens (the same FLOPs),
eager (cuBLAS picks the kernels; torch.compile's default mode keeps the same aten calls), CUDA events, median of 20:
  bmm_T     torch.bmm(xe, W12.view(n, 2h, d).mT): the layer's form (weights [n * 2h, d], rows = outputs)
  bmm_N     torch.bmm(xe, W12n) with the weights stored [n, d, 2h] (no transposed view)
  loop_lin  F.linear per expert on its rows (cuBLAS's plain GEMM, as the dense layer)
  loop_mm   torch.mm per expert with [d, 2h] weights
Gradients reach the inputs and the weights; .grad is cleared before every call (as an optimizer step does).
usage: bench_expert_gemm.py --d 1024 --tokens 32768 --experts 16 --active 2"""
import argparse, statistics
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=1024); p.add_argument("--tokens", type=int, default=32768)
p.add_argument("--experts", type=int, default=16); p.add_argument("--active", type=int, default=2)
a = p.parse_args()
dev, bf = "cuda", torch.bfloat16
d, H, n, G = a.d, 4 * a.d, a.experts, a.active
h, M = H // G, a.tokens
cap = G * M // n
torch.manual_seed(0)


def timed(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)


flops = 3 * 6 * d * H * M
x = torch.randn(M, d, device=dev, dtype=bf, requires_grad=True)
w12 = (torch.randn(2 * H, d, device=dev, dtype=bf) * 0.02).requires_grad_(True); w3 = (torch.randn(d, H, device=dev, dtype=bf) * 0.02).requires_grad_(True)
gy = torch.randn(M, d, device=dev, dtype=bf)
def dense():
    for t in (x, w12, w3): t.grad = None
    a1, a2 = F.linear(x, w12).chunk(2, dim=-1)
    F.linear(F.silu(a1) * a2, w3).backward(gy)
res = {"dense": timed(dense)}
xe = torch.randn(n, cap, d, device=dev, dtype=bf, requires_grad=True)
ge = torch.randn(n, cap, d, device=dev, dtype=bf)
E12 = (torch.randn(n * 2 * h, d, device=dev, dtype=bf) * 0.02).requires_grad_(True)          # the layer's storage
E3t = (torch.randn(n * h, d, device=dev, dtype=bf) * 0.02).requires_grad_(True)
E12n = (torch.randn(n, d, 2 * h, device=dev, dtype=bf) * 0.02).requires_grad_(True)          # [n, d, 2h]: no transposed view
E3n = (torch.randn(n, h, d, device=dev, dtype=bf) * 0.02).requires_grad_(True)
def clear(*ts):
    for t in ts: t.grad = None
def bmm_T():
    clear(xe, E12, E3t)
    a1, a2 = torch.bmm(xe, E12.view(n, 2 * h, d).mT).chunk(2, dim=-1)
    torch.bmm(F.silu(a1) * a2, E3t.view(n, h, d)).backward(ge)
def bmm_N():
    clear(xe, E12n, E3n)
    a1, a2 = torch.bmm(xe, E12n).chunk(2, dim=-1)
    torch.bmm(F.silu(a1) * a2, E3n).backward(ge)
W12l = E12.view(n, 2 * h, d); W3l = E3t.view(n, h, d)
def loop_lin():
    clear(xe, E12, E3t)
    outs = []
    for e in range(n):
        a1, a2 = F.linear(xe[e], W12l[e]).chunk(2, dim=-1)
        outs.append(F.linear(F.silu(a1) * a2, W3l[e].t()))
    torch.stack(outs).backward(ge)
def loop_mm():
    clear(xe, E12n, E3n)
    outs = []
    for e in range(n):
        a1, a2 = torch.mm(xe[e], E12n[e]).chunk(2, dim=-1)
        outs.append(torch.mm(F.silu(a1) * a2, E3n[e]))
    torch.stack(outs).backward(ge)
for name, fn in (("bmm_T", bmm_T), ("bmm_N", bmm_N), ("loop_lin", loop_lin), ("loop_mm", loop_mm)):
    res[name] = timed(fn)
print(f"{torch.cuda.get_device_name()}: d {d}, tokens {M}, experts {n}, active {G} (hidden {h}, {cap} rows per expert); forward + backward ms "
      f"(TFLOPS at the dense layer's FLOPs):", flush=True)
for k, v in res.items():
    print(f"  {k:9s} {v:7.2f} ms  {flops / v / 1e9:6.0f} TFLOPS  x{v / res['dense']:.3f} of dense", flush=True)
