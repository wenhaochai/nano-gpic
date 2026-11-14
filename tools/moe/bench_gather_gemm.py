#!/usr/bin/env python3
"""A grouped GEMM that reads its rows through the slot table, against the production path, on one GPU (2026-11-14; docs/moe.md
"Granularity: the shapes' cost"). The production path gathers the rows first (moe_ops' Triton row gather, v3) and multiplies with
cuBLAS (torch.bmm); a fused kernel reads x's rows by index inside the GEMM and never writes the gathered copy.
  gm  C[e, r, :] = X[idx[e, r], :] @ W[e]^T     (the experts' first GEMM; also the second GEMM's input gradient under v4)
      against: xe = gather(X, idx); bmm(xe, W.mT)
  gk  C[e] = P[e]^T @ X[idx[e], :]               (the weight gradients)
      against: bmm(P.mT, xe), xe already gathered (the production path keeps it from the forward)
Per shape: correctness against float32 references, the median time of 20 calls, TFLOPS, the best Triton config.
usage: bench_gather_gemm.py --shapes d12:16:2,d11:128:16 [--tokens 32768] [--out file.jsonl]"""
import argparse, json, os, statistics, sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))
from src.models.layers.moe_ops import _gather_rows   # noqa: E402

_CFG = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
        for bm, bn, bk, w, s in ((128, 128, 64, 8, 3), (128, 256, 64, 8, 3), (128, 128, 64, 4, 4), (64, 128, 64, 4, 4),
                                 (128, 64, 64, 4, 4), (256, 128, 64, 8, 3), (64, 256, 64, 4, 3), (128, 128, 32, 4, 5))]


@triton.autotune(configs=_CFG, key=["CAP", "N", "K"])
@triton.jit
def _gm_kernel(a_ptr, idx_ptr, b_ptr, c_ptr, CAP, N, K, s_a, s_be, s_bn, s_ce, s_cr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid0 = tl.program_id(0)
    mblocks = tl.cdiv(CAP, BM)
    e = pid0 // mblocks
    r = (pid0 % mblocks) * BM + tl.arange(0, BM)
    rmask = r < CAP
    tok = tl.load(idx_ptr + e * CAP + r, mask=rmask, other=0).to(tl.int64)
    nn = tl.program_id(1) * BN + tl.arange(0, BN)
    nmask = nn < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        kmask = k < K
        a = tl.load(a_ptr + tok[:, None] * s_a + k[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        b = tl.load(b_ptr + e.to(tl.int64) * s_be + nn[None, :] * s_bn + k[:, None], mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + e.to(tl.int64) * s_ce + r[:, None].to(tl.int64) * s_cr + nn[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=rmask[:, None] & nmask[None, :])


@triton.autotune(configs=_CFG, key=["CAP", "N1", "K"])
@triton.jit
def _gk_kernel(p_ptr, a_ptr, idx_ptr, c_ptr, CAP, N1, K, s_pe, s_pr, s_a, s_ce, s_ci, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid0 = tl.program_id(0)
    iblocks = tl.cdiv(N1, BM)
    e = pid0 // iblocks
    i = (pid0 % iblocks) * BM + tl.arange(0, BM)
    imask = i < N1
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    jmask = j < K
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for r0 in range(0, CAP, BK):
        r = r0 + tl.arange(0, BK)
        rmask = r < CAP
        tok = tl.load(idx_ptr + e * CAP + r, mask=rmask, other=0).to(tl.int64)
        pt = tl.load(p_ptr + e.to(tl.int64) * s_pe + r[None, :].to(tl.int64) * s_pr + i[:, None], mask=imask[:, None] & rmask[None, :], other=0.0)
        a = tl.load(a_ptr + tok[:, None] * s_a + j[None, :], mask=rmask[:, None] & jmask[None, :], other=0.0)
        acc = tl.dot(pt, a, acc)
    tl.store(c_ptr + e.to(tl.int64) * s_ce + i[:, None].to(tl.int64) * s_ci + j[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=imask[:, None] & jmask[None, :])


def gm(a, idx, b):
    n, cap = idx.shape
    N, K = b.shape[1], b.shape[2]
    c = torch.empty(n, cap, N, dtype=a.dtype, device=a.device)
    _gm_kernel[lambda M_: (n * triton.cdiv(cap, M_["BM"]), triton.cdiv(N, M_["BN"]))](
        a, idx, b, c, cap, N, K, a.stride(0), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


def gk(p, a, idx):
    n, cap, N1 = p.shape
    K = a.shape[1]
    c = torch.empty(n, N1, K, dtype=a.dtype, device=a.device)
    _gk_kernel[lambda M_: (n * triton.cdiv(N1, M_["BM"]), triton.cdiv(K, M_["BN"]))](
        p, a, idx, c, cap, N1, K, p.stride(0), p.stride(1), a.stride(0), c.stride(0), c.stride(1))
    return c


def timed(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="d12:16:2,d11:16:2,d11:64:8,d11:128:16")
ap.add_argument("--tokens", type=int, default=32768)
ap.add_argument("--out", default="")
a = ap.parse_args()
W = {"d10": 768, "d11": 896, "d12": 1024, "d13": 1152}
dev = "cuda"
print(torch.cuda.get_device_name(), flush=True)
for spec in a.shapes.split(","):
    sz, n, G = spec.split(":"); n, G = int(n), int(G)
    d = W[sz]; H = 4 * d; h = H // G; M = a.tokens; cap = G * M // n
    torch.manual_seed(0)
    x = torch.randn(M, d, device=dev, dtype=torch.bfloat16)
    idx = torch.stack([torch.randperm(M, device=dev)[:cap] for _ in range(n)]).to(torch.int32)
    flat = idx.reshape(-1)
    w12 = (torch.randn(n, 2 * h, d, device=dev) * 0.02).to(torch.bfloat16)
    p = torch.randn(n, cap, 2 * h, device=dev, dtype=torch.bfloat16)
    xe = _gather_rows(x, flat).view(n, cap, d)
    ref = torch.bmm(xe.float(), w12.float().mT)
    e1 = float((gm(x, idx, w12).float() - ref).abs().max() / ref.abs().max())
    refk = torch.bmm(p.float().mT, xe.float())
    e2 = float((gk(p, x, idx).float() - refk).abs().max() / refk.abs().max())
    fl = 2 * n * cap * 2 * h * d
    res = dict(gpu=torch.cuda.get_device_name(), shape=spec, d=d, experts=n, active=G, tokens=M, cap=cap,
               err_gm=e1, err_gk=e2,
               base_fwd_ms=timed(lambda: torch.bmm(_gather_rows(x, flat).view(n, cap, d), w12.mT)),
               gather_ms=timed(lambda: _gather_rows(x, flat)),
               bmm_fwd_ms=timed(lambda: torch.bmm(xe, w12.mT)),
               gm_ms=timed(lambda: gm(x, idx, w12)),
               bmm_wgrad_ms=timed(lambda: torch.bmm(p.mT, xe)),
               gk_ms=timed(lambda: gk(p, x, idx)))
    res["gm_cfg"], res["gk_cfg"] = str(_gm_kernel.best_config), str(_gk_kernel.best_config)
    tf = lambda ms: fl / ms / 1e9
    print(f"{spec}: cap {cap}, h {h}; rel err gm {e1:.1e}, gk {e2:.1e}\n"
          f"   forward: gather {res['gather_ms']:.3f} + bmm {res['bmm_fwd_ms']:.3f} ({tf(res['bmm_fwd_ms']):.0f} TFLOPS) = {res['base_fwd_ms']:.3f} ms | "
          f"gm {res['gm_ms']:.3f} ms ({tf(res['gm_ms']):.0f} TFLOPS)\n"
          f"   weight gradient: bmm {res['bmm_wgrad_ms']:.3f} ms ({tf(res['bmm_wgrad_ms']):.0f} TFLOPS) | gk {res['gk_ms']:.3f} ms ({tf(res['gk_ms']):.0f} TFLOPS)",
          flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
