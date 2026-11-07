#!/usr/bin/env python3
"""Lever K's kernels alone (src/models/layers/fused_norm.py) against the same ops compiled by Inductor, one GPU: forward +
backward of one site per width (B 32, N 256 image tokens, bf16, the modulation as chunks of one adaLN output), CUDA
events over many iterations (wall, bound by the host's launches in eager mode) and the profiler's sum of kernel times
(GPU time: what the kernels cost inside a compiled step, where Inductor launches them from its own code). Site 1 = norm1 + modulation; site 2 = residual add + norm2 + modulation (with the upstream
gradient of x_new). Also a sweep of the backward's rows per program and warps. The bytes the site must move at least
(each [B, N, C] tensor once) set the reference bandwidth.
usage: python bench_fused_norm.py [--widths 1024,1280,1536,1792] [--iters 200]"""
import argparse, os, sys
import torch
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen"))
sys.path.insert(0, SRC)
from src.models.layers import fused_norm as fn  # noqa: E402

EPS = 1e-6


def ref(x, a, gate, w, shift, scale, has_res):
    if has_res:
        x = x + gate * a
    hs = x.float()
    hs = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + EPS)
    return x, (w * hs).to(x.dtype) * (1 + scale) + shift


def gpu_time(f, iters):
    """Sum of the kernels' device time per iteration (ms), from the profiler."""
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total for e in prof.key_averages()) / iters / 1e3


def timed(f, iters):
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1024,1280,1536,1792")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--N", type=int, default=256)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev, bf = "cuda", torch.bfloat16
    cref = torch.compile(ref, dynamic=False)
    print(f"{torch.cuda.get_device_name()}; B {args.B}, N {args.N}; forward + backward per site (ms); min bytes at 3 TB/s")
    for C in map(int, args.widths.split(",")):
        B, N = args.B, args.N
        x = torch.randn(B, N, C, device=dev, dtype=bf).requires_grad_(True)
        a = (0.5 * torch.randn(B, N, C, device=dev, dtype=bf)).requires_grad_(True)
        mod = (0.3 * torch.randn(B, 1, 6 * C, device=dev, dtype=bf)).requires_grad_(True)
        w = (1 + 0.1 * torch.randn(C, device=dev)).requires_grad_(True)
        dh, dxn = torch.randn(B, N, C, device=dev, dtype=bf), torch.randn(B, N, C, device=dev, dtype=bf)
        unit = B * N * C * 2 / 3e12 * 1e3   # ms to move one [B, N, C] bf16 tensor at 3 TB/s

        def run(impl, has_res):
            def f():
                shift, scale, gate = mod.chunk(6, dim=-1)[:3]
                if impl == "k":
                    if has_res:
                        xn, h = fn.res_mod_rmsnorm(x, a, gate, w, shift, scale, EPS)
                    else:
                        h = fn.mod_rmsnorm(x, w, shift, scale, EPS); xn = None
                else:
                    xn, h = (cref if impl == "inductor" else ref)(x, a, gate, w, shift, scale, has_res)
                outs, gr = ([h, xn], [dh, dxn]) if has_res else ([h], [dh])
                torch.autograd.backward(outs, gr)
            return f
        for has_res, nbytes in ((False, 2 + 3), (True, 4 + 6)):   # site 1: x, h | dh, dx (+ x again); site 2: x, a, xn, h | dh, dxn, xn, a, dx, da
            site = "site 2" if has_res else "site 1"
            t = {impl: gpu_time(run(impl, has_res), 50) for impl in ("eager", "inductor", "k")}
            wall = {impl: timed(run(impl, has_res), args.iters) for impl in ("inductor", "k")}
            print(f"d{C} {site} GPU time: eager {t['eager']:.3f}, inductor {t['inductor']:.3f}, K {t['k']:.3f} ms "
                  f"(K saves {t['inductor'] - t['k']:+.3f} ms per site; floor {nbytes * unit:.3f}); wall in eager mode: "
                  f"inductor {wall['inductor']:.3f}, K {wall['k']:.3f}")
        best = None
        for rows in (16, 32, 64):
            for warps in (4, 8, 16):
                fn.ROWS_B = rows
                os.environ["GPIC_MFU_FUSED_NORM_WARPS"] = str(warps)
                try:
                    tk = gpu_time(run("k", True), 30)
                except Exception as e:   # register pressure etc.
                    print(f"   rows {rows} warps {warps}: {type(e).__name__}"); continue
                best = min(best or (9e9, 0, 0), (tk, rows, warps))
                print(f"   site 2 GPU time, backward rows per program {rows}, warps {warps}: {tk:.3f} ms")
        fn.ROWS_B = 16
        os.environ.pop("GPIC_MFU_FUSED_NORM_WARPS", None)
        print(f"d{C} best site-2 setting: rows {best[1]}, warps {best[2]}: {best[0]:.3f} ms GPU time")


if __name__ == "__main__":
    main()
