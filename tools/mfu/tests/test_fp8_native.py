#!/usr/bin/env python3
"""Native rowwise FP8 (src/models/layers/fp8_native.py), one GPU.
1. rowcast: e4m3 values and row scales bit for bit the plain-torch quantization (reference()'s).
2. fp8_linear forward against the dequantized-GEMM reference (fp32 accumulation): relative error at fp32/bf16 level; and
   against the bf16 GEMM: the size of the FP8 rounding (reported).
3. Its backward: dx = dy W and dw = dy^T x (the bf16 input), bf16 GEMMs, against fp32 references.
4. SwiGLU + w3: forward against the reference on the model's bf16 product, backward dz against autograd of the bf16
   SwiGLU at the same w3, dw3 against dy^T a_q.
5. Phase 2, the fused MLP (FP8 dgrads): the forward bit for bit phase 1's; the SwiGLU backward kernel's FP8 rows bit for
   bit the row cast of its own bf16 dz; dx, dw12 and dw3 against fp32 autograd of the bf16 MLP (FP8 dgrad error reported).
usage: python test_fp8_native.py"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.models.layers.fp8_native import rowcast, fp8_linear, fp8_swiglu_w3, fp8_mlp, reference, swiglu_rowcast, swiglu_bwd, FMAX, E4  # noqa

dev, BF = "cuda", torch.bfloat16
torch.manual_seed(0)
ok = True


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


for M, K, N in ((4096, 640, 1920), (2048, 1280, 5120), (1000, 1792, 1792)):
    x = (torch.randn(M, K, device=dev) * torch.rand(M, 1, device=dev) * 4).to(BF)
    x[3] = 0
    w = torch.randn(N, K, device=dev) * 0.02
    b = torch.randn(N, device=dev) * 0.1
    q, s = rowcast(x)
    a = x.float().abs().amax(1, keepdim=True).clamp_min(1e-12)
    rq = (x.float() * (FMAX / a)).clamp(-FMAX, FMAX).to(E4)
    same = torch.equal(q.view(torch.uint8), rq.view(torch.uint8)) and torch.allclose(s, a / FMAX, rtol=0, atol=0)
    y = fp8_linear(x, w, b)
    yr = reference(x, w, b)
    yb = (x.float() @ w.float().t() + b).to(BF)
    e_ref, e_bf = rel(y, yr), rel(y, yb)
    xl, wl, bl = x.clone().requires_grad_(), w.clone().requires_grad_(), b.clone().requires_grad_()
    dy = torch.randn(M, N, device=dev).to(BF)
    fp8_linear(xl, wl, bl).backward(dy)
    dx_r = dy.float() @ w.float()
    dw_r = dy.float().t() @ x.float()          # the bf16 input (fp8_linear keeps it for the weight gradient)
    e_dx, e_dw, e_db = rel(xl.grad, dx_r), rel(wl.grad, dw_r), rel(bl.grad, dy.float().sum(0))
    good = same and e_ref < 5e-3 and e_dx < 6e-3 and e_dw < 6e-3 and e_db < 6e-3 and torch.isfinite(y).all()
    ok &= bool(good)
    print(f"linear M={M} K={K} N={N}: rowcast bit for bit {same}; y vs dequantized reference {e_ref:.1e}, vs bf16 GEMM {e_bf:.1e} "
          f"(FP8 rounding); dx {e_dx:.1e} dw {e_dw:.1e} db {e_db:.1e}" + ("" if good else "  FAIL"))

for M, C in ((2048, 640), (1024, 1280)):
    H = 4 * C
    z = (torch.randn(M, 2 * H, device=dev) * 2).to(BF)
    w3 = torch.randn(C, H, device=dev) * 0.02
    zl, wl = z.clone().requires_grad_(), w3.clone().requires_grad_()
    y = fp8_swiglu_w3(zl, wl)
    z1, z2 = z.chunk(2, -1)
    a = torch.nn.functional.silu(z1) * z2            # the model's bf16 product
    yr = reference(a, w3)
    yb = (a.float() @ w3.float().t()).to(BF)
    aq, sa = swiglu_rowcast(z)
    dy = torch.randn(M, C, device=dev).to(BF)
    y.backward(dy)
    zr = z.clone().float().requires_grad_()
    r1, r2 = zr.chunk(2, -1)
    ((torch.nn.functional.silu(r1) * r2) @ w3.float().t()).backward(dy.float())
    dw_r = dy.float().t() @ (aq.float() * sa)
    e_y, e_b, e_dz, e_dw = rel(y, yr), rel(y, yb), rel(zl.grad, zr.grad), rel(wl.grad, dw_r)
    good = e_y < 5e-3 and e_dz < 1e-2 and e_dw < 6e-3 and torch.isfinite(y).all()
    ok &= bool(good)
    print(f"swiglu+w3 M={M} C={C}: y vs reference {e_y:.1e}, vs bf16 {e_b:.1e}; dz vs fp32 autograd {e_dz:.1e}; dw3 {e_dw:.1e}"
          + ("" if good else "  FAIL"))
for M, C in ((2048, 640), (1024, 1280)):
    H = 4 * C
    x = (torch.randn(M, C, device=dev) * 2).to(BF)
    w12 = torch.randn(2 * H, C, device=dev) * (C ** -0.5)
    w3 = torch.randn(C, H, device=dev) * (H ** -0.5)
    dy = torch.randn(M, C, device=dev).to(BF)
    L = [t.clone().requires_grad_() for t in (x, w12, w3)]
    y2 = fp8_mlp(*L)
    y1 = fp8_swiglu_w3(fp8_linear(x, w12), w3)
    same_fwd = torch.equal(y2, y1)
    y2.backward(dy)
    R = [t.clone().float().requires_grad_() for t in (x, w12, w3)]
    zr = R[0] @ R[1].t()
    a1, a2 = zr.chunk(2, -1)
    ((torch.nn.functional.silu(a1) * a2) @ R[2].t()).backward(dy.float())
    z = (x.float() @ w12.float().t()).to(BF)   # close to the FP8 forward's z; only for the kernel check below
    da = torch.randn(M, H, device=dev).to(BF)
    dz, dzq, sdz = swiglu_bwd(z, da)
    q2, s2 = rowcast(dz)
    kern = torch.equal(dzq.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(sdz, s2)
    e_dx, e_w12, e_w3 = rel(L[0].grad, R[0].grad), rel(L[1].grad, R[1].grad), rel(L[2].grad, R[2].grad)
    good = same_fwd and kern and e_dx < 8e-2 and e_w12 < 8e-2 and e_w3 < 8e-2 and torch.isfinite(L[0].grad).all()
    ok &= bool(good)
    print(f"mlp phase 2 M={M} C={C}: forward bit for bit phase 1 {same_fwd}; SwiGLU backward FP8 rows bit for bit {kern}; "
          f"vs fp32 autograd: dx {e_dx:.1e} dw12 {e_w12:.1e} dw3 {e_w3:.1e}" + ("" if good else "  FAIL"))
print("FP8_NATIVE_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
