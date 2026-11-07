#!/usr/bin/env python3
"""Lever K2 on lever E's packed captions (src/models/layers/fused_qk.py qk_prep_packed): against the padded K2 kernels
(test_fused_qk.py checks those against fp64). The same captions once padded [B, Tp, 2C] and once packed [S, 2C] (real
rows, then dummy rows with random values) with pidx = b * Tp + t (B * Tp for a dummy row).
1. Forward: q, the image keys and values, and every real caption key and value bit for bit the padded kernel's.
   Every key and value row past a caption is exactly zero, also when the memory it reuses held NaN (the CUDA cache is
   poisoned with NaN first), and on the GPU FA3 over the packed keys gives the padded keys' output bit for bit, all finite
   (uninitialised rows made every loss NaN, w1865).
2. Backward (dk, dv zero past each caption, as FA3 gives): d qkv, d q_norm.weight and the real rows of d kvy bit for bit;
   the dummy rows' gradient exactly zero; d k_norm.weight up to fp32 summation order (other row groups per program).
CPU (TRITON_INTERPRET=1) or one GPU. usage: python test_fused_qk_packed.py"""
import os, sys
import torch
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.models.layers.fused_qk import qk_prep, qk_prep_packed  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF = torch.bfloat16


def check(lens, N, H, HC, seed, dummy=5):
    g = torch.Generator().manual_seed(seed)
    r = lambda *sz: torch.randn(*sz, generator=g)
    B, C = len(lens), H * HC
    Tp = -(-max(lens) // 8) * 8
    qkv = r(B, N, 3 * C).to(BF).to(DEV)
    kvy_pad = r(B, Tp, 2 * C).to(BF).to(DEV)
    real = torch.cat([kvy_pad[b, :L] for b, L in enumerate(lens)])
    kvy = torch.cat([real, r(dummy, 2 * C).to(BF).to(DEV)])
    pidx = torch.tensor([b * Tp + t for b, L in enumerate(lens) for t in range(L)] + [B * Tp] * dummy, device=DEV)
    wq, wk = (1 + 0.1 * r(HC)).to(DEV), (1 + 0.1 * r(HC)).to(DEV)
    ang = r(N, HC // 2) * 3
    cos, sin = ang.cos().to(DEV), ang.sin().to(DEV)
    keep = torch.zeros(B, N + Tp, 1, 1, dtype=BF, device=DEV)
    for b, L in enumerate(lens):
        keep[b, :N + L] = 1
    dq = r(B, N, H, HC).to(BF).to(DEV)
    dk, dv = r(B, N + Tp, H, HC).to(BF).to(DEV) * keep, r(B, N + Tp, H, HC).to(BF).to(DEV) * keep
    if DEV == "cuda":   # poison the cache: the next allocations reuse NaN-filled blocks
        junk = torch.full((B * (N + Tp) * C * 8,), float("nan"), dtype=BF, device=DEV)
        del junk
    A = [t.clone().requires_grad_(True) for t in (qkv, kvy_pad, wq, wk)]
    P = [t.clone().requires_grad_(True) for t in (qkv, kvy, wq, wk)]
    qa, ka, va = qk_prep(A[0], A[1], A[2], A[3], cos, sin, H, 1e-6)
    qp, kp, vp = qk_prep_packed(P[0], P[1], pidx, Tp, P[2], P[3], cos, sin, H, 1e-6)
    m = keep.bool().expand_as(ka)
    fwd = torch.equal(qa, qp) and torch.equal(ka[m], kp[m]) and torch.equal(va[m], vp[m])
    fwd &= bool((kp[~m] == 0).all()) and bool((vp[~m] == 0).all())
    if DEV == "cuda":
        from src.models.layers.fa3_op import fa3_attention
        seq = torch.tensor([N + L for L in lens], dtype=torch.int32, device=DEV)
        oa, op = fa3_attention(qa.detach(), ka.detach(), va.detach(), seq), fa3_attention(qp.detach(), kp.detach(), vp.detach(), seq)
        fwd &= bool(torch.isfinite(op).all()) and torch.equal(oa, op)
    torch.autograd.backward([qa, ka, va], [dq, dk, dv])
    torch.autograd.backward([qp, kp, vp], [dq, dk, dv])
    S = len(real)
    real_rows = torch.cat([A[1].grad[b, :L] for b, L in enumerate(lens)])
    bwd = (torch.equal(A[0].grad, P[0].grad) and torch.equal(A[2].grad, P[2].grad)
           and torch.equal(real_rows, P[1].grad[:S]) and bool((P[1].grad[S:] == 0).all()))
    dwk = float((A[3].grad.double() - P[3].grad.double()).norm() / A[3].grad.double().norm())
    ok = fwd and bwd and dwk < 1e-5
    print(f"lens {lens} N={N} H={H} Hc={HC} Tp={Tp} dummy={dummy}: forward bit for bit, padded rows zero{', FA3 equal and finite' if DEV == 'cuda' else ''} {fwd}; d qkv, d q_norm.weight, "
          f"real d kvy bit for bit and dummy rows zero {bwd}; d k_norm.weight rel diff {dwk:.1e}" + ("" if ok else "  FAIL"))
    return ok


if __name__ == "__main__":
    print(f"device {DEV}, interpreter {os.environ.get('TRITON_INTERPRET') == '1'}")
    ok = check([5, 1, 12], 9, 3, 16, 0)
    ok &= check([7, 7], 12, 2, 16, 1, dummy=3)
    if DEV == "cuda":
        ok &= check([40, 17, 176, 52], 256, 16, 64, 2, dummy=91)
        ok &= check([24, 300, 8, 61, 33], 256, 28, 64, 3, dummy=34)
    print("FUSED_QK_PACKED_TEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
