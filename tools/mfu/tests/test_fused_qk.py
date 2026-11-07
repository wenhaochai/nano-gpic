#!/usr/bin/env python3
"""Tests of MFU lever K2 (src/models/layers/fused_qk.py: the image blocks' q/k RMSNorm, RoPE and the k/v concatenation).
CPU (TRITON_INTERPRET=1 when no GPU is visible) or one GPU. Small heads on CPU (Hc 16, 3 heads: a padded head tile), the
model's Hc 64 with 16 and 28 heads on the GPU.
1. Forward: q, k against an fp64 computation of the same math (the normalized values rounded to bf16 where the model
   casts): correctly rounded up to fp32 noise (on the GPU also one flip of the normalized value by the approximate rsqrt,
   rare); v exactly the concatenation.
2. Backward: the gradients of qkv, kvy and both norm weights against fp64 autograd: bf16 rounding level (fp32 for the
   weights) and no larger than the eager ops' own error; two runs give the same bits.
usage: python test_fused_qk.py"""
import os, sys
import torch
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
sys.path.insert(0, SRC)
from src.models.layers.fused_qk import qk_prep, reference  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF, F64 = torch.bfloat16, torch.float64
EPS = 1e-6


def ulp(v):
    return torch.pow(2.0, torch.floor(torch.log2(v.abs().clamp_min(2.0 ** -126))) - 7)


def rnd(t):   # value rounded to bf16, gradient passed through
    return t + (t.detach().to(BF).to(F64) - t.detach())


def ref64(qkv, kvy, wq, wk, cos, sin, H):
    leaf = lambda t: t.detach().to(F64).requires_grad_(True)
    QKV, KVY, WQ, WK = map(leaf, (qkv, kvy, wq, wk))
    c, s = cos.to(F64)[None, :, None], sin.to(F64)[None, :, None]
    B, N, C3 = qkv.shape
    C = C3 // 3

    def norm(x, w):
        return rnd(w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)))

    def rope(x):
        xf = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = xf[..., 0], xf[..., 1]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(3)

    q, kx, vx = QKV.view(B, N, 3, H, C // H).unbind(2)
    nq, nk = norm(q, WQ), norm(kx, WK)
    ky, vy = KVY.view(B, -1, 2, H, C // H).unbind(2)
    nky = norm(ky, WK)
    q_out, k_out = rope(nq), torch.cat([rope(nk), nky], dim=1)
    v_out = torch.cat([vx, vy], dim=1)
    # the size of a one-ulp flip of a normalized value, carried through the RoPE (|cos| + |sin| <= sqrt 2)
    with torch.no_grad():
        flip_q = 1.5 * rope(ulp(nq).abs()).abs() + 1.5 * ulp(nq).abs().max()
        flip_k = torch.cat([1.5 * rope(ulp(nk)).abs() + 1.5 * ulp(nk).abs().max(), ulp(nky)], dim=1)
        noise = lambda t: 4e-7 * (t.abs() + 1e-30)
    return (QKV, KVY, WQ, WK), q_out, k_out, v_out, flip_q, flip_k, noise


def ratio(mine, ref, extra):
    tol = 0.5 * ulp(ref) + 4e-7 * ref.abs().max() + extra
    return float(((mine.double() - ref).abs() / tol).max())


def check(B, N, T, H, HC, seed):
    g = torch.Generator().manual_seed(seed)
    C = H * HC
    r = lambda *sz: torch.randn(*sz, generator=g)
    qkv = r(B, N, 3 * C).to(BF).to(DEV)
    kvy = r(B, T, 2 * C).to(BF).to(DEV)
    wq, wk = (1 + 0.1 * r(HC)).to(DEV), (1 + 0.1 * r(HC)).to(DEV)
    ang = r(N, HC // 2) * 3
    cos, sin = ang.cos().to(DEV), ang.sin().to(DEV)
    dq, dk, dv = r(B, N, H, HC).to(BF).to(DEV), r(B, N + T, H, HC).to(BF).to(DEV), r(B, N + T, H, HC).to(BF).to(DEV)
    ok = True
    leaves = [t.detach().clone().requires_grad_(True) for t in (qkv, kvy, wq, wk)]
    q, k, v = qk_prep(*leaves, cos, sin, H, EPS)
    L64, q64, k64, v64, fq, fk, _ = ref64(qkv, kvy, wq, wk, cos, sin, H)
    extra_q = fq if DEV == "cuda" else 0.0
    extra_k = fk if DEV == "cuda" else 0.0
    rq, rk = ratio(q.detach(), q64.detach(), extra_q), ratio(k.detach(), k64.detach(), extra_k)
    qe, ke, ve = reference(qkv, kvy, wq, wk, cos, sin, H, EPS)
    rqe, rke = ratio(qe, q64.detach(), 0.0), ratio(ke, k64.detach(), 0.0)
    vsame = torch.equal(v, v64.detach().to(BF))
    ok &= rq <= 1.0 and rk <= 1.0 and vsame
    line = (f"B={B} N={N} T={T} H={H} Hc={HC}: forward error / (0.5 ulp + fp32 noise{' + rsqrt flip' if DEV == 'cuda' else ''}): "
            f"q {rq:.2f}, k {rk:.2f} (eager ops {rqe:.2f}, {rke:.2f}); v exact {vsame}")
    torch.autograd.backward([q, k, v], [dq, dk, dv])
    torch.autograd.backward([q64, k64, v64], [t.to(F64) for t in (dq, dk, dv)])
    le = [t.detach().clone().requires_grad_(True) for t in (qkv, kvy, wq, wk)]
    torch.autograd.backward(list(reference(*le, cos, sin, H, EPS)), [dq, dk, dv])
    for nm, mine, r64, eg in zip(("qkv", "kvy", "q_norm.weight", "k_norm.weight"), leaves, L64, le):
        ref = r64.grad
        rel = float((mine.grad.double() - ref).norm() / ref.norm())
        rel_e = float((eg.grad.double() - ref).norm() / ref.norm())
        lim = 2e-5 if "weight" in nm else 6e-3
        good = rel < lim and rel <= max(1.5 * rel_e, lim / 4)
        ok &= good
        line += f"\n    d{nm}: rel err {rel:.1e} (eager {rel_e:.1e}){'' if good else '  FAIL'}"
    l2 = [t.detach().clone().requires_grad_(True) for t in (qkv, kvy, wq, wk)]
    torch.autograd.backward(list(qk_prep(*l2, cos, sin, H, EPS)), [dq, dk, dv])
    same = all(torch.equal(a.grad, b.grad) for a, b in zip(leaves, l2))
    ok &= same
    print(line + f"\n    second run, same bits: {same}" + ("" if ok else "\n    FAIL"))
    return ok


if __name__ == "__main__":
    print(f"device {DEV}, interpreter {os.environ.get('TRITON_INTERPRET') == '1'}")
    ok = check(2, 12, 5, 3, 16, 0)
    ok &= check(3, 9, 7, 2, 16, 1)
    if DEV == "cuda":
        ok &= check(4, 256, 52, 16, 64, 2)
        ok &= check(4, 256, 37, 28, 64, 3)
    print("FUSED_QK_TEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
