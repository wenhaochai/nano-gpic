#!/usr/bin/env python3
"""Tests of MFU lever K (src/models/layers/fused_norm.py: residual + RMSNorm + adaLN modulation, forward and backward).
CPU (TRITON_INTERPRET=1, set below when no GPU is visible) or one GPU. Odd sizes (N = 37 tokens, C = 80 or 96: masked
columns and a partial row tile), the modulation as chunks of one [B, 1, 6C] tensor (strided views, as in the blocks).
1. Forward: x_new and h against an fp64 computation of the same math (x_new and n = w * x_hat rounded to bf16 where the
   model casts): at most 1 bf16 ulp apart; the model's eager ops (which round every bf16 intermediate) as the yardstick.
2. Backward: every gradient against fp64 autograd of the same math: relative error at bf16 rounding level (fp32 for the
   norm weight) and no larger than the eager bf16 ops' own error; the column sums deterministic (two runs, same bits).
3. The site without the residual (norm1) and the site whose x_new has no later use (no upstream gradient of x_new).
usage: python test_fused_norm.py"""
import os, sys
import torch
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
sys.path.insert(0, SRC)
from src.models.layers.fused_norm import mod_rmsnorm, res_mod_rmsnorm, reference  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF = torch.bfloat16
EPS = 1e-6


def rounding_ratio(mine, ref, noise):
    """max |mine - ref| / (0.5 bf16 ulp of ref + noise): at most 1 when mine is ref correctly rounded to bf16, up to the
    fp32 rounding of its unrounded intermediates (noise)."""
    e = torch.floor(torch.log2(ref.abs().clamp_min(2.0 ** -126)))
    tol = 0.5 * torch.pow(2.0, e - 7) + noise
    return float(((mine.double() - ref).abs() / tol).max())


def inputs(B, N, C, seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sd=1.0: (torch.randn(*s, generator=g) * sd)
    x = r(B, N, C).to(BF).to(DEV)
    a = r(B, N, C, sd=0.5).to(BF).to(DEV)
    mod = r(B, 1, 6 * C, sd=0.3).to(BF).to(DEV)            # one adaLN output, chunked as in the blocks
    w = (1 + r(C, sd=0.1)).to(DEV)                        # fp32 norm weight
    shift, scale, gate = mod.chunk(6, dim=-1)[:3]
    dh = r(B, N, C).to(BF).to(DEV)
    dxn = r(B, N, C).to(BF).to(DEV)
    return x, a, gate, w, shift, scale, dh, dxn


def ref64(x, a, gate, w, shift, scale, has_res):
    """fp64 autograd of the same math, the model's explicit casts emulated (x_new and n rounded to bf16)."""
    leaf = lambda t: t.detach().to(torch.float64).requires_grad_(True)
    X, A, G, Wt, SH, SC = map(leaf, (x, a, gate, w, shift, scale))
    xn = X + G * A if has_res else X
    xn_r = xn + (xn.detach().to(BF).to(torch.float64) - xn.detach())      # value rounded, gradient passed through
    t = xn_r * torch.rsqrt(xn_r.pow(2).mean(-1, keepdim=True) + EPS)
    n = Wt * t
    n_r = n + (n.detach().to(BF).to(torch.float64) - n.detach())
    h = n_r * (1 + SC) + SH
    with torch.no_grad():   # the fp32 rounding scale of the kernel's unrounded intermediates
        noise_h = 4e-7 * ((n_r * (1 + SC)).abs() + SH.abs())
        noise_x = 4e-7 * (X.abs() + (G * A).abs()) if has_res else None
    return (X, A, G, Wt, SH, SC), xn_r, h, noise_x, noise_h


def check(B, N, C, has_res, use_dxn, seed):
    x, a, gate, w, shift, scale, dh, dxn = inputs(B, N, C, seed)
    ok = True
    # forward
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, a, gate, w, shift, scale)]
    X, A, G, Wt, SH, SC = leaves
    if has_res:
        xn, h = res_mod_rmsnorm(X, A, G, Wt, SH, SC, EPS)
    else:
        h = mod_rmsnorm(X, Wt, SH, SC, EPS); xn = None
    L64, xn64, h64, noise_x, noise_h = ref64(x, a, gate, w, shift, scale, has_res)
    r_h = rounding_ratio(h.detach(), h64.detach(), noise_h)
    xe, he = reference(x, a, gate, w, shift, scale, EPS, has_res)
    r_he = rounding_ratio(he, h64.detach(), noise_h)
    line = (f"B={B} N={N} C={C} res={has_res} dxn={use_dxn}: forward error / (0.5 ulp + fp32 noise), at most 1 = "
            f"correctly rounded: h {r_h:.2f} (eager ops {r_he:.2f})")
    ok &= r_h <= 1.0
    if has_res:
        r_x = rounding_ratio(xn.detach(), xn64.detach(), noise_x)
        line += f", x_new {r_x:.2f}"
        ok &= r_x <= 1.0
    # backward
    outs, grads = [h], [dh]
    if has_res and use_dxn:
        outs.append(xn); grads.append(dxn)
    torch.autograd.backward(outs, grads)
    outs64 = [h64] + ([xn64] if has_res and use_dxn else [])
    torch.autograd.backward(outs64, [g.to(torch.float64) for g in grads])
    # eager bf16 ops as the yardstick
    le = [t.detach().clone().requires_grad_(True) for t in (x, a, gate, w, shift, scale)]
    xe, he = reference(*le, EPS, has_res)
    oe = [he] + ([xe] if has_res and use_dxn else [])
    torch.autograd.backward(oe, grads)
    names = ["x", "a", "gate", "w", "shift", "scale"]
    used = [True, has_res, has_res, True, True, True]
    for nm, u, mine, r64, eg in zip(names, used, leaves, L64, le):
        if not u:
            ok &= mine.grad is None or float(mine.grad.abs().max()) == 0
            continue
        ref = r64.grad.to(torch.float64)
        rel = float((mine.grad.double() - ref).norm() / ref.norm().clamp_min(1e-30))
        rel_e = float((eg.grad.double() - ref).norm() / ref.norm().clamp_min(1e-30))
        lim = 2e-5 if nm == "w" else 6e-3
        good = rel < lim and rel <= max(1.5 * rel_e, lim / 4)
        ok &= good
        line += f"\n    d{nm}: rel err {rel:.1e} (eager {rel_e:.1e}){'' if good else '  FAIL'}"
    # determinism of the column sums
    l2 = [t.detach().clone().requires_grad_(True) for t in (x, a, gate, w, shift, scale)]
    if has_res:
        xn2, h2 = res_mod_rmsnorm(*l2, EPS)
        o2 = [h2] + ([xn2] if use_dxn else [])
    else:
        o2 = [mod_rmsnorm(l2[0], l2[3], l2[4], l2[5], EPS)]
    torch.autograd.backward(o2, grads)
    same = all(torch.equal(p.grad, q.grad) for p, q, u in zip(leaves, l2, used) if u)
    ok &= same
    line += f"\n    second run, same bits: {same}"
    print(line + ("" if ok else "\n    FAIL"))
    return ok


if __name__ == "__main__":
    print(f"device {DEV}, interpreter {os.environ.get('TRITON_INTERPRET') == '1'}")
    ok = True
    ok &= check(3, 37, 80, True, True, 0)
    ok &= check(2, 37, 96, False, False, 1)
    ok &= check(2, 37, 96, True, False, 2)
    if DEV == "cuda":
        for C in (1024, 1280, 1792):
            ok &= check(4, 256, C, True, True, 3)
            ok &= check(4, 256, C, False, False, 4)
    print("FUSED_NORM_TEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
