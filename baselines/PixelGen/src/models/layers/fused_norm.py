"""MFU lever K (GPIC_MFU_FUSED_NORM=1, docs/mfu.md): the image blocks' residual add, RMSNorm and adaLN modulation as one
Triton kernel in the forward and one in the backward.

    site 1 (norm1):  h = RMSNorm(x) * (1 + scale) + shift                                   mod_rmsnorm
    site 2 (norm2):  x_new = x + gate * a;  h = RMSNorm(x_new) * (1 + scale) + shift        res_mod_rmsnorm

Inductor fuses each site's forward into a kernel or two. Its backward is several kernels that each read the large
[B, N, C] tensors again: RMSNorm's row reduction, and column reductions over the tokens for the gradients of shift, scale
and gate (per sample) and of the norm weight (over all tokens); at d1280 those column reductions
(triton_red_fused__to_copy_mul_sum) take 4.4 ms of the backward's 13.2 ms of Triton time at 1.2 TB/s (Inductor
profile_bandwidth, probe7). Here one backward kernel per site reads dh, x_new, a and the upstream gradient of x_new once,
writes dx and da, and keeps the column sums of its rows in registers; the per-program partial sums are added in a fixed
order afterwards, so the result is deterministic.

Math: the model's ops (src/models/layers/rmsnorm.py; modulate in JiT_T2I.py) in fp32, with the explicit casts of the
model kept (x_new and n = bf16(w * x_hat) rounded to the input dtype, so the norm and the next block see the same x_new),
the other intermediates unrounded (as Inductor's fused kernels compute them), rsqrt as Inductor's (libdevice), every
output rounded once to its tensor's dtype. The same math up to rounding and fp32 summation order.
Registered with torch.library.triton_op: torch.compile traces the wrapper and launches the kernels from its own code.
"""
import os
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

try:
    from triton.language.extra import libdevice
except ImportError:
    from triton.language.extra.cuda import libdevice

INTERP = os.environ.get("TRITON_INTERPRET") == "1"            # CPU tests: no libdevice in the interpreter
_launch = (lambda k: k) if INTERP else wrap_triton            # (the interpreter's kernels are not JITFunctions)
ROWS_F = int(os.environ.get("GPIC_MFU_FUSED_NORM_ROWS_F", "4"))    # rows per forward program
ROWS_B = int(os.environ.get("GPIC_MFU_FUSED_NORM_ROWS_B", "16"))   # rows per backward program (all of one sample)


def _warps(block_c):
    return int(os.environ.get("GPIC_MFU_FUSED_NORM_WARPS", "0")) or max(4, block_c // 256)


@triton.jit
def _rsqrt(v, INTERP: tl.constexpr):
    if INTERP:
        return 1.0 / tl.sqrt(v)
    else:
        return libdevice.rsqrt(v)


# fp32 -> DT, round to nearest even (the interpreter's own bf16 cast truncates: emulate it there). No docstrings in the
# jitted functions: with the C++ launcher Inductor embeds their source in a triple-quoted block of its generated file.
@triton.jit
def _to(v, DT: tl.constexpr, INTERP: tl.constexpr):
    if INTERP:
        if DT == tl.bfloat16:
            u = v.to(tl.uint32, bitcast=True)
            u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
            v = u.to(tl.float32, bitcast=True)
    return v.to(DT)


@triton.jit
def _fwd_kernel(X, A, G, W, SH, SC, XN, H, RSTD, n_rows, n_tok, C, s_g, s_sh, s_sc, eps,
                HAS_RES: tl.constexpr, BLOCK_C: tl.constexpr, ROWS: tl.constexpr, INTERP: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    cm = cols < C
    w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
    for i in range(ROWS):
        row = pid * ROWS + i
        ok = row < n_rows
        m = cm & ok
        b = row // n_tok
        x = tl.load(X + row * C + cols, mask=m, other=0.0).to(tl.float32)
        if HAS_RES:
            a = tl.load(A + row * C + cols, mask=m, other=0.0).to(tl.float32)
            g = tl.load(G + b * s_g + cols, mask=m, other=0.0).to(tl.float32)
            xb = _to(x + g * a, XN.dtype.element_ty, INTERP)
            tl.store(XN + row * C + cols, xb, mask=m)
            x = xb.to(tl.float32)
        var = tl.sum(x * x, axis=0) / C
        rstd = _rsqrt(var + eps, INTERP)
        t = x * rstd
        n = _to(w * t, H.dtype.element_ty, INTERP).to(tl.float32)
        sc = tl.load(SC + b * s_sc + cols, mask=m, other=0.0).to(tl.float32)
        sh = tl.load(SH + b * s_sh + cols, mask=m, other=0.0).to(tl.float32)
        h = n * (1.0 + sc) + sh
        tl.store(H + row * C + cols, _to(h, H.dtype.element_ty, INTERP), mask=m)
        tl.store(RSTD + row, rstd, mask=ok)


@triton.jit
def _bwd_kernel(DH, DXN, XN, A, G, W, SC, RSTD, DX, DA, PSH, PSC, PG, PW, n_tok, C, tiles_per_b, s_g, s_sc,
                HAS_RES: tl.constexpr, HAS_DXN: tl.constexpr, BLOCK_C: tl.constexpr, ROWS: tl.constexpr,
                INTERP: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // tiles_per_b
    tile = pid % tiles_per_b
    cols = tl.arange(0, BLOCK_C)
    cm = cols < C
    w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
    sc = tl.load(SC + b * s_sc + cols, mask=cm, other=0.0).to(tl.float32)
    g = tl.zeros([BLOCK_C], tl.float32)
    if HAS_RES:
        g = tl.load(G + b * s_g + cols, mask=cm, other=0.0).to(tl.float32)
    acc_sh = tl.zeros([BLOCK_C], tl.float32)
    acc_sc = tl.zeros([BLOCK_C], tl.float32)
    acc_w = tl.zeros([BLOCK_C], tl.float32)
    acc_g = tl.zeros([BLOCK_C], tl.float32)
    for i in range(ROWS):
        r = tile * ROWS + i                       # token index within sample b
        ok = r < n_tok
        m = cm & ok
        row = b * n_tok + r
        dh = tl.load(DH + row * C + cols, mask=m, other=0.0).to(tl.float32)
        x = tl.load(XN + row * C + cols, mask=m, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row, mask=ok, other=0.0)
        t = x * rstd
        n = _to(w * t, DH.dtype.element_ty, INTERP).to(tl.float32)
        acc_sh += dh
        acc_sc += dh * n
        dn = dh * (1.0 + sc)
        acc_w += dn * t
        dt = dn * w
        mean = tl.sum(dt * t, axis=0) / C
        dx = rstd * (dt - t * mean)
        if HAS_DXN:
            dx += tl.load(DXN + row * C + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(DX + row * C + cols, _to(dx, DX.dtype.element_ty, INTERP), mask=m)
        if HAS_RES:
            a = tl.load(A + row * C + cols, mask=m, other=0.0).to(tl.float32)
            tl.store(DA + row * C + cols, _to(dx * g, DA.dtype.element_ty, INTERP), mask=m)
            acc_g += dx * a
    tl.store(PSH + pid * C + cols, acc_sh, mask=cm)
    tl.store(PSC + pid * C + cols, acc_sc, mask=cm)
    tl.store(PW + pid * C + cols, acc_w, mask=cm)
    if HAS_RES:
        tl.store(PG + pid * C + cols, acc_g, mask=cm)


def _vec(t, B, C):
    """A [B, 1, C] modulation chunk (a view of the adaLN output) as a [B, C] view and its row stride, without a copy."""
    t = t.reshape(B, C)
    if t.stride(1) != 1:
        t = t.contiguous()
    return t, t.stride(0)


@triton_op("gpic::res_mod_rmsnorm_fwd", mutates_args=())
def res_mod_rmsnorm_fwd(x: torch.Tensor, a: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, shift: torch.Tensor,
                        scale: torch.Tensor, eps: float, has_res: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, C = x.shape
    xc = x.contiguous()
    rows = B * N
    sh, s_sh = _vec(shift, B, C)
    sc, s_sc = _vec(scale, B, C)
    if has_res:
        ac = a.contiguous()
        g, s_g = _vec(gate, B, C)
        xn = torch.empty_like(xc)
    else:
        ac, g, s_g, xn = xc, w, 0, xc
    h = torch.empty_like(xc)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    BC = triton.next_power_of_2(C)
    _launch(_fwd_kernel)[(triton.cdiv(rows, ROWS_F),)](
        xc, ac, g, w, sh, sc, xn, h, rstd, rows, N, C, s_g, s_sh, s_sc, eps,
        HAS_RES=has_res, BLOCK_C=BC, ROWS=ROWS_F, INTERP=INTERP, num_warps=_warps(BC))
    return (xn if has_res else x.new_empty(0)), h, rstd


@res_mod_rmsnorm_fwd.register_fake
def _(x, a, gate, w, shift, scale, eps, has_res):
    B, N, C = x.shape
    return (torch.empty_like(x) if has_res else x.new_empty(0)), torch.empty_like(x), x.new_empty(B * N, dtype=torch.float32)


@triton_op("gpic::res_mod_rmsnorm_bwd", mutates_args=())
def res_mod_rmsnorm_bwd(dh: torch.Tensor, dxn: torch.Tensor, xn: torch.Tensor, a: torch.Tensor, gate: torch.Tensor,
                        w: torch.Tensor, scale: torch.Tensor, rstd: torch.Tensor, has_res: bool, has_dxn: bool
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns dx, da, and the fp32 column sums: d_gate [B, C], d_w [C], d_shift [B, C], d_scale [B, C] (empty da and
    d_gate without the residual)."""
    B, N, C = xn.shape
    tiles = triton.cdiv(N, ROWS_B)
    P = B * tiles
    dhc = dh.contiguous()
    xnc = xn.contiguous()
    sc, s_sc = _vec(scale, B, C)
    dx = torch.empty_like(xnc)
    f32 = dict(dtype=torch.float32, device=xn.device)
    psh, psc, pw = torch.empty(P, C, **f32), torch.empty(P, C, **f32), torch.empty(P, C, **f32)
    if has_res:
        ac = a.contiguous()
        g, s_g = _vec(gate, B, C)
        da = torch.empty_like(ac)
        pg = torch.empty(P, C, **f32)
    else:
        ac, g, s_g, da, pg = dhc, w, 0, dx, psh
    BC = triton.next_power_of_2(C)
    _launch(_bwd_kernel)[(P,)](
        dhc, dxn.contiguous() if has_dxn else dhc, xnc, ac, g, w, sc, rstd, dx, da, psh, psc, pg, pw, N, C, tiles, s_g, s_sc,
        HAS_RES=has_res, HAS_DXN=has_dxn, BLOCK_C=BC, ROWS=ROWS_B, INTERP=INTERP, num_warps=_warps(BC))
    dsh = psh.view(B, tiles, C).sum(1)
    dsc = psc.view(B, tiles, C).sum(1)
    dw = pw.sum(0)
    if has_res:
        return dx, da, pg.view(B, tiles, C).sum(1), dw, dsh, dsc
    return dx, xn.new_empty(0), dsh.new_empty(0), dw, dsh, dsc


@res_mod_rmsnorm_bwd.register_fake
def _(dh, dxn, xn, a, gate, w, scale, rstd, has_res, has_dxn):
    B, N, C = xn.shape
    v = xn.new_empty(B, C, dtype=torch.float32)
    return (torch.empty_like(xn), torch.empty_like(xn) if has_res else xn.new_empty(0),
            torch.empty_like(v) if has_res else v.new_empty(0), w.new_empty(C, dtype=torch.float32), torch.empty_like(v),
            torch.empty_like(v))


def _setup(ctx, inputs, output):
    x, a, gate, w, shift, scale, eps, has_res = inputs
    xn, h, rstd = output
    ctx.has_res = has_res
    ctx.dt = (gate.dtype, w.dtype, shift.dtype, scale.dtype)
    ctx.save_for_backward(xn if has_res else x, a if has_res else None, gate if has_res else None, w, scale, rstd)


def _backward(ctx, dxn_, dh, drstd):
    xn, a, gate, w, scale, rstd = ctx.saved_tensors
    has_res = ctx.has_res
    has_dxn = has_res and dxn_ is not None
    B, N, C = xn.shape
    dx, da, dg, dw, dsh, dsc = res_mod_rmsnorm_bwd(
        dh, dxn_ if has_dxn else dh, xn, a if has_res else xn, gate if has_res else w, w, scale, rstd, has_res, has_dxn)
    t_g, t_w, t_sh, t_sc = ctx.dt
    vec = lambda v, t: v.to(t).view(B, 1, C)
    return (dx, da if has_res else None, vec(dg, t_g) if has_res else None, dw.to(t_w), vec(dsh, t_sh), vec(dsc, t_sc),
            None, None)


res_mod_rmsnorm_fwd.register_autograd(_backward, setup_context=_setup)


def mod_rmsnorm(x, w, shift, scale, eps):
    """h = RMSNorm(x) * (1 + scale) + shift; x [B, N, C], w [C], shift and scale [B, 1, C]."""
    return res_mod_rmsnorm_fwd(x, x, x, w, shift, scale, eps, False)[1]


def res_mod_rmsnorm(x, a, gate, w, shift, scale, eps):
    """x_new = x + gate * a; h = RMSNorm(x_new) * (1 + scale) + shift. Returns (x_new, h)."""
    xn, h, _ = res_mod_rmsnorm_fwd(x, a, gate, w, shift, scale, eps, True)
    return xn, h


def reference(x, a, gate, w, shift, scale, eps, has_res):
    """The model's own ops in eager mode (bf16 tensors, the norm in fp32), for tests: returns (x_new or x, h)."""
    if has_res:
        x = x + gate * a
    hs = x.to(torch.float32)
    var = hs.pow(2).mean(-1, keepdim=True)
    hs = hs * torch.rsqrt(var + eps)
    n = (w * hs).to(x.dtype)
    return x, n * (1 + scale) + shift
