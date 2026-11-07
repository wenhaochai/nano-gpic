"""Native FP8 GEMMs for the image blocks (GPIC_MFU_FP8=native; owner 2026-11-07: 「FP8要做」, 「直接做b」; docs/mfu.md F8b).

torchao's Float8Linear (lever F8) swaps one nn.Linear at a time, so every GEMM pays its own passes over bf16 tensors
(amax, cast, transposed copies), and with our fused kernels K and K2 Inductor cannot fold those passes into the producers:
FP8 was 6-12% slower at d896-d1024 and only 1-2% faster at d1536-d1792. The GEMMs themselves are 1.6-1.8x faster in FP8
at our shapes (tools/mfu/fp8_gemm_bench.py, w1879). Here:
- tensorwise scaling with the current amax (no state carried across steps, so resumes and FSDP2 see nothing new): an amax
  pass with an atomic max, then one cast pass (rowwise scaling would fold the amax into the cast, but torch 2.7's
  rowwise-scaled _scaled_mm is slower than bf16 below d1280 at our shapes, w1880);
- the SwiGLU's silu(z1) * z2 is cast to FP8 by its own two kernels (its bf16 output, 4C wide, is never stored);
- the forward GEMMs run in FP8 (torch._scaled_mm, e4m3, fast accumulation, bf16 output); the backward stays in bf16
  (phase 1): dgrad from the bf16 weight, wgrad from the bf16 input (qkv_x, proj, w12: their producers write it anyway) or
  from the saved FP8 product dequantized (w3: the SwiGLU product is never stored in bf16).
Changes the numerics: the forward GEMM inputs are rounded to e4m3 (3 mantissa bits) with one scale per tensor; the master weights stay as
they are (fp32 under DDP, the FSDP2 bf16 compute copy). The SwiGLU product is kept in FP8 (half its bf16 memory).
"""
import os

import torch
import triton
import triton.language as tl
from torch.library import triton_op

from src.models.layers.fused_norm import INTERP, _launch

E4 = torch.float8_e4m3fn
FMAX = 448.0          # the largest finite e4m3fn value
TINY = 1e-12          # an all-zero tensor keeps scale FMAX / TINY: its values stay 0
BLOCK = 4096          # elements per program of the flat kernels


# Tensorwise scaling (torch 2.7's rowwise-scaled _scaled_mm is slower than bf16 below d1280 at our shapes, w1880; the
# tensorwise one is 1.5-1.8x faster than bf16): one amax per tensor, gathered with an atomic max on the float's bits
# (non-negative floats order as their int32 bit patterns), then one cast pass that also writes the dequantization scale.
@triton.jit
def _amax_kernel(X, N, AMAX, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    a = tl.max(tl.abs(tl.load(X + i, mask=i < N, other=0.0).to(tl.float32)), axis=0)
    tl.atomic_max(AMAX, a.to(tl.int32, bitcast=True))


@triton.jit
def _cast_kernel(X, Q, S, N, AMAX, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    a = tl.maximum(tl.load(AMAX).to(tl.float32, bitcast=True), 1e-12)
    scale = tl.math.div_rn(448.0, a)   # IEEE division (Triton's / is approximate): the cast is torch's to the bit
    i = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, mask=i < N, other=0.0).to(tl.float32)
    tl.store(Q + i, tl.minimum(tl.maximum(x * scale, -448.0), 448.0).to(tl.float8e4nv), mask=i < N)
    if pid == 0:
        tl.store(S, tl.math.div_rn(a, 448.0))


# The SwiGLU product of a tile, as the model computes it (silu out in bf16, the product in bf16).
@triton.jit
def _swiglu_tile(zr, H, k, m):
    z1 = tl.load(zr + k, mask=m, other=0.0).to(tl.float32)
    z2 = tl.load(zr + H + k, mask=m, other=0.0).to(tl.float32)
    sl = (z1 * tl.sigmoid(z1)).to(tl.bfloat16).to(tl.float32)
    return (sl * z2).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _swiglu_amax_kernel(Z, H, sz, AMAX, BH: tl.constexpr):
    zr = Z + tl.program_id(0).to(tl.int64) * sz
    k = tl.program_id(1) * BH + tl.arange(0, BH)
    a = _swiglu_tile(zr, H, k, k < H)
    tl.atomic_max(AMAX, tl.max(tl.abs(a), axis=0).to(tl.int32, bitcast=True))


@triton.jit
def _swiglu_cast_kernel(Z, Q, S, H, sz, sq, AMAX, BH: tl.constexpr):
    row = tl.program_id(0)
    am = tl.maximum(tl.load(AMAX).to(tl.float32, bitcast=True), 1e-12)
    scale = tl.math.div_rn(448.0, am)
    k = tl.program_id(1) * BH + tl.arange(0, BH)
    a = _swiglu_tile(Z + row.to(tl.int64) * sz, H, k, k < H)
    tl.store(Q + row.to(tl.int64) * sq + k, tl.minimum(tl.maximum(a * scale, -448.0), 448.0).to(tl.float8e4nv), mask=k < H)
    if (row == 0) & (tl.program_id(1) == 0):
        tl.store(S, tl.math.div_rn(am, 448.0))


# The SwiGLU backward of a tile (phase 2): dz = [dz1 | dz2] from z and da = d(silu(z1) * z2), written in bf16 (w12's weight
# gradient), and its amax gathered for the FP8 copy (w12's dgrad), cast by _cast_kernel afterwards.
@triton.jit
def _swiglu_bwd_kernel(Z, DA, DZ, H, sz, sda, sdz, AMAX, BH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * BH + tl.arange(0, BH)
    m = k < H
    z1 = tl.load(Z + row * sz + k, mask=m, other=0.0).to(tl.float32)
    z2 = tl.load(Z + row * sz + H + k, mask=m, other=0.0).to(tl.float32)
    da = tl.load(DA + row * sda + k, mask=m, other=0.0).to(tl.float32)
    sg = tl.sigmoid(z1)
    sl = (z1 * sg).to(tl.bfloat16).to(tl.float32)
    d1 = (da * z2 * (sg * (1.0 + z1 * (1.0 - sg)))).to(tl.bfloat16)
    d2 = (da * sl).to(tl.bfloat16)
    tl.store(DZ + row * sdz + k, d1, mask=m)
    tl.store(DZ + row * sdz + H + k, d2, mask=m)
    a = tl.maximum(tl.max(tl.abs(d1.to(tl.float32)), axis=0), tl.max(tl.abs(d2.to(tl.float32)), axis=0))
    tl.atomic_max(AMAX, a.to(tl.int32, bitcast=True))


def _bh(h):
    return min(2048, triton.next_power_of_2(h))


def _cast_flat(x, amax):
    """x contiguous, amax int32 [1] (the float's bits) -> e4m3 like x and the dequantization scale [1] fp32."""
    n = x.numel()
    q = torch.empty(x.shape, dtype=E4, device=x.device)
    s = torch.empty(1, dtype=torch.float32, device=x.device)
    _launch(_cast_kernel)[(max(triton.cdiv(n, BLOCK), 1),)](x, q, s, n, amax, BLOCK=BLOCK, num_warps=8)
    return q, s


@triton_op("gpic::fp8_tcast", mutates_args=())
def tcast(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x (any float dtype) -> e4m3 (contiguous, x's shape) and its tensorwise dequantization scale amax / 448 [1] fp32."""
    xc = x.contiguous()
    n = xc.numel()
    amax = torch.zeros(1, dtype=torch.int32, device=x.device)
    if n:
        _launch(_amax_kernel)[(triton.cdiv(n, BLOCK),)](xc, n, amax, BLOCK=BLOCK, num_warps=8)
    return _cast_flat(xc, amax)


@tcast.register_fake
def _(x):
    return x.new_empty(x.shape, dtype=E4), x.new_empty(1, dtype=torch.float32)


@triton_op("gpic::fp8_swiglu_tcast", mutates_args=())
def swiglu_tcast(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """z [M, 2H] bf16 (w12's output) -> silu(z1) * z2 as e4m3 [M, H] and its tensorwise scale [1] fp32."""
    M, H2 = z.shape
    H = H2 // 2
    zc = z if z.stride(-1) == 1 else z.contiguous()
    q = torch.empty(M, H, dtype=E4, device=z.device)
    s = torch.empty(1, dtype=torch.float32, device=z.device)
    amax = torch.zeros(1, dtype=torch.int32, device=z.device)
    if M:
        BH = _bh(H)
        grid = (M, triton.cdiv(H, BH))
        _launch(_swiglu_amax_kernel)[grid](zc, H, zc.stride(0), amax, BH=BH, num_warps=8)
        _launch(_swiglu_cast_kernel)[grid](zc, q, s, H, zc.stride(0), q.stride(0), amax, BH=BH, num_warps=8)
    return q, s


@swiglu_tcast.register_fake
def _(z):
    M, H2 = z.shape
    return z.new_empty(M, H2 // 2, dtype=E4), z.new_empty(1, dtype=torch.float32)


@triton_op("gpic::fp8_swiglu_bwd", mutates_args=())
def swiglu_bwd(z: torch.Tensor, da: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z [M, 2H] bf16, da [M, H] bf16 -> dz [M, 2H] bf16, dz as e4m3 [M, 2H] and its tensorwise scale [1] fp32."""
    M, H2 = z.shape
    H = H2 // 2
    zc, dac = z.contiguous(), da.contiguous()
    dz = torch.empty(M, H2, dtype=torch.bfloat16, device=z.device)
    amax = torch.zeros(1, dtype=torch.int32, device=z.device)
    if M:
        BH = _bh(H)
        _launch(_swiglu_bwd_kernel)[(M, triton.cdiv(H, BH))](zc, dac, dz, H, zc.stride(0), dac.stride(0), dz.stride(0),
                                                              amax, BH=BH, num_warps=8)
    dq, s = _cast_flat(dz, amax)
    return dz, dq, s


@swiglu_bwd.register_fake
def _(z, da):
    M, H2 = z.shape
    return z.new_empty(M, H2), z.new_empty(M, H2, dtype=E4), z.new_empty(1, dtype=torch.float32)


def _mm(xq, sx, wq, sw, fast=True):
    """[M, K] e4m3 x [N, K] e4m3 (each with its tensorwise scale) -> [M, N] bf16. Fast accumulation for the forward only
    (gradients accumulate in full fp32, as Transformer Engine does)."""
    return torch._scaled_mm(xq, wq.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16, use_fast_accum=fast)


class _Fp8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, sx = tcast(x2)
        wq, sw = tcast(w)
        y = _mm(xq, sx, wq, sw)
        if b is not None:
            y = y + b.to(y.dtype)
        # the bf16 input itself for the weight gradient: its producer (K, FA3) writes it anyway, so keeping it costs the
        # memory the bf16 path keeps and no dequantization pass
        ctx.save_for_backward(x2, w)
        ctx.has_b, ctx.shape, ctx.wdt = b is not None, shape, w.dtype
        return y.view(*shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2, w = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        dx = (dy2 @ w.to(torch.bfloat16)).view(ctx.shape)
        dw = (dy2.t() @ x2).to(ctx.wdt)
        db = dy2.sum(0).to(ctx.wdt) if ctx.has_b else None
        return dx, dw, db


class _SwiGLUw3(torch.autograd.Function):
    """y = w3(silu(z1) * z2), the product written as FP8 rows for the GEMM."""
    @staticmethod
    def forward(ctx, z, w3):
        shape = z.shape
        z2d = z.reshape(-1, shape[-1])
        aq, sa = swiglu_tcast(z2d)
        wq, sw = tcast(w3)
        y = _mm(aq, sa, wq, sw)
        ctx.save_for_backward(z2d, aq, sa, w3)
        ctx.shape, ctx.wdt = shape, w3.dtype
        return y.view(*shape[:-1], w3.shape[0])

    @staticmethod
    def backward(ctx, dy):
        z, aq, sa, w3 = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        da = dy2 @ w3.to(torch.bfloat16)
        dw3 = (dy2.t() @ (aq.to(torch.float32) * sa).to(torch.bfloat16)).to(ctx.wdt)
        z1, z2 = z.chunk(2, dim=-1)
        z1f = z1.float()
        sg = torch.sigmoid(z1f)
        sl = (z1f * sg).to(torch.bfloat16)
        dz2 = da * sl
        dz1 = (da.float() * z2.float() * (sg * (1 + z1f * (1 - sg)))).to(torch.bfloat16)
        return torch.cat([dz1, dz2], dim=-1).view(ctx.shape), dw3


class _Fp8MLP(torch.autograd.Function):
    """Phase 2: the whole SwiGLU MLP, y = w3(silu(z1) * z2) with z = w12(x), so the backward can hand FP8 gradients from one
    GEMM to the next: the forward GEMMs in FP8 (phase 1), and the dgrads of w3 and w12 in FP8 too: dy cast, the
    SwiGLU backward kernel writing dz in bf16 (w12's weight gradient) and gathering its amax for the FP8 copy (w12's
    dgrad). Weight gradients in bf16. The weights' FP8 copies for the dgrads are cast from their transposes."""
    @staticmethod
    def forward(ctx, x, w12, w3):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, sx = tcast(x2)
        w12q, s12 = tcast(w12)
        z = _mm(xq, sx, w12q, s12)
        aq, sa = swiglu_tcast(z)
        w3q, s3 = tcast(w3)
        y = _mm(aq, sa, w3q, s3)
        ctx.save_for_backward(x2, z, aq, sa, w12, w3)
        ctx.shape, ctx.dt = shape, (w12.dtype, w3.dtype)
        return y.view(*shape[:-1], w3.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2, z, aq, sa, w12, w3 = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        dyq, sdy = tcast(dy2)
        w3tq, s3t = tcast(w3.t().contiguous())          # [H, C]: the dgrad's B operand
        da = _mm(dyq, sdy, w3tq, s3t, fast=False)          # [M, H]
        dz, dzq, sdz = swiglu_bwd(z, da)
        w12tq, s12t = tcast(w12.t().contiguous())        # [C, 2H]
        dx = _mm(dzq, sdz, w12tq, s12t, fast=False)        # [M, C]
        dw12 = (dz.t() @ x2).to(ctx.dt[0])
        dw3 = (dy2.t() @ (aq.to(torch.float32) * sa).to(torch.bfloat16)).to(ctx.dt[1])
        return dx.view(ctx.shape), dw12, dw3


# ---------------------------------------------------------------------------------------------------------------------
# Phase 3 (GPIC_MFU_FP8=native3): phase 2 with delayed scaling for the activation casts and the weights' FP8 transposes
# made in the forward. Each activation cast site keeps the amaxes of its last L_HIST steps; this step's scale is 448 /
# their max, so the cast is one pass, and it records this step's amax. A site without history (the first step, or after
# ds_reset) is cast with its own current amax, as in phases 1 and 2 (the same arithmetic: the same values bit for bit).
# The first cast of a step (block 0's attention input) rolls every site's history first. The state lives in DS.st
# [sites, W_ST] fp32 on the device: the history, this step's amax (its int32 bits, an atomic max) and the history's max (0
# without history); each cast derives its scale from that max with its own format's largest value. The casts that read or write it are opaque custom ops: the compiler
# sees pure functions, the state is touched only when they run, and their order is the data's (each site's input
# depends on the sites before it).
L_HIST = 16
W_ST = L_HIST + 2     # history, this step's amax, the history's max


class _DSState:
    st = None      # [sites, W_ST] fp32
    sti = None     # the same memory as int32 (the atomic max on the floats' bits)
    sites = 0


DS = _DSState()


def ds_state(device):
    """(st, sti) on device, made at the first cast (opaque ops run outside the compiled graph, so this is plain Python)."""
    if DS.st is None or DS.st.device != device:
        DS.st = torch.zeros(max(DS.sites, 1), W_ST, dtype=torch.float32, device=device)
        DS.sti = DS.st.view(torch.int32)
    return DS.st, DS.sti


def ds_reset():
    """Forget every site's history (after CompileWarmup's dummy batches: their amaxes are not the data's)."""
    if DS.st is not None:
        DS.st.zero_()


@triton.jit
def _ds_roll_kernel(ST, STI, L: tl.constexpr, W: tl.constexpr):
    s = tl.program_id(0)
    cur = tl.load(ST + s * W + L)
    if cur > 0.0:
        i = tl.arange(0, L)
        h = tl.where(i == 0, cur, tl.load(ST + s * W + i - 1, mask=i > 0, other=0.0))
        tl.store(ST + s * W + i, h)
        tl.store(STI + s * W + L, 0)
        tl.store(ST + s * W + L + 1, tl.max(h, axis=0))


@triton.jit
def _ds_amax_kernel(X, N, ST, STI, site, L: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(ST + site * W + L + 1) == 0.0:   # no history: the cast needs this tensor's amax first
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        a = tl.max(tl.abs(tl.load(X + i, mask=i < N, other=0.0).to(tl.float32)), axis=0)
        tl.atomic_max(STI + site * W + L, a.to(tl.int32, bitcast=True))


@triton.jit
def _ds_cast_kernel(X, Q, S, N, ST, STI, site, L: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, mask=i < N, other=0.0).to(tl.float32)
    mx = tl.load(ST + site * W + L + 1)
    if mx > 0.0:
        scale = tl.math.div_rn(448.0, mx)
        dq = tl.math.div_rn(mx, 448.0)
        tl.atomic_max(STI + site * W + L, tl.max(tl.abs(x), axis=0).to(tl.int32, bitcast=True))
    else:
        a = tl.maximum(tl.load(ST + site * W + L), 1e-12)
        scale = tl.math.div_rn(448.0, a)
        dq = tl.math.div_rn(a, 448.0)
    tl.store(Q + i, tl.minimum(tl.maximum(x * scale, -448.0), 448.0).to(tl.float8e4nv), mask=i < N)
    if pid == 0:
        tl.store(S, dq)


@triton.jit
def _swiglu_ds_amax_kernel(Z, H, sz, ST, STI, site, L: tl.constexpr, W: tl.constexpr, BH: tl.constexpr):
    if tl.load(ST + site * W + L + 1) == 0.0:
        zr = Z + tl.program_id(0).to(tl.int64) * sz
        k = tl.program_id(1) * BH + tl.arange(0, BH)
        a = _swiglu_tile(zr, H, k, k < H)
        tl.atomic_max(STI + site * W + L, tl.max(tl.abs(a), axis=0).to(tl.int32, bitcast=True))


@triton.jit
def _swiglu_ds_cast_kernel(Z, Q, S, H, sz, sq, ST, STI, site, L: tl.constexpr, W: tl.constexpr, BH: tl.constexpr):
    row = tl.program_id(0)
    k = tl.program_id(1) * BH + tl.arange(0, BH)
    a = _swiglu_tile(Z + row.to(tl.int64) * sz, H, k, k < H)
    mx = tl.load(ST + site * W + L + 1)
    if mx > 0.0:
        scale = tl.math.div_rn(448.0, mx)
        dq = tl.math.div_rn(mx, 448.0)
        tl.atomic_max(STI + site * W + L, tl.max(tl.abs(a), axis=0).to(tl.int32, bitcast=True))
    else:
        am = tl.maximum(tl.load(ST + site * W + L), 1e-12)
        scale = tl.math.div_rn(448.0, am)
        dq = tl.math.div_rn(am, 448.0)
    tl.store(Q + row.to(tl.int64) * sq + k, tl.minimum(tl.maximum(a * scale, -448.0), 448.0).to(tl.float8e4nv), mask=k < H)
    if (row == 0) & (tl.program_id(1) == 0):
        tl.store(S, dq)


# The SwiGLU backward with delayed scaling: dz in bf16 (w12's weight gradient), its amax recorded, and with a history its
# FP8 copy written here too (w12's dgrad); without history _ds_late_cast_kernel casts dz afterwards (phase 2's way).
@triton.jit
def _swiglu_bwd_ds_kernel(Z, DA, DZ, DQ, H, sz, sda, sdz, sdq, ST, STI, site, L: tl.constexpr, W: tl.constexpr,
                          BH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * BH + tl.arange(0, BH)
    m = k < H
    z1 = tl.load(Z + row * sz + k, mask=m, other=0.0).to(tl.float32)
    z2 = tl.load(Z + row * sz + H + k, mask=m, other=0.0).to(tl.float32)
    da = tl.load(DA + row * sda + k, mask=m, other=0.0).to(tl.float32)
    sg = tl.sigmoid(z1)
    sl = (z1 * sg).to(tl.bfloat16).to(tl.float32)
    d1 = (da * z2 * (sg * (1.0 + z1 * (1.0 - sg)))).to(tl.bfloat16)
    d2 = (da * sl).to(tl.bfloat16)
    tl.store(DZ + row * sdz + k, d1, mask=m)
    tl.store(DZ + row * sdz + H + k, d2, mask=m)
    a = tl.maximum(tl.max(tl.abs(d1.to(tl.float32)), axis=0), tl.max(tl.abs(d2.to(tl.float32)), axis=0))
    tl.atomic_max(STI + site * W + L, a.to(tl.int32, bitcast=True))
    mx = tl.load(ST + site * W + L + 1)
    if mx > 0.0:
        sc = tl.math.div_rn(448.0, mx)
        tl.store(DQ + row * sdq + k, tl.minimum(tl.maximum(d1.to(tl.float32) * sc, -448.0), 448.0).to(tl.float8e4nv), mask=m)
        tl.store(DQ + row * sdq + H + k, tl.minimum(tl.maximum(d2.to(tl.float32) * sc, -448.0), 448.0).to(tl.float8e4nv),
                 mask=m)


@triton.jit
def _ds_late_cast_kernel(X, Q, S, N, ST, site, L: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    mx = tl.load(ST + site * W + L + 1)
    if mx > 0.0:   # the producer wrote the FP8 copy with the delayed scale
        if pid == 0:
            tl.store(S, tl.math.div_rn(mx, 448.0))
    else:
        a = tl.maximum(tl.load(ST + site * W + L), 1e-12)
        scale = tl.math.div_rn(448.0, a)
        i = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + i, mask=i < N, other=0.0).to(tl.float32)
        tl.store(Q + i, tl.minimum(tl.maximum(x * scale, -448.0), 448.0).to(tl.float8e4nv), mask=i < N)
        if pid == 0:
            tl.store(S, tl.math.div_rn(a, 448.0))


# A weight's FP8 copy and its transpose in one pass (current scaling, as tcast: both bit for bit tcast(w) and
# tcast(w.t().contiguous()), whose amax is the same), so the backward makes no transposed copy of the weight.
@triton.jit
def _cast_t_kernel(Wt, Q, QT, S, R, C, AMAX, BR: tl.constexpr, BC: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    a = tl.maximum(tl.load(AMAX).to(tl.float32, bitcast=True), 1e-12)
    scale = tl.math.div_rn(448.0, a)
    r = pr * BR + tl.arange(0, BR)
    c = pc * BC + tl.arange(0, BC)
    m = (r < R)[:, None] & (c < C)[None, :]
    w = tl.load(Wt + r.to(tl.int64)[:, None] * C + c[None, :], mask=m, other=0.0).to(tl.float32)
    v = tl.minimum(tl.maximum(w * scale, -448.0), 448.0)
    tl.store(Q + r.to(tl.int64)[:, None] * C + c[None, :], v.to(tl.float8e4nv), mask=m)
    tl.store(QT + c.to(tl.int64)[:, None] * R + r[None, :], tl.trans(v).to(tl.float8e4nv), mask=tl.trans(m))
    if (pr == 0) & (pc == 0):
        tl.store(S, tl.math.div_rn(a, 448.0))


@triton_op("gpic::fp8_tcast_t", mutates_args=())
def tcast_t(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """w [R, C] -> e4m3 [R, C], its tensorwise scale [1] fp32, and e4m3 [C, R] (the transpose, the same scale)."""
    wc = w.contiguous()
    R, C = wc.shape
    amax = torch.zeros(1, dtype=torch.int32, device=w.device)
    n = wc.numel()
    if n:
        _launch(_amax_kernel)[(triton.cdiv(n, BLOCK),)](wc, n, amax, BLOCK=BLOCK, num_warps=8)
    q = torch.empty(R, C, dtype=E4, device=w.device)
    qt = torch.empty(C, R, dtype=E4, device=w.device)
    s = torch.empty(1, dtype=torch.float32, device=w.device)
    _launch(_cast_t_kernel)[(max(triton.cdiv(R, 64), 1), max(triton.cdiv(C, 64), 1))](wc, q, qt, s, R, C, amax, BR=64, BC=64,
                                                                                       num_warps=4)
    return q, s, qt


@tcast_t.register_fake
def _(w):
    R, C = w.shape
    return w.new_empty(R, C, dtype=E4), w.new_empty(1, dtype=torch.float32), w.new_empty(C, R, dtype=E4)


@torch.library.custom_op("gpic::fp8_dcast", mutates_args=())
def dcast(x: torch.Tensor, site: int, roll: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """x -> e4m3 (x's shape) and the dequantization scale [1], with site's delayed scale (its own amax without history)."""
    st, sti = ds_state(x.device)
    if roll:
        _ds_roll_kernel[(st.shape[0],)](st, sti, L=L_HIST, W=W_ST, num_warps=1)
    xc = x.contiguous()
    n = xc.numel()
    q = torch.empty(x.shape, dtype=E4, device=x.device)
    s = torch.empty(1, dtype=torch.float32, device=x.device)
    if n:
        g = (triton.cdiv(n, BLOCK),)
        _ds_amax_kernel[g](xc, n, st, sti, site, L=L_HIST, W=W_ST, BLOCK=BLOCK, num_warps=8)
        _ds_cast_kernel[g](xc, q, s, n, st, sti, site, L=L_HIST, W=W_ST, BLOCK=BLOCK, num_warps=8)
    return q, s


@dcast.register_fake
def _(x, site, roll):
    return x.new_empty(x.shape, dtype=E4), x.new_empty(1, dtype=torch.float32)


@torch.library.custom_op("gpic::fp8_swiglu_dcast", mutates_args=())
def swiglu_dcast(z: torch.Tensor, site: int) -> tuple[torch.Tensor, torch.Tensor]:
    """z [M, 2H] bf16 -> silu(z1) * z2 as e4m3 [M, H] and its dequantization scale [1], with site's delayed scale."""
    st, sti = ds_state(z.device)
    M, H2 = z.shape
    H = H2 // 2
    zc = z if z.stride(-1) == 1 else z.contiguous()
    q = torch.empty(M, H, dtype=E4, device=z.device)
    s = torch.empty(1, dtype=torch.float32, device=z.device)
    if M:
        BH = _bh(H)
        grid = (M, triton.cdiv(H, BH))
        _swiglu_ds_amax_kernel[grid](zc, H, zc.stride(0), st, sti, site, L=L_HIST, W=W_ST, BH=BH, num_warps=8)
        _swiglu_ds_cast_kernel[grid](zc, q, s, H, zc.stride(0), q.stride(0), st, sti, site, L=L_HIST, W=W_ST, BH=BH,
                                     num_warps=8)
    return q, s


@swiglu_dcast.register_fake
def _(z, site):
    M, H2 = z.shape
    return z.new_empty(M, H2 // 2, dtype=E4), z.new_empty(1, dtype=torch.float32)


@torch.library.custom_op("gpic::fp8_swiglu_bwd_ds", mutates_args=())
def swiglu_bwd_ds(z: torch.Tensor, da: torch.Tensor, site: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z [M, 2H], da [M, H] bf16 -> dz [M, 2H] bf16, dz as e4m3 and its dequantization scale [1], with site's delayed scale."""
    st, sti = ds_state(z.device)
    M, H2 = z.shape
    H = H2 // 2
    zc, dac = z.contiguous(), da.contiguous()
    dz = torch.empty(M, H2, dtype=torch.bfloat16, device=z.device)
    dq = torch.empty(M, H2, dtype=E4, device=z.device)
    s = torch.empty(1, dtype=torch.float32, device=z.device)
    if M:
        BH = _bh(H)
        _swiglu_bwd_ds_kernel[(M, triton.cdiv(H, BH))](zc, dac, dz, dq, H, zc.stride(0), dac.stride(0), dz.stride(0),
                                                       dq.stride(0), st, sti, site, L=L_HIST, W=W_ST, BH=BH, num_warps=8)
        n = dz.numel()
        _ds_late_cast_kernel[(triton.cdiv(n, BLOCK),)](dz, dq, s, n, st, site, L=L_HIST, W=W_ST, BLOCK=BLOCK, num_warps=8)
    return dz, dq, s


@swiglu_bwd_ds.register_fake
def _(z, da, site):
    M, H2 = z.shape
    return z.new_empty(M, H2), z.new_empty(M, H2, dtype=E4), z.new_empty(1, dtype=torch.float32)


class _Fp8Linear3(torch.autograd.Function):
    """Phase 1's FP8 forward GEMM with the input cast by delayed scaling; the backward in bf16."""
    @staticmethod
    def forward(ctx, x, w, b, site, roll):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, sx = dcast(x2, site, roll)
        wq, sw = tcast(w)
        y = _mm(xq, sx, wq, sw)
        if b is not None:
            y = y + b.to(y.dtype)
        ctx.save_for_backward(x2, w)
        ctx.has_b, ctx.shape, ctx.wdt = b is not None, shape, w.dtype
        return y.view(*shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2, w = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        dx = (dy2 @ w.to(torch.bfloat16)).view(ctx.shape)
        dw = (dy2.t() @ x2).to(ctx.wdt)
        db = dy2.sum(0).to(ctx.wdt) if ctx.has_b else None
        return dx, dw, db, None, None


class _Fp8MLP3(torch.autograd.Function):
    """Phase 2's MLP with delayed scaling at its four activation sites (s0: x, s0 + 1: the SwiGLU product, s0 + 2: dy,
    s0 + 3: dz) and the weights' FP8 transposes from the forward."""
    @staticmethod
    def forward(ctx, x, w12, w3, s0):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, sx = dcast(x2, s0, False)
        w12q, s12, w12tq = tcast_t(w12)
        z = _mm(xq, sx, w12q, s12)
        aq, sa = swiglu_dcast(z, s0 + 1)
        w3q, s3, w3tq = tcast_t(w3)
        y = _mm(aq, sa, w3q, s3)
        ctx.save_for_backward(x2, z, aq, sa, w12tq, s12, w3tq, s3)
        ctx.shape, ctx.dt, ctx.s0 = shape, (w12.dtype, w3.dtype), s0
        return y.view(*shape[:-1], w3.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2, z, aq, sa, w12tq, s12, w3tq, s3 = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        dyq, sdy = dcast(dy2, ctx.s0 + 2, False)
        da = _mm(dyq, sdy, w3tq, s3, fast=False)          # [M, H]
        dz, dzq, sdz = swiglu_bwd_ds(z, da, ctx.s0 + 3)
        dx = _mm(dzq, sdz, w12tq, s12, fast=False)        # [M, C]
        dw12 = (dz.t() @ x2).to(ctx.dt[0])
        dw3 = (dy2.t() @ (aq.to(torch.float32) * sa).to(torch.bfloat16)).to(ctx.dt[1])
        return dx.view(ctx.shape), dw12, dw3, None


def fp8_linear3(x, w, b, site, roll):
    with torch.autocast("cuda", enabled=False):
        return _Fp8Linear3.apply(x.to(torch.bfloat16), w, b, site, roll)


def fp8_mlp3(x, w12, w3, s0):
    with torch.autocast("cuda", enabled=False):
        return _Fp8MLP3.apply(x.to(torch.bfloat16), w12, w3, s0)


class Fp8Linear3(torch.nn.Linear):
    """nn.Linear with the FP8 forward GEMM and a delayed-scaling input cast (site and roll set by convert_native)."""
    def forward(self, x):
        return fp8_linear3(x, self.weight, self.bias, self._fp8_site, self._fp8_roll)


# ---------------------------------------------------------------------------------------------------------------------
# Phase 4 (GPIC_MFU_FP8=native4): phase 3 with every backward GEMM of the image blocks in FP8: the weight gradients of
# qkv_x, proj, w12 and w3 (dW = dY^T X from both operands' transposed FP8 copies, fp32 accumulation) and the dgrads of
# qkv_x and proj. Each activation and gradient cast writes its row-major and its transposed FP8 copy in one pass (tiles,
# delayed scaling as phase 3); the SwiGLU backward writes dz's two FP8 copies and no bf16 dz. Sites per block: qkv_x x 0
# (it rolls) and dy 1, proj x 2 and dy 3, MLP x 4, product 5, dy 6, dz 7. GPIC_MFU_FP8_GRAD=e5m2 casts the gradients to
# e5m2 (range 2^-16..57344) instead of e4m3; _scaled_mm takes an e5m2 operand beside an e4m3 one.
_GRAD_E5 = os.environ.get("GPIC_MFU_FP8_GRAD", "e4m3") == "e5m2"
E5 = torch.float8_e5m2


@triton.jit
def _ds_scale(ST, STI, site, tile_amax, L: tl.constexpr, W: tl.constexpr, FMAX: tl.constexpr):
    mx = tl.load(ST + site * W + L + 1)
    if mx > 0.0:
        scale = tl.math.div_rn(FMAX, mx)
        dq = tl.math.div_rn(mx, FMAX)
        tl.atomic_max(STI + site * W + L, tile_amax.to(tl.int32, bitcast=True))
    else:
        a = tl.maximum(tl.load(ST + site * W + L), 1e-12)
        scale = tl.math.div_rn(FMAX, a)
        dq = tl.math.div_rn(a, FMAX)
    return scale, dq


@triton.jit
def _ds_cast_t_kernel(X, Q, QT, S, M, K, ST, STI, site, L: tl.constexpr, W: tl.constexpr, FMAX: tl.constexpr,
                      E5M2: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    c = pk * BK + tl.arange(0, BK)
    m = (r < M)[:, None] & (c < K)[None, :]
    x = tl.load(X + r.to(tl.int64)[:, None] * K + c[None, :], mask=m, other=0.0).to(tl.float32)
    scale, dq = _ds_scale(ST, STI, site, tl.max(tl.max(tl.abs(x), axis=1), axis=0), L, W, FMAX)
    v = tl.minimum(tl.maximum(x * scale, -FMAX), FMAX)
    if E5M2:
        tl.store(Q + r.to(tl.int64)[:, None] * K + c[None, :], v.to(tl.float8e5), mask=m)
        tl.store(QT + c.to(tl.int64)[:, None] * M + r[None, :], tl.trans(v).to(tl.float8e5), mask=tl.trans(m))
    else:
        tl.store(Q + r.to(tl.int64)[:, None] * K + c[None, :], v.to(tl.float8e4nv), mask=m)
        tl.store(QT + c.to(tl.int64)[:, None] * M + r[None, :], tl.trans(v).to(tl.float8e4nv), mask=tl.trans(m))
    if (pm == 0) & (pk == 0):
        tl.store(S, dq)


@triton.jit
def _swiglu_ds_cast_t_kernel(Z, Q, QT, S, M, H, sz, ST, STI, site, L: tl.constexpr, W: tl.constexpr, BM: tl.constexpr,
                             BH: tl.constexpr):
    pm = tl.program_id(0)
    ph = tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    k = ph * BH + tl.arange(0, BH)
    m = (r < M)[:, None] & (k < H)[None, :]
    zr = Z + r.to(tl.int64)[:, None] * sz
    z1 = tl.load(zr + k[None, :], mask=m, other=0.0).to(tl.float32)
    z2 = tl.load(zr + H + k[None, :], mask=m, other=0.0).to(tl.float32)
    a = ((z1 * tl.sigmoid(z1)).to(tl.bfloat16).to(tl.float32) * z2).to(tl.bfloat16).to(tl.float32)
    scale, dq = _ds_scale(ST, STI, site, tl.max(tl.max(tl.abs(a), axis=1), axis=0), L, W, 448.0)
    v = tl.minimum(tl.maximum(a * scale, -448.0), 448.0)
    tl.store(Q + r.to(tl.int64)[:, None] * H + k[None, :], v.to(tl.float8e4nv), mask=m)
    tl.store(QT + k.to(tl.int64)[:, None] * M + r[None, :], tl.trans(v).to(tl.float8e4nv), mask=tl.trans(m))
    if (pm == 0) & (ph == 0):
        tl.store(S, dq)


@triton.jit
def _swiglu_bwd_tile(Z, DA, r, k, m, H, sz, sda):
    zr = Z + r.to(tl.int64)[:, None] * sz
    z1 = tl.load(zr + k[None, :], mask=m, other=0.0).to(tl.float32)
    z2 = tl.load(zr + H + k[None, :], mask=m, other=0.0).to(tl.float32)
    da = tl.load(DA + r.to(tl.int64)[:, None] * sda + k[None, :], mask=m, other=0.0).to(tl.float32)
    sg = tl.sigmoid(z1)
    sl = (z1 * sg).to(tl.bfloat16).to(tl.float32)
    d1 = (da * z2 * (sg * (1.0 + z1 * (1.0 - sg)))).to(tl.bfloat16).to(tl.float32)
    d2 = (da * sl).to(tl.bfloat16).to(tl.float32)
    return d1, d2


@triton.jit
def _swiglu_bwd_ds_amax_kernel(Z, DA, M, H, sz, sda, ST, STI, site, L: tl.constexpr, W: tl.constexpr, BM: tl.constexpr,
                               BH: tl.constexpr):
    if tl.load(ST + site * W + L + 1) == 0.0:
        r = tl.program_id(0) * BM + tl.arange(0, BM)
        k = tl.program_id(1) * BH + tl.arange(0, BH)
        m = (r < M)[:, None] & (k < H)[None, :]
        d1, d2 = _swiglu_bwd_tile(Z, DA, r, k, m, H, sz, sda)
        a = tl.maximum(tl.max(tl.max(tl.abs(d1), axis=1), axis=0), tl.max(tl.max(tl.abs(d2), axis=1), axis=0))
        tl.atomic_max(STI + site * W + L, a.to(tl.int32, bitcast=True))


@triton.jit
def _swiglu_bwd_dt_kernel(Z, DA, DQ, DQT, S, M, H, sz, sda, ST, STI, site, L: tl.constexpr, W: tl.constexpr,
                          FMAX: tl.constexpr, E5M2: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr):
    pm = tl.program_id(0)
    ph = tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    k = ph * BH + tl.arange(0, BH)
    m = (r < M)[:, None] & (k < H)[None, :]
    d1, d2 = _swiglu_bwd_tile(Z, DA, r, k, m, H, sz, sda)
    a = tl.maximum(tl.max(tl.max(tl.abs(d1), axis=1), axis=0), tl.max(tl.max(tl.abs(d2), axis=1), axis=0))
    scale, dq = _ds_scale(ST, STI, site, a, L, W, FMAX)
    v1 = tl.minimum(tl.maximum(d1 * scale, -FMAX), FMAX)
    v2 = tl.minimum(tl.maximum(d2 * scale, -FMAX), FMAX)
    row = r.to(tl.int64)[:, None] * (2 * H)
    col = k.to(tl.int64)[:, None] * M
    if E5M2:
        tl.store(DQ + row + k[None, :], v1.to(tl.float8e5), mask=m)
        tl.store(DQ + row + H + k[None, :], v2.to(tl.float8e5), mask=m)
        tl.store(DQT + col + r[None, :], tl.trans(v1).to(tl.float8e5), mask=tl.trans(m))
        tl.store(DQT + H * M + col + r[None, :], tl.trans(v2).to(tl.float8e5), mask=tl.trans(m))
    else:
        tl.store(DQ + row + k[None, :], v1.to(tl.float8e4nv), mask=m)
        tl.store(DQ + row + H + k[None, :], v2.to(tl.float8e4nv), mask=m)
        tl.store(DQT + col + r[None, :], tl.trans(v1).to(tl.float8e4nv), mask=tl.trans(m))
        tl.store(DQT + H * M + col + r[None, :], tl.trans(v2).to(tl.float8e4nv), mask=tl.trans(m))
    if (pm == 0) & (ph == 0):
        tl.store(S, dq)


def _fmt(grad):
    e5 = grad and _GRAD_E5
    return (E5, 57344.0, True) if e5 else (E4, FMAX, False)


@torch.library.custom_op("gpic::fp8_dcast_t", mutates_args=())
def dcast_t(x: torch.Tensor, site: int, roll: bool, grad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, K] -> FP8 [M, K], FP8 [K, M] (its transpose) and the dequantization scale [1], with site's delayed scale;
    grad: a gradient (e5m2 under GPIC_MFU_FP8_GRAD=e5m2)."""
    st, sti = ds_state(x.device)
    if roll:
        _ds_roll_kernel[(st.shape[0],)](st, sti, L=L_HIST, W=W_ST, num_warps=1)
    dt, fmax, e5 = _fmt(grad)
    xc = x.contiguous()
    M, K = xc.shape
    q = torch.empty(M, K, dtype=dt, device=x.device)
    qt = torch.empty(K, M, dtype=dt, device=x.device)
    s = torch.empty(1, dtype=torch.float32, device=x.device)
    if xc.numel():
        _ds_amax_kernel[(triton.cdiv(xc.numel(), BLOCK),)](xc, xc.numel(), st, sti, site, L=L_HIST, W=W_ST, BLOCK=BLOCK,
                                                          num_warps=8)
        _ds_cast_t_kernel[(triton.cdiv(M, 64), triton.cdiv(K, 64))](xc, q, qt, s, M, K, st, sti, site, L=L_HIST, W=W_ST,
                                                                      FMAX=fmax, E5M2=e5, BM=64, BK=64, num_warps=4)
    return q, qt, s


@dcast_t.register_fake
def _(x, site, roll, grad):
    dt = _fmt(grad)[0]
    M, K = x.shape
    return x.new_empty(M, K, dtype=dt), x.new_empty(K, M, dtype=dt), x.new_empty(1, dtype=torch.float32)


@torch.library.custom_op("gpic::fp8_swiglu_dcast_t", mutates_args=())
def swiglu_dcast_t(z: torch.Tensor, site: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z [M, 2H] bf16 -> silu(z1) * z2 as e4m3 [M, H] and [H, M] (its transpose) and the dequantization scale [1]."""
    st, sti = ds_state(z.device)
    M, H2 = z.shape
    H = H2 // 2
    zc = z if z.stride(-1) == 1 else z.contiguous()
    q = torch.empty(M, H, dtype=E4, device=z.device)
    qt = torch.empty(H, M, dtype=E4, device=z.device)
    s = torch.empty(1, dtype=torch.float32, device=z.device)
    if M:
        BH = _bh(H)
        _swiglu_ds_amax_kernel[(M, triton.cdiv(H, BH))](zc, H, zc.stride(0), st, sti, site, L=L_HIST, W=W_ST, BH=BH,
                                                         num_warps=8)
        _swiglu_ds_cast_t_kernel[(triton.cdiv(M, 64), triton.cdiv(H, 64))](zc, q, qt, s, M, H, zc.stride(0), st, sti, site,
                                                                             L=L_HIST, W=W_ST, BM=64, BH=64, num_warps=4)
    return q, qt, s


@swiglu_dcast_t.register_fake
def _(z, site):
    M, H2 = z.shape
    return z.new_empty(M, H2 // 2, dtype=E4), z.new_empty(H2 // 2, M, dtype=E4), z.new_empty(1, dtype=torch.float32)


@torch.library.custom_op("gpic::fp8_swiglu_bwd_dt", mutates_args=())
def swiglu_bwd_dt(z: torch.Tensor, da: torch.Tensor, site: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z [M, 2H], da [M, H] bf16 -> dz as FP8 [M, 2H] and [2H, M] (its transpose) and the dequantization scale [1]; no bf16 dz."""
    st, sti = ds_state(z.device)
    dt, fmax, e5 = _fmt(True)
    M, H2 = z.shape
    H = H2 // 2
    zc, dac = z.contiguous(), da.contiguous()
    dq = torch.empty(M, H2, dtype=dt, device=z.device)
    dqt = torch.empty(H2, M, dtype=dt, device=z.device)
    s = torch.empty(1, dtype=torch.float32, device=z.device)
    if M:
        g = (triton.cdiv(M, 64), triton.cdiv(H, 64))
        _swiglu_bwd_ds_amax_kernel[g](zc, dac, M, H, zc.stride(0), dac.stride(0), st, sti, site, L=L_HIST, W=W_ST, BM=64,
                                      BH=64, num_warps=4)
        _swiglu_bwd_dt_kernel[g](zc, dac, dq, dqt, s, M, H, zc.stride(0), dac.stride(0), st, sti, site, L=L_HIST, W=W_ST,
                                 FMAX=fmax, E5M2=e5, BM=64, BH=64, num_warps=4)
    return dq, dqt, s


@swiglu_bwd_dt.register_fake
def _(z, da, site):
    M, H2 = z.shape
    dt = _fmt(True)[0]
    return z.new_empty(M, H2, dtype=dt), z.new_empty(H2, M, dtype=dt), z.new_empty(1, dtype=torch.float32)


def _mmo(aq, sa, bq, sb, out_dtype):
    """[M, K] FP8 x [N, K] FP8 (each with its tensorwise scale) -> [M, N] out_dtype, full fp32 accumulation."""
    return torch._scaled_mm(aq, bq.t(), scale_a=sa, scale_b=sb, out_dtype=out_dtype, use_fast_accum=False)


class _Fp8Linear4(torch.autograd.Function):
    """y = x W^T + b with every GEMM in FP8: the forward (fast accumulation), dX = dY W and dW = dY^T X (full accumulation);
    db = the sum of dY over the rows (bf16 dY, fp32 sum)."""
    @staticmethod
    def forward(ctx, x, w, b, site, roll):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, xqt, sx = dcast_t(x2, site, roll, False)
        wq, sw, wtq = tcast_t(w)
        y = _mm(xq, sx, wq, sw)
        if b is not None:
            y = y + b.to(y.dtype)
        ctx.save_for_backward(xqt, sx, wtq, sw)
        ctx.shape, ctx.wdt, ctx.site, ctx.has_b = shape, w.dtype, site, b is not None
        return y.view(*shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        xqt, sx, wtq, sw = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        dyq, dyqt, sdy = dcast_t(dy2, ctx.site + 1, False, True)
        dx = _mmo(dyq, sdy, wtq, sw, torch.bfloat16).view(ctx.shape)
        dw = _mmo(dyqt, sdy, xqt, sx, ctx.wdt)
        db = dy2.sum(0, dtype=torch.float32).to(ctx.wdt) if ctx.has_b else None
        return dx, dw, db, None, None


class _Fp8MLP4(torch.autograd.Function):
    """Phase 3's MLP with the weight gradients in FP8 too (sites s0: x, s0 + 1: the product, s0 + 2: dy, s0 + 3: dz)."""
    @staticmethod
    def forward(ctx, x, w12, w3, s0):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, xqt, sx = dcast_t(x2, s0, False, False)
        w12q, s12, w12tq = tcast_t(w12)
        z = _mm(xq, sx, w12q, s12)
        aq, aqt, sa = swiglu_dcast_t(z, s0 + 1)
        w3q, s3, w3tq = tcast_t(w3)
        y = _mm(aq, sa, w3q, s3)
        ctx.save_for_backward(xqt, sx, z, aqt, sa, w12tq, s12, w3tq, s3)
        ctx.shape, ctx.dt, ctx.s0 = shape, (w12.dtype, w3.dtype), s0
        return y.view(*shape[:-1], w3.shape[0])

    @staticmethod
    def backward(ctx, dy):
        xqt, sx, z, aqt, sa, w12tq, s12, w3tq, s3 = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        dyq, dyqt, sdy = dcast_t(dy2, ctx.s0 + 2, False, True)
        da = _mmo(dyq, sdy, w3tq, s3, torch.bfloat16)          # [M, H]
        dw3 = _mmo(dyqt, sdy, aqt, sa, ctx.dt[1])              # [C, H] = dY^T A
        dzq, dzqt, sdz = swiglu_bwd_dt(z, da, ctx.s0 + 3)
        dx = _mmo(dzq, sdz, w12tq, s12, torch.bfloat16)        # [M, C]
        dw12 = _mmo(dzqt, sdz, xqt, sx, ctx.dt[0])             # [2H, C] = dZ^T X
        return dx.view(ctx.shape), dw12, dw3, None


def fp8_linear4(x, w, b, site, roll):
    with torch.autocast("cuda", enabled=False):
        return _Fp8Linear4.apply(x.to(torch.bfloat16), w, b, site, roll)


def fp8_mlp4(x, w12, w3, s0):
    with torch.autocast("cuda", enabled=False):
        return _Fp8MLP4.apply(x.to(torch.bfloat16), w12, w3, s0)


class Fp8Linear4(torch.nn.Linear):
    """nn.Linear with all three GEMMs in FP8 (sites set by convert_native)."""
    def forward(self, x):
        return fp8_linear4(x, self.weight, self.bias, self._fp8_site, self._fp8_roll)


def fp8_mlp(x, w12, w3):
    with torch.autocast("cuda", enabled=False):
        return _Fp8MLP.apply(x.to(torch.bfloat16), w12, w3)


def fp8_linear(x, w, b=None):
    with torch.autocast("cuda", enabled=False):
        return _Fp8Linear.apply(x.to(torch.bfloat16), w, b)


def fp8_swiglu_w3(z, w3):
    with torch.autocast("cuda", enabled=False):
        return _SwiGLUw3.apply(z.to(torch.bfloat16), w3)


class Fp8Linear(torch.nn.Linear):
    """nn.Linear with the FP8 forward GEMM (the same parameters and state-dict keys)."""
    def forward(self, x):
        return fp8_linear(x, self.weight, self.bias)


def convert_native(net, phase=1):
    """Phase 1: the image blocks' qkv_x, proj, w12 to Fp8Linear and their SwiGLU's w3 GEMM to the fused path. Phase 2:
    qkv_x and proj as in phase 1, the MLP as one fused FP8 function with FP8 dgrads. Phase 3: phase 2 with delayed
    scaling (6 activation sites per block; block 0's qkv_x input rolls the histories). Returns the count of GEMMs."""
    n = 0
    if phase >= 4:
        site = 0
        for blk in getattr(net, "blocks", []):
            for name in ("qkv_x", "proj"):
                lin = getattr(blk.attn, name)
                if type(lin) is torch.nn.Linear:
                    lin.__class__ = Fp8Linear4
                    lin._fp8_site, lin._fp8_roll = site, site == 0
                    site += 2
                    n += 1
            mlp = blk.mlp
            if not getattr(mlp, "_fp8", False):
                mlp._fp8 = True
                mlp.forward = (lambda m, s0: (lambda x: fp8_mlp4(x, m.w12.weight, m.w3.weight, s0)))(mlp, site)
                site += 4
                n += 2
        DS.sites, DS.st = site, None
        return n
    if phase >= 3:
        site = 0
        for blk in getattr(net, "blocks", []):
            for name in ("qkv_x", "proj"):
                lin = getattr(blk.attn, name)
                if type(lin) is torch.nn.Linear:
                    lin.__class__ = Fp8Linear3
                    lin._fp8_site, lin._fp8_roll = site, site == 0
                    site += 1
                    n += 1
            mlp = blk.mlp
            if not getattr(mlp, "_fp8", False):
                mlp._fp8 = True
                mlp.forward = (lambda m, s0: (lambda x: fp8_mlp3(x, m.w12.weight, m.w3.weight, s0)))(mlp, site)
                site += 4
                n += 2
        DS.sites, DS.st = site, None
        return n
    for blk in getattr(net, "blocks", []):
        if phase >= 2:
            for name in ("qkv_x", "proj"):
                lin = getattr(blk.attn, name)
                if type(lin) is torch.nn.Linear:
                    lin.__class__ = Fp8Linear
                    n += 1
            mlp = blk.mlp
            if not getattr(mlp, "_fp8", False):
                mlp._fp8 = True
                mlp.forward = (lambda m: (lambda x: fp8_mlp(x, m.w12.weight, m.w3.weight)))(mlp)
                n += 2
            continue
        for parent, name in ((blk.attn, "qkv_x"), (blk.attn, "proj"), (blk.mlp, "w12")):
            lin = getattr(parent, name)
            if type(lin) is torch.nn.Linear:
                lin.__class__ = Fp8Linear
                n += 1
        mlp = blk.mlp
        if not getattr(mlp, "_fp8", False):
            mlp._fp8 = True
            mlp.forward = (lambda m: (lambda x: fp8_swiglu_w3(m.w12(x), m.w3.weight)))(mlp)
            n += 1
    return n


def reference(x, w, b=None):
    """The tensorwise FP8 forward in plain torch ops (tests)."""
    def q(t):
        a = t.float().abs().amax().clamp_min(TINY)
        return (t.float() * (FMAX / a)).clamp(-FMAX, FMAX).to(E4), a / FMAX
    x2 = x.reshape(-1, x.shape[-1])
    xq, sx = q(x2)
    wq, sw = q(w)
    y = (xq.float() * sx) @ (wq.float() * sw).t()
    if b is not None:
        y = y + b.float()
    return y.to(torch.bfloat16).view(*x.shape[:-1], w.shape[0])
