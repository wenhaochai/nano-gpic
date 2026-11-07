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
    qkv_x and proj as in phase 1, the MLP as one fused FP8 function with FP8 dgrads. Returns the count of GEMMs."""
    n = 0
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
