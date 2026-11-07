"""Native FP8 GEMMs for the image blocks (GPIC_MFU_FP8=native; owner 2026-11-07: 「FP8要做」, 「直接做b」; docs/mfu.md F8b).

torchao's Float8Linear (lever F8) swaps one nn.Linear at a time, so every GEMM pays its own passes over bf16 tensors
(amax, cast, transposed copies), and with our fused kernels K and K2 Inductor cannot fold those passes into the producers:
FP8 was 6-12% slower at d896-d1024 and only 1-2% faster at d1536-d1792. The GEMMs themselves are 1.6-1.8x faster in FP8
at our shapes (tools/mfu/fp8_gemm_bench.py, w1879). Here:
- rowwise scaling: each activation row gets its own scale, computed by the kernel that writes the row (one pass, no global
  amax, no state carried across steps, so resumes and FSDP2 see nothing new); weights get one scale per output channel;
- the SwiGLU's silu(z1) * z2 is written straight as FP8 rows for w3 (its bf16 output, 4C wide, is never stored);
- the forward GEMMs run in FP8 (torch._scaled_mm, e4m3, fast accumulation, bf16 output); the backward stays in bf16
  (phase 1): dgrad from the bf16 weight, wgrad from the bf16 input (qkv_x, proj, w12: their producers write it anyway) or
  from the saved FP8 product dequantized (w3: the SwiGLU product is never stored in bf16).
Changes the numerics: the forward GEMM inputs are rounded to e4m3 (3 mantissa bits) per row; the master weights stay as
they are (fp32 under DDP, the FSDP2 bf16 compute copy). The SwiGLU product is kept in FP8 (half its bf16 memory).
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op

from src.models.layers.fused_norm import INTERP, _launch

E4 = torch.float8_e4m3fn
FMAX = 448.0          # the largest finite e4m3fn value
TINY = 1e-12          # a row of zeros keeps scale 1 / TINY: its values stay 0


# One row: its amax over K, then the scaled, clamped e4m3 values and the row's dequantization scale amax / FMAX.
@triton.jit
def _rowcast_kernel(X, Q, S, K, sx, sq, BK: tl.constexpr):
    row = tl.program_id(0)
    xr = X + row.to(tl.int64) * sx
    qr = Q + row.to(tl.int64) * sq
    amax = tl.zeros([BK], tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        amax = tl.maximum(amax, tl.abs(tl.load(xr + k, mask=k < K, other=0.0).to(tl.float32)))
    a = tl.maximum(tl.max(amax, axis=0), 1e-12)
    scale = 448.0 / a
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(xr + k, mask=k < K, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(x * scale, -448.0), 448.0)
        tl.store(qr + k, q.to(tl.float8e4nv), mask=k < K)
    tl.store(S + row, a / 448.0)


# One row of the SwiGLU: z = [z1 | z2] (2H), a = silu(z1) * z2 (H), written as scaled e4m3 with the row's scale.
@triton.jit
def _swiglu_rowcast_kernel(Z, Q, S, H, sz, sq, BK: tl.constexpr):
    row = tl.program_id(0)
    zr = Z + row.to(tl.int64) * sz
    qr = Q + row.to(tl.int64) * sq
    amax = tl.zeros([BK], tl.float32)
    for k0 in range(0, H, BK):
        k = k0 + tl.arange(0, BK)
        z1 = tl.load(zr + k, mask=k < H, other=0.0).to(tl.float32)
        z2 = tl.load(zr + H + k, mask=k < H, other=0.0).to(tl.float32)
        # silu as the model computes it (bf16 out of silu, then the product in bf16), then fp32 for the cast
        sl = (z1 * tl.sigmoid(z1)).to(tl.bfloat16).to(tl.float32)
        a = (sl * z2).to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(a))
    m = tl.maximum(tl.max(amax, axis=0), 1e-12)
    scale = 448.0 / m
    for k0 in range(0, H, BK):
        k = k0 + tl.arange(0, BK)
        z1 = tl.load(zr + k, mask=k < H, other=0.0).to(tl.float32)
        z2 = tl.load(zr + H + k, mask=k < H, other=0.0).to(tl.float32)
        sl = (z1 * tl.sigmoid(z1)).to(tl.bfloat16).to(tl.float32)
        a = (sl * z2).to(tl.bfloat16).to(tl.float32)
        q = tl.minimum(tl.maximum(a * scale, -448.0), 448.0)
        tl.store(qr + k, q.to(tl.float8e4nv), mask=k < H)
    tl.store(S + row, m / 448.0)


# The SwiGLU backward of one row (phase 2): from z = [z1 | z2] and da (the gradient of silu(z1) * z2), dz = [dz1 | dz2] in
# bf16 (for w12's weight gradient) and as scaled e4m3 rows with the row's scale (for w12's FP8 dgrad). The first loop writes
# bf16 dz and finds the row amax, the second recomputes dz from the inputs (no read of what the first wrote) and casts it.
@triton.jit
def _swiglu_bwd_kernel(Z, DA, DZ, DZQ, S, H, sz, sda, sdz, sdq, BK: tl.constexpr):
    row = tl.program_id(0)
    zr = Z + row.to(tl.int64) * sz
    dar = DA + row.to(tl.int64) * sda
    dzr = DZ + row.to(tl.int64) * sdz
    dqr = DZQ + row.to(tl.int64) * sdq
    amax = tl.zeros([BK], tl.float32)
    for k0 in range(0, H, BK):
        k = k0 + tl.arange(0, BK)
        m = k < H
        z1 = tl.load(zr + k, mask=m, other=0.0).to(tl.float32)
        z2 = tl.load(zr + H + k, mask=m, other=0.0).to(tl.float32)
        da = tl.load(dar + k, mask=m, other=0.0).to(tl.float32)
        sg = tl.sigmoid(z1)
        sl = (z1 * sg).to(tl.bfloat16).to(tl.float32)
        d1 = (da * z2 * (sg * (1.0 + z1 * (1.0 - sg)))).to(tl.bfloat16)
        d2 = (da * sl).to(tl.bfloat16)
        tl.store(dzr + k, d1, mask=m)
        tl.store(dzr + H + k, d2, mask=m)
        amax = tl.maximum(amax, tl.maximum(tl.abs(d1.to(tl.float32)), tl.abs(d2.to(tl.float32))))
    a = tl.maximum(tl.max(amax, axis=0), 1e-12)
    scale = 448.0 / a
    for k0 in range(0, H, BK):
        k = k0 + tl.arange(0, BK)
        m = k < H
        z1 = tl.load(zr + k, mask=m, other=0.0).to(tl.float32)
        z2 = tl.load(zr + H + k, mask=m, other=0.0).to(tl.float32)
        da = tl.load(dar + k, mask=m, other=0.0).to(tl.float32)
        sg = tl.sigmoid(z1)
        sl = (z1 * sg).to(tl.bfloat16).to(tl.float32)
        d1 = (da * z2 * (sg * (1.0 + z1 * (1.0 - sg)))).to(tl.bfloat16).to(tl.float32)
        d2 = (da * sl).to(tl.bfloat16).to(tl.float32)
        tl.store(dqr + k, tl.minimum(tl.maximum(d1 * scale, -448.0), 448.0).to(tl.float8e4nv), mask=m)
        tl.store(dqr + H + k, tl.minimum(tl.maximum(d2 * scale, -448.0), 448.0).to(tl.float8e4nv), mask=m)
    tl.store(S + row, a / 448.0)


def _bk(k):
    return min(4096, triton.next_power_of_2(k))


@triton_op("gpic::fp8_rowcast", mutates_args=())
def rowcast(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] (any float dtype, last dim contiguous) -> e4m3 [M, K] and the per-row dequantization scale [M, 1] fp32."""
    M, K = x.shape
    xc = x if x.stride(-1) == 1 else x.contiguous()
    q = torch.empty(M, K, dtype=E4, device=x.device)
    s = torch.empty(M, 1, dtype=torch.float32, device=x.device)
    if M:
        _launch(_rowcast_kernel)[(M,)](xc, q, s, K, xc.stride(0), q.stride(0), BK=_bk(K), num_warps=8)
    return q, s


@rowcast.register_fake
def _(x):
    M, K = x.shape
    return x.new_empty(M, K, dtype=E4), x.new_empty(M, 1, dtype=torch.float32)


@triton_op("gpic::fp8_swiglu_rowcast", mutates_args=())
def swiglu_rowcast(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """z [M, 2H] bf16 (w12's output) -> silu(z1) * z2 as e4m3 [M, H] and its per-row scale [M, 1] fp32."""
    M, H2 = z.shape
    H = H2 // 2
    zc = z if z.stride(-1) == 1 else z.contiguous()
    q = torch.empty(M, H, dtype=E4, device=z.device)
    s = torch.empty(M, 1, dtype=torch.float32, device=z.device)
    if M:
        _launch(_swiglu_rowcast_kernel)[(M,)](zc, q, s, H, zc.stride(0), q.stride(0), BK=_bk(H), num_warps=8)
    return q, s


@swiglu_rowcast.register_fake
def _(z):
    M, H2 = z.shape
    return z.new_empty(M, H2 // 2, dtype=E4), z.new_empty(M, 1, dtype=torch.float32)


@triton_op("gpic::fp8_swiglu_bwd", mutates_args=())
def swiglu_bwd(z: torch.Tensor, da: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z [M, 2H] bf16, da [M, H] bf16 -> dz [M, 2H] bf16, dz as e4m3 rows [M, 2H] and their scales [M, 1] fp32."""
    M, H2 = z.shape
    zc, dac = z.contiguous(), da.contiguous()
    dz = torch.empty(M, H2, dtype=torch.bfloat16, device=z.device)
    dq = torch.empty(M, H2, dtype=E4, device=z.device)
    s = torch.empty(M, 1, dtype=torch.float32, device=z.device)
    if M:
        _launch(_swiglu_bwd_kernel)[(M,)](zc, dac, dz, dq, s, H2 // 2, zc.stride(0), dac.stride(0), dz.stride(0),
                                          dq.stride(0), BK=_bk(H2 // 2), num_warps=8)
    return dz, dq, s


@swiglu_bwd.register_fake
def _(z, da):
    M, H2 = z.shape
    return z.new_empty(M, H2), z.new_empty(M, H2, dtype=E4), z.new_empty(M, 1, dtype=torch.float32)


def _mm(xq, sx, wq, sw, fast=True):
    """[M, K] e4m3 with row scales x [N, K] e4m3 with row (= output channel) scales -> [M, N] bf16. Fast accumulation
    for the forward only (gradients accumulate in full fp32, as Transformer Engine does)."""
    return torch._scaled_mm(xq, wq.t(), scale_a=sx, scale_b=sw.view(1, -1), out_dtype=torch.bfloat16,
                            use_fast_accum=fast)


class _Fp8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, sx = rowcast(x2)
        wq, sw = rowcast(w)
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
        aq, sa = swiglu_rowcast(z2d)
        wq, sw = rowcast(w3)
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
    GEMM to the next: the forward GEMMs in FP8 (phase 1), and the dgrads of w3 and w12 in FP8 too: dy cast in rows, the
    SwiGLU backward kernel writing dz both in bf16 (w12's weight gradient) and in FP8 rows (w12's dgrad). Weight
    gradients in bf16. The weights' FP8 copies for the dgrads are taken from their transposes, one scale per input
    channel (the dgrad's output column)."""
    @staticmethod
    def forward(ctx, x, w12, w3):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, sx = rowcast(x2)
        w12q, s12 = rowcast(w12)
        z = _mm(xq, sx, w12q, s12)
        aq, sa = swiglu_rowcast(z)
        w3q, s3 = rowcast(w3)
        y = _mm(aq, sa, w3q, s3)
        ctx.save_for_backward(x2, z, aq, sa, w12, w3)
        ctx.shape, ctx.dt = shape, (w12.dtype, w3.dtype)
        return y.view(*shape[:-1], w3.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2, z, aq, sa, w12, w3 = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        dyq, sdy = rowcast(dy2)
        w3tq, s3t = rowcast(w3.t().contiguous())          # [H, C]: one scale per hidden channel
        da = _mm(dyq, sdy, w3tq, s3t, fast=False)          # [M, H]
        dz, dzq, sdz = swiglu_bwd(z, da)
        w12tq, s12t = rowcast(w12.t().contiguous())        # [C, 2H]: one scale per input channel
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
    """The rowwise FP8 forward in plain torch ops (tests)."""
    def q(t):
        a = t.float().abs().amax(dim=1, keepdim=True).clamp_min(TINY)
        return (t.float() * (FMAX / a)).clamp(-FMAX, FMAX).to(E4), a / FMAX
    x2 = x.reshape(-1, x.shape[-1])
    xq, sx = q(x2)
    wq, sw = q(w)
    y = (xq.float() * sx) @ (wq.float() * sw).t()
    if b is not None:
        y = y + b.float()
    return y.to(torch.bfloat16).view(*x.shape[:-1], w.shape[0])
