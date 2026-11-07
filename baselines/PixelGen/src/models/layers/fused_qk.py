"""MFU lever K2 (GPIC_MFU_FUSED_QK=1, docs/mfu.md): the image blocks' attention inputs as one Triton kernel per token
kind, forward and backward. From the projections qkv = qkv_x(x) [B, N, 3C] and kvy = kv_y(y) [B, T, 2C]:

    q = RoPE(RMSNorm_q(q_x))                        [B, N, H, Hc]
    k = [RoPE(RMSNorm_k(k_x)); RMSNorm_k(k_y)]      [B, N + T, H, Hc]   (image keys, then caption keys)
    v = [v_x; v_y]                                  [B, N + T, H, Hc]

the per-head RMSNorm over Hc (q_norm, k_norm), the 2D RoPE on the interleaved pairs (perf C4: real arithmetic) and the
concatenations of the image and caption keys and values (JiT_T2I.Attention, lever A's path). Inductor runs them as a norm
+ RoPE kernel for q, one for k with its concatenation, a concatenation for v, and in the backward the norms' row
reductions, the norm weights' column sums over all B * (N + T) * H rows, and the copies that assemble the projections'
gradients. Here the forward reads qkv and kvy once and writes q, k and v; the backward reads dq, dk, dv and the
projections once and writes their gradients directly, the norm weights' gradients as per-program partial sums added in
a fixed order (deterministic).
Math: the model's ops in fp32 with its explicit casts kept (the normalized q and k rounded to bf16 before the RoPE, as
RMSNorm's .to(input_dtype); the RoPE output rounded as .type_as), the other intermediates unrounded (as Inductor's fused
kernels), rsqrt as Inductor's (libdevice). The same math up to rounding and fp32 summation order.
"""
import os
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from src.models.layers.fused_norm import INTERP, _launch, _rsqrt, _to

ROWS = int(os.environ.get("GPIC_MFU_FUSED_QK_ROWS", "4"))   # tokens per program


# One token's heads as a [BH, Hc] tile, split into the even and odd elements of the RoPE pairs ([BH, Hc / 2] each).
@triton.jit
def _load_pairs(P, BH: tl.constexpr, HC: tl.constexpr, H, m):
    h = tl.arange(0, BH)[:, None]
    j = tl.arange(0, HC)[None, :]
    x = tl.load(P + h * HC + j, mask=(h < H) & m, other=0.0).to(tl.float32)
    return tl.split(tl.reshape(x, [BH, HC // 2, 2]))


@triton.jit
def _store_pairs(P, e, o, BH: tl.constexpr, HC: tl.constexpr, H, m, INTERP: tl.constexpr):
    h = tl.arange(0, BH)[:, None]
    j = tl.arange(0, HC)[None, :]
    x = tl.reshape(tl.join(e, o), [BH, HC])
    tl.store(P + h * HC + j, _to(x, P.dtype.element_ty, INTERP), mask=(h < H) & m)


@triton.jit
def _norm_pairs(e, o, we, wo, eps, HC: tl.constexpr, INTERP: tl.constexpr, DT: tl.constexpr):
    rstd = _rsqrt((tl.sum(e * e, axis=1) + tl.sum(o * o, axis=1)) / HC + eps, INTERP)
    ne = _to(we * (e * rstd[:, None]), DT, INTERP).to(tl.float32)
    no = _to(wo * (o * rstd[:, None]), DT, INTERP).to(tl.float32)
    return ne, no, rstd


@triton.jit
def _fwd_image(QKV, WQ, WK, COS, SIN, Q, K, V, RQ, RK, n_rows, N, T, H, C, eps,
               BH: tl.constexpr, HC: tl.constexpr, ROWS: tl.constexpr, INTERP: tl.constexpr):
    pid = tl.program_id(0)
    j2 = tl.arange(0, HC // 2)
    wqe, wqo = tl.split(tl.reshape(tl.load(WQ + tl.arange(0, HC)).to(tl.float32), [HC // 2, 2]))
    wke, wko = tl.split(tl.reshape(tl.load(WK + tl.arange(0, HC)).to(tl.float32), [HC // 2, 2]))
    hh = tl.arange(0, BH)
    for i in range(ROWS):
        row = pid * ROWS + i                      # image token b * N + n
        ok = row < n_rows
        b = row // N
        n = row % N
        c = tl.load(COS + n * (HC // 2) + j2, mask=ok, other=0.0)[None, :]
        s = tl.load(SIN + n * (HC // 2) + j2, mask=ok, other=0.0)[None, :]
        src = QKV + row * 3 * C
        kv_row = b * (N + T) + n
        # q
        e, o = _load_pairs(src, BH, HC, H, ok)
        ne, no, rq = _norm_pairs(e, o, wqe[None, :], wqo[None, :], eps, HC, INTERP, Q.dtype.element_ty)
        _store_pairs(Q + row * C, ne * c - no * s, ne * s + no * c, BH, HC, H, ok, INTERP)
        tl.store(RQ + row * H + hh, rq, mask=(hh < H) & ok)
        # k (image)
        e, o = _load_pairs(src + C, BH, HC, H, ok)
        ne, no, rk = _norm_pairs(e, o, wke[None, :], wko[None, :], eps, HC, INTERP, K.dtype.element_ty)
        _store_pairs(K + kv_row * C, ne * c - no * s, ne * s + no * c, BH, HC, H, ok, INTERP)
        tl.store(RK + kv_row * H + hh, rk, mask=(hh < H) & ok)
        # v (image): a copy
        e, o = _load_pairs(src + 2 * C, BH, HC, H, ok)
        _store_pairs(V + kv_row * C, e, o, BH, HC, H, ok, INTERP)


# The caption row's sample b and position t, and whether it is a real token. Padded layout: row = b * T + t. Packed
# layout (lever E): row s of y [S, 2C] goes to padded row PIDX[s] = b * T + t; a dummy row has PIDX[s] = B * T.
# (No docstring inside a jit function: Inductor's C++ launcher copies the source into a string, and the quotes break it.)
@triton.jit
def _text_row(PIDX, row, n_rows, T, BT, PACKED: tl.constexpr):
    inr = row < n_rows
    if PACKED:
        p = tl.load(PIDX + row, mask=inr, other=BT)
        return p // T, p % T, inr & (p < BT), inr
    else:
        return row // T, row % T, inr, inr


@triton.jit
def _fwd_text(KVY, PIDX, WK, K, V, RK, n_rows, N, T, BT, H, C, eps,
              BH: tl.constexpr, HC: tl.constexpr, ROWS: tl.constexpr, INTERP: tl.constexpr, PACKED: tl.constexpr):
    pid = tl.program_id(0)
    wke, wko = tl.split(tl.reshape(tl.load(WK + tl.arange(0, HC)).to(tl.float32), [HC // 2, 2]))
    hh = tl.arange(0, BH)
    for i in range(ROWS):
        row = pid * ROWS + i                      # caption token (see _text_row)
        b, t, ok, inr = _text_row(PIDX, row, n_rows, T, BT, PACKED)
        src = KVY + row * 2 * C
        kv_row = b * (N + T) + N + t
        e, o = _load_pairs(src, BH, HC, H, ok)
        ne, no, rk = _norm_pairs(e, o, wke[None, :], wko[None, :], eps, HC, INTERP, K.dtype.element_ty)
        _store_pairs(K + kv_row * C, ne, no, BH, HC, H, ok, INTERP)
        tl.store(RK + kv_row * H + hh, rk, mask=(hh < H) & ok)
        e, o = _load_pairs(src + C, BH, HC, H, ok)
        _store_pairs(V + kv_row * C, e, o, BH, HC, H, ok, INTERP)


# RMSNorm backward of one token's heads from the gradient (ge, go) of the normalized pairs: returns dx and adds to the
# weight's column sums
@triton.jit
def _norm_bwd(ge, go, e, o, r, we, wo, HC: tl.constexpr):
    te = e * r[:, None]
    to = o * r[:, None]
    dte = ge * we
    dto = go * wo
    mean = (tl.sum(dte * te, axis=1) + tl.sum(dto * to, axis=1)) / HC
    return r[:, None] * (dte - te * mean[:, None]), r[:, None] * (dto - to * mean[:, None]), tl.sum(ge * te, axis=0), tl.sum(go * to, axis=0)


@triton.jit
def _bwd_image(DQ, DK, DV, QKV, WQ, WK, COS, SIN, RQ, RK, DQKV, PWQ, PWK, n_rows, N, T, H, C,
               BH: tl.constexpr, HC: tl.constexpr, ROWS: tl.constexpr, INTERP: tl.constexpr):
    pid = tl.program_id(0)
    j2 = tl.arange(0, HC // 2)
    wqe, wqo = tl.split(tl.reshape(tl.load(WQ + tl.arange(0, HC)).to(tl.float32), [HC // 2, 2]))
    wke, wko = tl.split(tl.reshape(tl.load(WK + tl.arange(0, HC)).to(tl.float32), [HC // 2, 2]))
    hh = tl.arange(0, BH)
    aqe = tl.zeros([HC // 2], tl.float32)
    aqo = tl.zeros([HC // 2], tl.float32)
    ake = tl.zeros([HC // 2], tl.float32)
    ako = tl.zeros([HC // 2], tl.float32)
    for i in range(ROWS):
        row = pid * ROWS + i
        ok = row < n_rows
        b = row // N
        n = row % N
        c = tl.load(COS + n * (HC // 2) + j2, mask=ok, other=0.0)[None, :]
        s = tl.load(SIN + n * (HC // 2) + j2, mask=ok, other=0.0)[None, :]
        src = QKV + row * 3 * C
        dst = DQKV + row * 3 * C
        kv_row = b * (N + T) + n
        # q: the RoPE's transpose, then the norm's backward
        de, do = _load_pairs(DQ + row * C, BH, HC, H, ok)
        e, o = _load_pairs(src, BH, HC, H, ok)
        r = tl.load(RQ + row * H + hh, mask=(hh < H) & ok, other=0.0)
        dxe, dxo, swe, swo = _norm_bwd(de * c + do * s, do * c - de * s, e, o, r, wqe[None, :], wqo[None, :], HC)
        aqe += swe
        aqo += swo
        _store_pairs(dst, dxe, dxo, BH, HC, H, ok, INTERP)
        # k (image)
        de, do = _load_pairs(DK + kv_row * C, BH, HC, H, ok)
        e, o = _load_pairs(src + C, BH, HC, H, ok)
        r = tl.load(RK + kv_row * H + hh, mask=(hh < H) & ok, other=0.0)
        dxe, dxo, swe, swo = _norm_bwd(de * c + do * s, do * c - de * s, e, o, r, wke[None, :], wko[None, :], HC)
        ake += swe
        ako += swo
        _store_pairs(dst + C, dxe, dxo, BH, HC, H, ok, INTERP)
        # v (image)
        de, do = _load_pairs(DV + kv_row * C, BH, HC, H, ok)
        _store_pairs(dst + 2 * C, de, do, BH, HC, H, ok, INTERP)
    tl.store(PWQ + pid * HC + tl.arange(0, HC), tl.reshape(tl.join(aqe, aqo), [HC]))
    tl.store(PWK + pid * HC + tl.arange(0, HC), tl.reshape(tl.join(ake, ako), [HC]))


@triton.jit
def _bwd_text(DK, DV, KVY, PIDX, WK, RK, DKVY, PWK, n_rows, N, T, BT, H, C,
              BH: tl.constexpr, HC: tl.constexpr, ROWS: tl.constexpr, INTERP: tl.constexpr, PACKED: tl.constexpr):
    pid = tl.program_id(0)
    wke, wko = tl.split(tl.reshape(tl.load(WK + tl.arange(0, HC)).to(tl.float32), [HC // 2, 2]))
    hh = tl.arange(0, BH)
    ake = tl.zeros([HC // 2], tl.float32)
    ako = tl.zeros([HC // 2], tl.float32)
    for i in range(ROWS):
        row = pid * ROWS + i
        b, t, ok, inr = _text_row(PIDX, row, n_rows, T, BT, PACKED)
        src = KVY + row * 2 * C
        dst = DKVY + row * 2 * C
        kv_row = b * (N + T) + N + t
        # a dummy row (packed layout) loads zeros, so its gradient is written as zero and adds nothing to the sums
        de, do = _load_pairs(DK + kv_row * C, BH, HC, H, ok)
        e, o = _load_pairs(src, BH, HC, H, ok)
        r = tl.load(RK + kv_row * H + hh, mask=(hh < H) & ok, other=0.0)
        dxe, dxo, swe, swo = _norm_bwd(de, do, e, o, r, wke[None, :], wko[None, :], HC)
        ake += swe
        ako += swo
        _store_pairs(dst, dxe, dxo, BH, HC, H, inr, INTERP)
        de, do = _load_pairs(DV + kv_row * C, BH, HC, H, ok)
        _store_pairs(dst + C, de, do, BH, HC, H, inr, INTERP)
    tl.store(PWK + pid * HC + tl.arange(0, HC), tl.reshape(tl.join(ake, ako), [HC]))


def _meta(qkv, kvy, H):
    B, N, C3 = qkv.shape
    T = kvy.shape[1]
    C = C3 // 3
    HC = C // H
    assert HC * H == C and HC % 2 == 0 and triton.next_power_of_2(HC) == HC, (C, H)
    return B, N, T, C, HC, triton.next_power_of_2(H)


@triton_op("gpic::qk_prep_fwd", mutates_args=())
def qk_prep_fwd(qkv: torch.Tensor, kvy: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, H: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                               torch.Tensor]:
    B, N, T, C, HC, BH = _meta(qkv, kvy, H)
    qkvc, kvyc = qkv.contiguous(), kvy.contiguous()
    q = qkv.new_empty(B, N, H, HC)
    k = qkv.new_empty(B, N + T, H, HC)
    v = qkv.new_empty(B, N + T, H, HC)
    rq = qkv.new_empty(B, N, H, dtype=torch.float32)
    rk = qkv.new_empty(B, N + T, H, dtype=torch.float32)
    _launch(_fwd_image)[(triton.cdiv(B * N, ROWS),)](
        qkvc, wq, wk, cos.contiguous(), sin.contiguous(), q, k, v, rq, rk, B * N, N, T, H, C, eps,
        BH=BH, HC=HC, ROWS=ROWS, INTERP=INTERP, num_warps=4)
    _launch(_fwd_text)[(triton.cdiv(B * T, ROWS),)](
        kvyc, kvyc, wk, k, v, rk, B * T, N, T, B * T, H, C, eps, BH=BH, HC=HC, ROWS=ROWS, INTERP=INTERP, PACKED=False,
        num_warps=4)
    return q, k, v, rq, rk


@qk_prep_fwd.register_fake
def _(qkv, kvy, wq, wk, cos, sin, H, eps):
    B, N, C3 = qkv.shape
    T, C = kvy.shape[1], C3 // 3
    f = dict(dtype=torch.float32)
    return (qkv.new_empty(B, N, H, C // H), qkv.new_empty(B, N + T, H, C // H), qkv.new_empty(B, N + T, H, C // H),
            qkv.new_empty(B, N, H, **f), qkv.new_empty(B, N + T, H, **f))


@triton_op("gpic::qk_prep_bwd", mutates_args=())
def qk_prep_bwd(dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor, qkv: torch.Tensor, kvy: torch.Tensor,
                wq: torch.Tensor, wk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rq: torch.Tensor,
                rk: torch.Tensor, H: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns d qkv [B, N, 3C], d kvy [B, T, 2C], and the fp32 gradients of the q and k norm weights [Hc]."""
    B, N, T, C, HC, BH = _meta(qkv, kvy, H)
    dqkv = torch.empty_like(qkv, memory_format=torch.contiguous_format)
    dkvy = torch.empty_like(kvy, memory_format=torch.contiguous_format)
    pi, pt = triton.cdiv(B * N, ROWS), triton.cdiv(B * T, ROWS)
    pwq = qkv.new_empty(pi, HC, dtype=torch.float32)
    pwk = qkv.new_empty(pi + pt, HC, dtype=torch.float32)
    dkc, dvc = dk.contiguous(), dv.contiguous()
    _launch(_bwd_image)[(pi,)](
        dq.contiguous(), dkc, dvc, qkv.contiguous(), wq, wk, cos.contiguous(), sin.contiguous(), rq, rk, dqkv, pwq, pwk,
        B * N, N, T, H, C, BH=BH, HC=HC, ROWS=ROWS, INTERP=INTERP, num_warps=4)
    kvyc = kvy.contiguous()
    _launch(_bwd_text)[(pt,)](
        dkc, dvc, kvyc, kvyc, wk, rk, dkvy, pwk[pi:], B * T, N, T, B * T, H, C, BH=BH, HC=HC, ROWS=ROWS, INTERP=INTERP,
        PACKED=False, num_warps=4)
    return dqkv, dkvy, pwq.sum(0), pwk.sum(0)


@qk_prep_bwd.register_fake
def _(dq, dk, dv, qkv, kvy, wq, wk, cos, sin, rq, rk, H):
    HC = qkv.shape[-1] // 3 // H
    return (torch.empty_like(qkv, memory_format=torch.contiguous_format),
            torch.empty_like(kvy, memory_format=torch.contiguous_format), wq.new_empty(HC, dtype=torch.float32),
            wk.new_empty(HC, dtype=torch.float32))


def _setup(ctx, inputs, output):
    qkv, kvy, wq, wk, cos, sin, H, eps = inputs
    q, k, v, rq, rk = output
    ctx.H = H
    ctx.dt = (wq.dtype, wk.dtype)
    ctx.save_for_backward(qkv, kvy, wq, wk, cos, sin, rq, rk)


def _backward(ctx, dq, dk, dv, drq, drk):
    qkv, kvy, wq, wk, cos, sin, rq, rk = ctx.saved_tensors
    q_shape = (qkv.shape[0], qkv.shape[1], ctx.H, qkv.shape[2] // 3 // ctx.H)
    kv_shape = (qkv.shape[0], qkv.shape[1] + kvy.shape[1], ctx.H, q_shape[3])
    dq = torch.zeros(q_shape, dtype=qkv.dtype, device=qkv.device) if dq is None else dq
    dk = torch.zeros(kv_shape, dtype=qkv.dtype, device=qkv.device) if dk is None else dk
    dv = torch.zeros(kv_shape, dtype=qkv.dtype, device=qkv.device) if dv is None else dv
    dqkv, dkvy, dwq, dwk = qk_prep_bwd(dq, dk, dv, qkv, kvy, wq, wk, cos, sin, rq, rk, ctx.H)
    return dqkv, dkvy, dwq.to(ctx.dt[0]), dwk.to(ctx.dt[1]), None, None, None, None


qk_prep_fwd.register_autograd(_backward, setup_context=_setup)


def qk_prep(qkv, kvy, wq, wk, cos, sin, H, eps):
    """q [B, N, H, Hc], k and v [B, N + T, H, Hc] from qkv [B, N, 3C] and kvy [B, T, 2C] (see the module docstring)."""
    q, k, v, _, _ = qk_prep_fwd(qkv, kvy, wq, wk, cos, sin, H, eps)
    return q, k, v


# Lever E (packed captions): the caption keys and values come packed, kvy [S, 2C] with S rows (real tokens, then dummy
# rows), and pidx [S] int64 places row s at padded row b * Tp + t (B * Tp for a dummy row). The kernels write k and v in
# the padded key layout [B, N + Tp] that the image blocks' FA3 call reads with per-sample key counts (lens + N), so the
# per-block zero buffer and index_copy of the unfused packed path go away. Rows past a sample's caption are zeroed (one
# strided memset of the caption part, B * Tp rows): FA3 masks those keys but still loads the last tile of V, and 0 x NaN is
# NaN, so uninitialised rows turned every loss into NaN (w1865, 2026-11-07). FA3's dk and dv start at zero.
@triton_op("gpic::qk_prep_packed_fwd", mutates_args=())
def qk_prep_packed_fwd(qkv: torch.Tensor, kvy: torch.Tensor, pidx: torch.Tensor, Tp: int, wq: torch.Tensor,
                       wk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, H: int, eps: float
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, C3 = qkv.shape
    C = C3 // 3
    HC, BH, S = C // H, triton.next_power_of_2(H), kvy.shape[0]
    assert HC * H == C and HC % 2 == 0 and triton.next_power_of_2(HC) == HC, (C, H)
    qkvc, kvyc = qkv.contiguous(), kvy.contiguous()
    q = qkv.new_empty(B, N, H, HC)
    k = qkv.new_empty(B, N + Tp, H, HC)
    v = qkv.new_empty(B, N + Tp, H, HC)
    rq = qkv.new_empty(B, N, H, dtype=torch.float32)
    rk = qkv.new_empty(B, N + Tp, H, dtype=torch.float32)
    k[:, N:].zero_()
    v[:, N:].zero_()
    _launch(_fwd_image)[(triton.cdiv(B * N, ROWS),)](
        qkvc, wq, wk, cos.contiguous(), sin.contiguous(), q, k, v, rq, rk, B * N, N, Tp, H, C, eps,
        BH=BH, HC=HC, ROWS=ROWS, INTERP=INTERP, num_warps=4)
    _launch(_fwd_text)[(triton.cdiv(S, ROWS),)](
        kvyc, pidx.contiguous(), wk, k, v, rk, S, N, Tp, B * Tp, H, C, eps, BH=BH, HC=HC, ROWS=ROWS, INTERP=INTERP,
        PACKED=True, num_warps=4)
    return q, k, v, rq, rk


@qk_prep_packed_fwd.register_fake
def _(qkv, kvy, pidx, Tp, wq, wk, cos, sin, H, eps):
    B, N, C3 = qkv.shape
    C = C3 // 3
    f = dict(dtype=torch.float32)
    return (qkv.new_empty(B, N, H, C // H), qkv.new_empty(B, N + Tp, H, C // H), qkv.new_empty(B, N + Tp, H, C // H),
            qkv.new_empty(B, N, H, **f), qkv.new_empty(B, N + Tp, H, **f))


@triton_op("gpic::qk_prep_packed_bwd", mutates_args=())
def qk_prep_packed_bwd(dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor, qkv: torch.Tensor, kvy: torch.Tensor,
                       pidx: torch.Tensor, Tp: int, wq: torch.Tensor, wk: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor, rq: torch.Tensor, rk: torch.Tensor, H: int
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns d qkv [B, N, 3C], d kvy [S, 2C] (zero for dummy rows), and the fp32 gradients of the norm weights."""
    B, N, C3 = qkv.shape
    C = C3 // 3
    HC, BH, S = C // H, triton.next_power_of_2(H), kvy.shape[0]
    dqkv = torch.empty_like(qkv, memory_format=torch.contiguous_format)
    dkvy = torch.empty_like(kvy, memory_format=torch.contiguous_format)
    pi, pt = triton.cdiv(B * N, ROWS), triton.cdiv(S, ROWS)
    pwq = qkv.new_empty(pi, HC, dtype=torch.float32)
    pwk = qkv.new_empty(pi + pt, HC, dtype=torch.float32)
    dkc, dvc, kvyc = dk.contiguous(), dv.contiguous(), kvy.contiguous()
    _launch(_bwd_image)[(pi,)](
        dq.contiguous(), dkc, dvc, qkv.contiguous(), wq, wk, cos.contiguous(), sin.contiguous(), rq, rk, dqkv, pwq, pwk,
        B * N, N, Tp, H, C, BH=BH, HC=HC, ROWS=ROWS, INTERP=INTERP, num_warps=4)
    _launch(_bwd_text)[(pt,)](
        dkc, dvc, kvyc, pidx.contiguous(), wk, rk, dkvy, pwk[pi:], S, N, Tp, B * Tp, H, C, BH=BH, HC=HC, ROWS=ROWS,
        INTERP=INTERP, PACKED=True, num_warps=4)
    return dqkv, dkvy, pwq.sum(0), pwk.sum(0)


@qk_prep_packed_bwd.register_fake
def _(dq, dk, dv, qkv, kvy, pidx, Tp, wq, wk, cos, sin, rq, rk, H):
    HC = qkv.shape[-1] // 3 // H
    return (torch.empty_like(qkv, memory_format=torch.contiguous_format),
            torch.empty_like(kvy, memory_format=torch.contiguous_format), wq.new_empty(HC, dtype=torch.float32),
            wk.new_empty(HC, dtype=torch.float32))


def _setup_packed(ctx, inputs, output):
    qkv, kvy, pidx, Tp, wq, wk, cos, sin, H, eps = inputs
    q, k, v, rq, rk = output
    ctx.H, ctx.Tp = H, Tp
    ctx.dt = (wq.dtype, wk.dtype)
    ctx.save_for_backward(qkv, kvy, pidx, wq, wk, cos, sin, rq, rk)


def _backward_packed(ctx, dq, dk, dv, drq, drk):
    qkv, kvy, pidx, wq, wk, cos, sin, rq, rk = ctx.saved_tensors
    B, N, C3 = qkv.shape
    q_shape = (B, N, ctx.H, C3 // 3 // ctx.H)
    kv_shape = (B, N + ctx.Tp, ctx.H, q_shape[3])
    dq = torch.zeros(q_shape, dtype=qkv.dtype, device=qkv.device) if dq is None else dq
    dk = torch.zeros(kv_shape, dtype=qkv.dtype, device=qkv.device) if dk is None else dk
    dv = torch.zeros(kv_shape, dtype=qkv.dtype, device=qkv.device) if dv is None else dv
    dqkv, dkvy, dwq, dwk = qk_prep_packed_bwd(dq, dk, dv, qkv, kvy, pidx, ctx.Tp, wq, wk, cos, sin, rq, rk, ctx.H)
    return dqkv, dkvy, None, None, dwq.to(ctx.dt[0]), dwk.to(ctx.dt[1]), None, None, None, None


qk_prep_packed_fwd.register_autograd(_backward_packed, setup_context=_setup_packed)


def qk_prep_packed(qkv, kvy, pidx, Tp, wq, wk, cos, sin, H, eps):
    """q [B, N, H, Hc], k and v [B, N + Tp, H, Hc] from qkv [B, N, 3C] and packed kvy [S, 2C] placed by pidx [S]."""
    q, k, v, _, _ = qk_prep_packed_fwd(qkv, kvy, pidx, Tp, wq, wk, cos, sin, H, eps)
    return q, k, v


def reference(qkv, kvy, wq, wk, cos, sin, H, eps):
    """The model's own ops (lever A's path in JiT_T2I.Attention), for tests."""
    B, N, C3 = qkv.shape
    C = C3 // 3

    def norm(x, w):
        hs = x.to(torch.float32)
        hs = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + eps)
        return (w * hs).to(x.dtype)

    def rope(x):
        c, s = cos[None, :, None], sin[None, :, None]
        xf = x.float().reshape(*x.shape[:-1], -1, 2)
        x0, x1 = xf[..., 0], xf[..., 1]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(3).type_as(x)

    q, kx, vx = qkv.view(B, N, 3, H, C // H).unbind(2)
    q, kx = rope(norm(q, wq)), rope(norm(kx, wk))
    ky, vy = kvy.view(B, -1, 2, H, C // H).unbind(2)
    return q, torch.cat([kx, norm(ky, wk)], dim=1), torch.cat([vx, vy], dim=1)
