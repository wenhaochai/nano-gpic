"""MoE operators in Triton (owner 2026-11-13: 「Infra上也要努力的 moe算子」; branch moe), switched on with GPIC_MOE_KERNELS=1.

The layer's slot table maps every slot s (an expert's row, s = e * cap + r) to a token flat[s]. Two sums run over a token's slots:
the combine (out[t] = sum of gate[s] * o[s] over the slots of token t) and the backward of the gather (grad_x[t] = sum of grad_xe[s]).
The torch version (moe.py) adds them with fp32 atomics into an fp32 [tokens, d] buffer, then casts it to bf16: a zero fill, an atomic
read-modify-write per slot row, and a cast pass, and the order of a token's terms changes from run to run. Here both sums are one
segmented sum over the slots sorted by token (CSR: perm = slots in token order, offsets = each token's range): one program per
(token, column block) reads its slot rows in a fixed order, accumulates in fp32 and writes the bf16 row once. Same values up to the
order of a token's fp32 terms, and the same on every run.

Ops (torch.library custom ops, opaque to Inductor, so a compiled block calls them without a graph break):
  gpic::moe_csr(flat, tokens) -> perm, offsets                    the slots in token order (stable), offsets [tokens + 1]
  gpic::moe_segsum(src, perm, offsets, tokens) -> out             out[t] = sum of src[s] over t's slots (the gather's backward)
  gpic::moe_gather(x, flat, perm, offsets) -> xe                  xe[s] = x[flat[s]]; backward: the segmented sum of grad_xe
  gpic::moe_combine(o, gate, flat, perm, offsets) -> out          out[t] = sum gate[s] o[s]; backward: grad_o[s] = gate[s] g[flat[s]],
                                                                   grad_gate[s] = <o[s], g[flat[s]]> (fp32), one kernel
tools/moe/tests/test_moe_ops.py checks them against the torch path."""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _segsum_kernel(src_ptr, w_ptr, perm_ptr, off_ptr, out_ptr, D, s_src, s_out, HAS_W: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D
    start = tl.load(off_ptr + t)
    end = tl.load(off_ptr + t + 1)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in range(start, end):
        s = tl.load(perm_ptr + k).to(tl.int64)
        v = tl.load(src_ptr + s * s_src + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_W:
            v = v * tl.load(w_ptr + s)
        acc += v
    tl.store(out_ptr + t.to(tl.int64) * s_out + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _combine_bwd_kernel(g_ptr, o_ptr, w_ptr, flat_ptr, go_ptr, gw_ptr, D, s_g, s_o, s_go, HAS_W: tl.constexpr, BLOCK_D: tl.constexpr):
    s = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    t = tl.load(flat_ptr + s).to(tl.int64)
    g = tl.load(g_ptr + t * s_g + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_W:
        w = tl.load(w_ptr + s)
        o = tl.load(o_ptr + s * s_o + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(gw_ptr + s, tl.sum(o * g, axis=0))
        g = g * w
    tl.store(go_ptr + s * s_go + cols, g.to(go_ptr.dtype.element_ty), mask=mask)


def _segsum(src, w, perm, offsets, tokens):
    S, D = src.shape
    out = torch.empty(tokens, D, dtype=src.dtype, device=src.device)
    BD = min(1024, triton.next_power_of_2(D))
    _segsum_kernel[(tokens, triton.cdiv(D, BD))](src, w if w is not None else src, perm, offsets, out, D, src.stride(0), out.stride(0),
                                                 HAS_W=w is not None, BLOCK_D=BD, num_warps=4)
    return out


# ---- the segmented sum as an op of its own: the gather's backward calls it, and a backward that launched the kernel directly would be
# traced by torch.compile (AOT autograd) with fake tensors ----
@torch.library.custom_op("gpic::moe_segsum", mutates_args=())
def moe_segsum(src: torch.Tensor, perm: torch.Tensor, offsets: torch.Tensor, tokens: int) -> torch.Tensor:
    return _segsum(src.contiguous(), None, perm, offsets, tokens)


@moe_segsum.register_fake
def _(src, perm, offsets, tokens):
    return src.new_empty(tokens, src.shape[1])


# ---- the CSR of a slot table ----
@torch.library.custom_op("gpic::moe_csr", mutates_args=())
def moe_csr(flat: torch.Tensor, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.sort(flat, stable=True).indices
    offsets = torch.searchsorted(flat.index_select(0, perm), torch.arange(tokens + 1, device=flat.device, dtype=flat.dtype))
    return perm, offsets


@moe_csr.register_fake
def _(flat, tokens):
    return flat.new_empty(flat.shape[0]), flat.new_empty(tokens + 1)


# ---- gather: xe[s] = x[flat[s]] ----
@torch.library.custom_op("gpic::moe_gather", mutates_args=())
def moe_gather(x: torch.Tensor, flat: torch.Tensor, perm: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, flat)


@moe_gather.register_fake
def _(x, flat, perm, offsets):
    return x.new_empty(flat.shape[0], x.shape[1])


def _gather_setup(ctx, inputs, output):
    x, flat, perm, offsets = inputs
    ctx.save_for_backward(perm, offsets)
    ctx.tokens = x.shape[0]


def _gather_bwd(ctx, g):
    perm, offsets = ctx.saved_tensors
    return torch.ops.gpic.moe_segsum(g, perm, offsets, ctx.tokens), None, None, None


moe_gather.register_autograd(_gather_bwd, setup_context=_gather_setup)


# ---- combine: out[t] = sum over t's slots of gate[s] o[s] ----
@torch.library.custom_op("gpic::moe_combine", mutates_args=())
def moe_combine(o: torch.Tensor, gate: torch.Tensor, flat: torch.Tensor, perm: torch.Tensor, offsets: torch.Tensor, tokens: int,
                has_gate: bool) -> torch.Tensor:
    return _segsum(o.contiguous(), gate.float().contiguous() if has_gate else None, perm, offsets, tokens)


@moe_combine.register_fake
def _(o, gate, flat, perm, offsets, tokens, has_gate):
    return o.new_empty(tokens, o.shape[1])


@torch.library.custom_op("gpic::moe_combine_bwd", mutates_args=())
def moe_combine_bwd(g: torch.Tensor, o: torch.Tensor, gate: torch.Tensor, flat: torch.Tensor, has_gate: bool) -> tuple[torch.Tensor, torch.Tensor]:
    S, D = o.shape
    g = g.contiguous()
    go = torch.empty_like(o)
    gw = torch.zeros(S, dtype=torch.float32, device=o.device)
    BD = triton.next_power_of_2(D)
    _combine_bwd_kernel[(S,)](g, o, gate.float().contiguous() if has_gate else o, flat, go, gw, D, g.stride(0), o.stride(0), go.stride(0),
                              HAS_W=has_gate, BLOCK_D=BD, num_warps=8 if BD >= 2048 else 4)
    return go, gw


@moe_combine_bwd.register_fake
def _(g, o, gate, flat, has_gate):
    return torch.empty_like(o), o.new_empty(o.shape[0], dtype=torch.float32)


def _combine_setup(ctx, inputs, output):
    o, gate, flat, perm, offsets, tokens, has_gate = inputs
    ctx.save_for_backward(o, gate, flat)
    ctx.has_gate = has_gate


def _combine_bwd(ctx, g):
    o, gate, flat = ctx.saved_tensors
    go, gw = moe_combine_bwd(g, o, gate, flat, ctx.has_gate)
    return go, (gw.to(gate.dtype) if ctx.has_gate else None), None, None, None, None, None


moe_combine.register_autograd(_combine_bwd, setup_context=_combine_setup)


# ==== v2 (GPIC_MOE_KERNELS=2; 2026-11-14, after the step profile w2686) ====
# The step profile put about 1.16 ms per layer (d12, 16 experts, 2 active) into the torch path's glue: two fp32 index_add (atomics into a
# zero-filled fp32 [tokens, d] buffer), the casts around them and the zero fills; v1's sorted segmented sums did not beat it (the sort per
# layer, one token per program). v2 sums in token order through the inverse slot map instead: inv [tokens, n] int32, inv[t, e] = the row r
# of token t among expert e's slots (slot s = e * cap + r), -1 where expert e did not take t. A sum over a token's slots reads its n entries
# in expert order, loads only the valid rows, adds them in fp32 registers and writes the output dtype once: no sort, no atomics, no fp32
# buffer, and the same order on every run. Empty slots (gate 0) are left out: their terms are 0 in the combine and their gradient rows are 0
# in the gather's backward, so the sums are the same.
#   gpic::moe_inv(flat, valid, tokens, n) -> inv                          valid [S] bool (False: an empty slot)
#   gpic::moe_gather2(x, flat, inv) -> xe                                 xe[s] = x[flat[s]]; backward: grad_x[t] = sum over t's slots
#   gpic::moe_combine2(o, gate, flat, inv, tokens, has_gate) -> out       out[t] = sum gate[s] o[s]; backward as v1 (one kernel per slot)

@triton.jit
def _tsum_kernel(src_ptr, w_ptr, inv_ptr, out_ptr, M, D, cap, s_src, s_out,
                 N_EXP: tl.constexpr, HAS_W: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    tmask = t < M
    cmask = cols < D
    t64 = t.to(tl.int64)
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    for e in range(N_EXP):
        r = tl.load(inv_ptr + t64 * N_EXP + e, mask=tmask, other=-1)
        valid = r >= 0
        row = (e * cap + r).to(tl.int64)
        v = tl.load(src_ptr + row[:, None] * s_src + cols[None, :], mask=valid[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        if HAS_W:
            v = v * tl.load(w_ptr + row, mask=valid, other=0.0)[:, None]
        acc += v
    tl.store(out_ptr + t64[:, None] * s_out + cols[None, :], acc.to(out_ptr.dtype.element_ty), mask=tmask[:, None] & cmask[None, :])


def _tsum(src, w, inv, tokens, n):
    S, D = src.shape
    cap = S // n
    out = torch.empty(tokens, D, dtype=src.dtype, device=src.device)
    BT, BD = 16, min(128, triton.next_power_of_2(D))
    _tsum_kernel[(triton.cdiv(tokens, BT), triton.cdiv(D, BD))](src, w if w is not None else src, inv, out, tokens, D, cap,
                                                              src.stride(0), out.stride(0), N_EXP=n, HAS_W=w is not None,
                                                              BLOCK_T=BT, BLOCK_D=BD, num_warps=4)
    return out


@torch.library.custom_op("gpic::moe_inv", mutates_args=())
def moe_inv(flat: torch.Tensor, valid: torch.Tensor, tokens: int, n: int) -> torch.Tensor:
    S = flat.shape[0]
    cap = S // n
    s = torch.arange(S, device=flat.device)
    # every slot writes once to its own place: a valid slot to (token, expert), an empty one to a spare cell past the table
    dest = torch.where(valid, flat.to(torch.int64) * n + s // cap, tokens * n + s)
    inv = torch.full((tokens * n + S,), -1, dtype=torch.int32, device=flat.device)
    inv.scatter_(0, dest, (s % cap).to(torch.int32))
    return inv[:tokens * n].view(tokens, n).clone()


@moe_inv.register_fake
def _(flat, valid, tokens, n):
    return flat.new_empty(tokens, n, dtype=torch.int32)


@torch.library.custom_op("gpic::moe_tsum", mutates_args=())
def moe_tsum(src: torch.Tensor, inv: torch.Tensor, tokens: int, n: int) -> torch.Tensor:
    return _tsum(src.contiguous(), None, inv, tokens, n)


@moe_tsum.register_fake
def _(src, inv, tokens, n):
    return src.new_empty(tokens, src.shape[1])


@torch.library.custom_op("gpic::moe_gather2", mutates_args=())
def moe_gather2(x: torch.Tensor, flat: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, flat)


@moe_gather2.register_fake
def _(x, flat, inv):
    return x.new_empty(flat.shape[0], x.shape[1])


def _gather2_setup(ctx, inputs, output):
    x, flat, inv = inputs
    ctx.save_for_backward(inv)
    ctx.tokens = x.shape[0]


def _gather2_bwd(ctx, g):
    (inv,) = ctx.saved_tensors
    return torch.ops.gpic.moe_tsum(g, inv, ctx.tokens, inv.shape[1]), None, None


moe_gather2.register_autograd(_gather2_bwd, setup_context=_gather2_setup)


@torch.library.custom_op("gpic::moe_combine2", mutates_args=())
def moe_combine2(o: torch.Tensor, gate: torch.Tensor, flat: torch.Tensor, inv: torch.Tensor, tokens: int, has_gate: bool) -> torch.Tensor:
    return _tsum(o.contiguous(), gate.float().contiguous() if has_gate else None, inv, tokens, inv.shape[1])


@moe_combine2.register_fake
def _(o, gate, flat, inv, tokens, has_gate):
    return o.new_empty(tokens, o.shape[1])


def _combine2_setup(ctx, inputs, output):
    o, gate, flat, inv, tokens, has_gate = inputs
    ctx.save_for_backward(o, gate, flat)
    ctx.has_gate = has_gate


def _combine2_bwd(ctx, g):
    o, gate, flat = ctx.saved_tensors
    go, gw = moe_combine_bwd(g, o, gate, flat, ctx.has_gate)
    return go, (gw.to(gate.dtype) if ctx.has_gate else None), None, None, None, None


moe_combine2.register_autograd(_combine2_bwd, setup_context=_combine2_setup)


# ==== v3 (GPIC_MOE_KERNELS=3; 2026-11-14) ====
# v2's token sums loop over all n experts per token with a masked load each (at G = 2 of n = 16, 2 of 16 loads are real; at G = 8 of
# n = 64, 8 of 64). v3 builds each token's slot list (CSR, in expert order) from the inverse map with counts and prefix sums, no sort:
# offsets [tokens + 1] = the running sum of each token's slot count, rank[t, e] = the number of experts e' < e that took t, and
# perm[offsets[t] + rank[t, e]] = e * cap + inv[t, e]; every position is written once, so the CSR is the same on every run. The sum
# kernel then loops, per block of tokens, only as often as the block's longest list. The same terms in the same order as v2.
#   gpic::moe_csr3(inv, cap) -> perm [S], offsets [tokens + 1]     (perm's tail past offsets[-1] is unused)
#   gpic::moe_gather3(x, flat, perm, offsets) -> xe                xe[s] = x[flat[s]]; backward: grad_x[t] = sum over t's slots
#   gpic::moe_combine3(o, gate, flat, perm, offsets, tokens, has_gate) -> out   out[t] = sum gate[s] o[s]; backward as v1

@triton.jit
def _csum_kernel(src_ptr, w_ptr, perm_ptr, off_ptr, out_ptr, M, D, s_src, s_out,
                 HAS_W: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    tmask = t < M
    cmask = cols < D
    start = tl.load(off_ptr + t, mask=tmask, other=0)
    end = tl.load(off_ptr + t + 1, mask=tmask, other=0)
    n_k = end - start
    longest = tl.max(n_k, axis=0)
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    for k in range(0, longest):
        live = k < n_k
        s = tl.load(perm_ptr + start + k, mask=live, other=0).to(tl.int64)
        v = tl.load(src_ptr + s[:, None] * s_src + cols[None, :], mask=live[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        if HAS_W:
            v = v * tl.load(w_ptr + s, mask=live, other=0.0)[:, None]
        acc += v
    tl.store(out_ptr + t.to(tl.int64)[:, None] * s_out + cols[None, :], acc.to(out_ptr.dtype.element_ty), mask=tmask[:, None] & cmask[None, :])


def _csum(src, w, perm, offsets, tokens):
    S, D = src.shape
    out = torch.empty(tokens, D, dtype=src.dtype, device=src.device)
    BT, BD = 8, min(256, triton.next_power_of_2(D))
    _csum_kernel[(triton.cdiv(tokens, BT), triton.cdiv(D, BD))](src, w if w is not None else src, perm, offsets, out, tokens, D,
                                                              src.stride(0), out.stride(0), HAS_W=w is not None, BLOCK_T=BT,
                                                              BLOCK_D=BD, num_warps=4)
    return out


@torch.library.custom_op("gpic::moe_csr3", mutates_args=())
def moe_csr3(inv: torch.Tensor, cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    M, n = inv.shape
    valid = inv >= 0
    counts = valid.sum(1, dtype=torch.int32)
    offsets = torch.zeros(M + 1, dtype=torch.int32, device=inv.device)
    offsets[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    # rank[t, e] = the experts e' <= e that took t, minus 1, as a matmul with an upper triangle of ones: torch.cumsum along the 16-wide expert
    # dimension took 0.39 ms on an A40 at 32,768 tokens (w2722's H100 profile: tensor_kernel_scan_innermost_dim 0.22 ms per layer), the
    # matmul 0.03 ms; exact (fp16 holds every count up to 2048, n <= 64 here)
    assert n <= 2048, n
    U = torch.triu(torch.ones(n, n, device=inv.device, dtype=torch.float16))
    rank = (valid.to(torch.float16) @ U).to(torch.int32) - 1
    e = torch.arange(n, device=inv.device, dtype=torch.int32)
    S = n * cap
    pos = torch.where(valid, offsets[:-1, None] + rank, S + torch.arange(M * n, device=inv.device, dtype=torch.int32).view(M, n))
    perm = torch.zeros(S + M * n, dtype=torch.int32, device=inv.device)
    perm.scatter_(0, pos.reshape(-1).to(torch.int64), (e[None, :] * cap + inv).reshape(-1))
    return perm[:S].clone(), offsets


@moe_csr3.register_fake
def _(inv, cap):
    M, n = inv.shape
    return inv.new_empty(n * cap), inv.new_empty(M + 1)


@torch.library.custom_op("gpic::moe_csum", mutates_args=())
def moe_csum(src: torch.Tensor, perm: torch.Tensor, offsets: torch.Tensor, tokens: int) -> torch.Tensor:
    return _csum(src.contiguous(), None, perm, offsets, tokens)


@moe_csum.register_fake
def _(src, perm, offsets, tokens):
    return src.new_empty(tokens, src.shape[1])


@triton.jit
def _gather_rows_kernel(x_ptr, flat_ptr, out_ptr, S, D, s_x, s_out, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    s = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    m = (s < S)[:, None] & (cols < D)[None, :]
    t = tl.load(flat_ptr + s, mask=s < S, other=0).to(tl.int64)
    v = tl.load(x_ptr + t[:, None] * s_x + cols[None, :], mask=m)
    tl.store(out_ptr + s.to(tl.int64)[:, None] * s_out + cols[None, :], v, mask=m)


def _gather_rows(x, flat):
    # ATen's index_select reached about a third of the A40's bandwidth here (2026-11-14, bench_moe_layer.py --mp bf16): 16 rows x 256
    # columns per program, a pure copy
    x = x.contiguous()
    S, D = flat.shape[0], x.shape[1]
    out = torch.empty(S, D, dtype=x.dtype, device=x.device)
    BS, BD = 16, min(256, triton.next_power_of_2(D))
    _gather_rows_kernel[(triton.cdiv(S, BS), triton.cdiv(D, BD))](x, flat, out, S, D, x.stride(0), out.stride(0), BLOCK_S=BS, BLOCK_D=BD,
                                                                num_warps=4)
    return out


@torch.library.custom_op("gpic::moe_gather3", mutates_args=())
def moe_gather3(x: torch.Tensor, flat: torch.Tensor, perm: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    return _gather_rows(x, flat)


@moe_gather3.register_fake
def _(x, flat, perm, offsets):
    return x.new_empty(flat.shape[0], x.shape[1])


def _gather3_setup(ctx, inputs, output):
    x, flat, perm, offsets = inputs
    ctx.save_for_backward(perm, offsets)
    ctx.tokens = x.shape[0]


def _gather3_bwd(ctx, g):
    perm, offsets = ctx.saved_tensors
    return torch.ops.gpic.moe_csum(g, perm, offsets, ctx.tokens), None, None, None


moe_gather3.register_autograd(_gather3_bwd, setup_context=_gather3_setup)


@torch.library.custom_op("gpic::moe_combine3", mutates_args=())
def moe_combine3(o: torch.Tensor, gate: torch.Tensor, flat: torch.Tensor, perm: torch.Tensor, offsets: torch.Tensor, tokens: int,
                 has_gate: bool) -> torch.Tensor:
    return _csum(o.contiguous(), gate.float().contiguous() if has_gate else None, perm, offsets, tokens)


@moe_combine3.register_fake
def _(o, gate, flat, perm, offsets, tokens, has_gate):
    return o.new_empty(tokens, o.shape[1])


def _combine3_setup(ctx, inputs, output):
    o, gate, flat, perm, offsets, tokens, has_gate = inputs
    ctx.save_for_backward(o, gate, flat)
    ctx.has_gate = has_gate


def _combine3_bwd(ctx, g):
    o, gate, flat = ctx.saved_tensors
    go, gw = moe_combine_bwd(g, o, gate, flat, ctx.has_gate)
    return go, (gw.to(gate.dtype) if ctx.has_gate else None), None, None, None, None, None


moe_combine3.register_autograd(_combine3_bwd, setup_context=_combine3_setup)


# ==== v4 (GPIC_MOE_KERNELS=4; 2026-11-14, w2722's profile and the granularity runs) ====
# The slot tensors [S, d] (S = tokens x G slots) are the layer's memory traffic: every pass over one reads or writes S x d values, and S
# grows with the granularity. v3's combine carries the gates: its backward reads g and o and writes the gradient of o (one kernel over
# three slot tensors: 0.16 ms a layer at d12 with 2 active experts, against 0.08 ms for a plain row gather), and o is kept for it until
# the backward. v4 puts the gate on the experts' hidden activations instead (y = sum gate[s] (hid[s] @ W3) = sum (gate[s] hid[s]) @ W3:
# the same function): the combine is a plain sum over a token's slots, its backward a plain row gather of g (grad_o[s] = g[flat[s]]), o is
# not kept, and the gate's gradient comes from the hidden side ([S, h], h = hidden / G wide). An empty slot (gate 0) has a zero gated
# hidden, so its row of o is 0, and the gradients of its hidden and of the weights are 0 as in v3. The gated hidden is rounded to bf16
# before the last GEMM, so the forward differs from v3's in bf16 rounding.
#   gpic::moe_sum4(o, flat, perm, offsets, tokens) -> out      out[t] = sum of o[s] over t's (valid) slots; backward: g[flat[s]]

@torch.library.custom_op("gpic::moe_rows4", mutates_args=())
def moe_rows4(g: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    return _gather_rows(g, flat)


@moe_rows4.register_fake
def _(g, flat):
    return g.new_empty(flat.shape[0], g.shape[1])


@torch.library.custom_op("gpic::moe_sum4", mutates_args=())
def moe_sum4(o: torch.Tensor, flat: torch.Tensor, perm: torch.Tensor, offsets: torch.Tensor, tokens: int) -> torch.Tensor:
    return _csum(o.contiguous(), None, perm, offsets, tokens)


@moe_sum4.register_fake
def _(o, flat, perm, offsets, tokens):
    return o.new_empty(tokens, o.shape[1])


def _sum4_setup(ctx, inputs, output):
    o, flat, perm, offsets, tokens = inputs
    ctx.save_for_backward(flat)


def _sum4_bwd(ctx, g):
    (flat,) = ctx.saved_tensors
    return torch.ops.gpic.moe_rows4(g, flat), None, None, None, None


moe_sum4.register_autograd(_sum4_bwd, setup_context=_sum4_setup)


# ==== the router's scores (GPIC_MOE_ROUTER=1; 2026-11-14, w2722's profile) ====
# moe.py's _scores casts the layer's input to fp32 and calls F.linear: a cast pass over [tokens, d] (0.10 ms a layer at d12), an fp32 copy
# of the input kept for the backward (134 MB a layer), a TF32 GEMM, and in the backward an fp32 [tokens, d] gradient and its cast back:
# about 3 ms a step at d12. Here one kernel reads the input in its own dtype and the fp32 weight and writes the fp32 scores (products and
# sums in IEEE fp32; a bf16 value is exact in fp32); the backward writes the input's gradient in the input's dtype and sums the weight's
# gradient in fp32 (per chunk of tokens, then over the chunks). Nothing in fp32 of the size of the input is made or kept.
# The products: TF32 on the tensor cores by default, as the GEMM of F.linear on an H100 (torch's fp32 matmul precision "high": the inputs
# rounded to 10 mantissa bits, which holds a bf16 value exactly; fp32 sums). GPIC_MOE_ROUTER_PREC=ieee: IEEE fp32 products (no tensor
# cores: about 2 TFLOPS on an A40, 8 x the time at 128 experts).
#   gpic::moe_scores(x [M, d], w [n, d]) -> s [M, n] fp32      s = x @ w.T

@triton.jit
def _scores_kernel(x_ptr, w_ptr, out_ptr, M, D, N, s_x, s_w, s_out,
                   BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PREC: tl.constexpr):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    e = tl.arange(0, BLOCK_N)
    tmask = t < M
    emask = e < N
    acc = tl.zeros([BLOCK_T, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        kmask = k < D
        xt = tl.load(x_ptr + t.to(tl.int64)[:, None] * s_x + k[None, :], mask=tmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
        wt = tl.load(w_ptr + k[:, None] + e[None, :] * s_w, mask=kmask[:, None] & emask[None, :], other=0.0).to(tl.float32)   # w.T's block
        acc = tl.dot(xt, wt, acc, input_precision=PREC)
    tl.store(out_ptr + t.to(tl.int64)[:, None] * s_out + e[None, :], acc, mask=tmask[:, None] & emask[None, :])


@triton.jit
def _scores_gx_kernel(g_ptr, w_ptr, out_ptr, M, D, N, s_g, s_w, s_out,
                      BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PREC: tl.constexpr):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    e = tl.arange(0, BLOCK_N)
    tmask = t < M
    kmask = k < D
    emask = e < N
    g = tl.load(g_ptr + t.to(tl.int64)[:, None] * s_g + e[None, :], mask=tmask[:, None] & emask[None, :], other=0.0).to(tl.float32)
    w = tl.load(w_ptr + e[:, None] * s_w + k[None, :], mask=emask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
    out = tl.dot(g, w, input_precision=PREC)
    tl.store(out_ptr + t.to(tl.int64)[:, None] * s_out + k[None, :], out.to(out_ptr.dtype.element_ty), mask=tmask[:, None] & kmask[None, :])


@triton.jit
def _scores_gw_kernel(g_ptr, x_ptr, out_ptr, M, D, N, s_g, s_x,
                      CHUNK: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PREC: tl.constexpr):
    c = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    e = tl.arange(0, BLOCK_N)
    kmask = k < D
    emask = e < N
    acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    for i in range(0, CHUNK, BLOCK_T):
        t = (c * CHUNK + i + tl.arange(0, BLOCK_T)).to(tl.int64)
        tmask = t < M
        gt = tl.load(g_ptr + e[:, None] + t[None, :] * s_g, mask=emask[:, None] & tmask[None, :], other=0.0).to(tl.float32)   # g.T's block
        xt = tl.load(x_ptr + t[:, None] * s_x + k[None, :], mask=tmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
        acc = tl.dot(gt, xt, acc, input_precision=PREC)
    row = (c * N + e).to(tl.int64)
    tl.store(out_ptr + row[:, None] * D + k[None, :], acc, mask=emask[:, None] & kmask[None, :])


_SC_BT, _SC_BK, _SC_CHUNK = 64, 64, 512
_SC_PREC = os.environ.get("GPIC_MOE_ROUTER_PREC", "tf32")
assert _SC_PREC in ("tf32", "ieee"), _SC_PREC


def _scores(x, w):
    x, w = x.contiguous(), w.contiguous()
    M, D = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    _scores_kernel[(triton.cdiv(M, _SC_BT),)](x, w, out, M, D, N, x.stride(0), w.stride(0), out.stride(0), BLOCK_T=_SC_BT,
                                             BLOCK_N=max(16, triton.next_power_of_2(N)), BLOCK_K=_SC_BK, PREC=_SC_PREC, num_warps=4)
    return out


@torch.library.custom_op("gpic::moe_scores", mutates_args=())
def moe_scores(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _scores(x, w)


@moe_scores.register_fake
def _(x, w):
    return x.new_empty(x.shape[0], w.shape[0], dtype=torch.float32)


@torch.library.custom_op("gpic::moe_scores_gx", mutates_args=())
def moe_scores_gx(g: torch.Tensor, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The gradient of x (x itself gives only the dtype and the shape): g @ w, [M, d]."""
    g, w = g.float().contiguous(), w.contiguous()
    M, N = g.shape
    D = w.shape[1]
    out = torch.empty(M, D, dtype=x.dtype, device=g.device)
    _scores_gx_kernel[(triton.cdiv(M, _SC_BT), triton.cdiv(D, _SC_BK))](g, w, out, M, D, N, g.stride(0), w.stride(0), out.stride(0),
                                                                       BLOCK_T=_SC_BT, BLOCK_N=max(16, triton.next_power_of_2(N)),
                                                                       BLOCK_K=_SC_BK, PREC=_SC_PREC, num_warps=4)
    return out


@moe_scores_gx.register_fake
def _(g, w, x):
    return x.new_empty(x.shape)


@torch.library.custom_op("gpic::moe_scores_gw", mutates_args=())
def moe_scores_gw(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The gradient of w: g.T @ x, [n, d] fp32."""
    g, x = g.float().contiguous(), x.contiguous()
    M, N = g.shape
    D = x.shape[1]
    C = triton.cdiv(M, _SC_CHUNK)
    part = torch.empty(C, N, D, dtype=torch.float32, device=g.device)
    _scores_gw_kernel[(C, triton.cdiv(D, _SC_BK))](g, x, part, M, D, N, g.stride(0), x.stride(0), CHUNK=_SC_CHUNK, BLOCK_T=_SC_BT,
                                                  BLOCK_N=max(16, triton.next_power_of_2(N)), BLOCK_K=_SC_BK, PREC=_SC_PREC, num_warps=4)
    return part.sum(0)


@moe_scores_gw.register_fake
def _(g, x):
    return g.new_empty(g.shape[1], x.shape[1], dtype=torch.float32)


def _scores_setup(ctx, inputs, output):
    x, w = inputs
    ctx.save_for_backward(x, w)


def _scores_bwd(ctx, g):
    x, w = ctx.saved_tensors
    return torch.ops.gpic.moe_scores_gx(g, w, x), torch.ops.gpic.moe_scores_gw(g, x).to(w.dtype)


moe_scores.register_autograd(_scores_bwd, setup_context=_scores_setup)
