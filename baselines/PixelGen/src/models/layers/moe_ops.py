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
