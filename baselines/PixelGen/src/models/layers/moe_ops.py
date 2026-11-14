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
