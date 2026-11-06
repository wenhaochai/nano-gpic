"""FlashAttention-3 (Hopper) for the ladder's key-padding attention, as torch custom ops that torch.compile can trace.

The ladder masks padded caption tokens as keys (SDPA boolean mask [B, 1, 1, Sk]), and SDPA's flash kernel takes no mask, so
production runs the memory-efficient kernel (an sm80 CUTLASS kernel, about 75 TFLOP/s on H100 at these shapes). The keys of
every sample are its 256 image tokens followed by its valid caption tokens and then the padding, so the valid keys of
sample b are exactly its first seqused_k[b] keys: FA3's varlen interface with a uniform cu_seqlens (the padded layout) and
seqused_k computes the same attention without a mask. Queries are all computed, as with the SDPA mask.

Tensors are [B, S, H, D] (FA's layout; no transpose to [B, H, S, D]). The kernels come from the HF kernels hub build
kernels-community/flash-attn3 (torch27-cxx11-cu126, revision e29f138), at GPIC_FA3_PATH (default
envs/mfu-extra/fa3_torch27_cu126). Its Python interface registers no fake kernels, so these ops do.
Backward: dk and dv start at zero, so padded keys get exactly zero gradient (as under the SDPA mask) whether or not the
kernel writes their rows.
"""
import os
import sys

import torch

_FA3 = None


def _lib():
    global _FA3
    if _FA3 is None:
        path = os.environ.get("GPIC_FA3_PATH", "/scratch/gpfs/GROUP/USER/envs/mfu-extra/fa3_torch27_cu126")
        if path not in sys.path:
            sys.path.insert(0, path)
        from flash_attn3 import flash_attn_interface as fai
        _FA3 = fai
    return _FA3


def _cu(B, S, device):
    return torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=device)


@torch.library.custom_op("gpic::fa3_fwd", mutates_args=(), device_types="cuda")
def fa3_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqused_k: torch.Tensor,
            softmax_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    out, lse, *_ = _lib()._flash_attn_forward(
        q.reshape(B * Sq, H, D), k.reshape(B * Sk, H, D), v.reshape(B * Sk, H, D), k_new=None, v_new=None, qv=None,
        out=None, cu_seqlens_q=_cu(B, Sq, q.device), cu_seqlens_k=_cu(B, Sk, q.device), cu_seqlens_k_new=None,
        seqused_q=None, seqused_k=seqused_k, max_seqlen_q=Sq, max_seqlen_k=Sk, page_table=None, kv_batch_idx=None,
        leftpad_k=None, rotary_cos=None, rotary_sin=None, seqlens_rotary=None, q_descale=None, k_descale=None,
        v_descale=None, softmax_scale=softmax_scale, causal=False)
    return out.view(B, Sq, H, D), lse


@fa3_fwd.register_fake
def _(q, k, v, seqused_k, softmax_scale):
    B, Sq, H, D = q.shape
    return torch.empty_like(q), q.new_empty((H, B * Sq), dtype=torch.float32)


@torch.library.custom_op("gpic::fa3_bwd", mutates_args=(), device_types="cuda")
def fa3_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
            lse: torch.Tensor, seqused_k: torch.Tensor, softmax_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    dq = torch.empty((B * Sq, H, D), dtype=q.dtype, device=q.device)
    dk = torch.zeros((B * Sk, H, D), dtype=k.dtype, device=k.device)
    dv = torch.zeros((B * Sk, H, D), dtype=v.dtype, device=v.device)
    _lib()._flash_attn_backward(
        dout.reshape(B * Sq, H, D), q.reshape(B * Sq, H, D), k.reshape(B * Sk, H, D), v.reshape(B * Sk, H, D),
        out.reshape(B * Sq, H, D), lse, cu_seqlens_q=_cu(B, Sq, q.device), cu_seqlens_k=_cu(B, Sk, q.device),
        sequed_q=None, sequed_k=seqused_k, max_seqlen_q=Sq, max_seqlen_k=Sk, dq=dq, dk=dk, dv=dv,
        softmax_scale=softmax_scale, causal=False)
    return dq.view(B, Sq, H, D), dk.view(B, Sk, H, D), dv.view(B, Sk, H, D)


@fa3_bwd.register_fake
def _(dout, q, k, v, out, lse, seqused_k, softmax_scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup(ctx, inputs, output):
    q, k, v, seqused_k, softmax_scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, seqused_k)
    ctx.softmax_scale = softmax_scale


def _backward(ctx, dout, dlse):
    q, k, v, out, lse, seqused_k = ctx.saved_tensors
    dq, dk, dv = fa3_bwd(dout.contiguous(), q, k, v, out, lse, seqused_k, ctx.softmax_scale)
    return dq, dk, dv, None, None


fa3_fwd.register_autograd(_backward, setup_context=_setup)


def fa3_attention(q, k, v, seqused_k, softmax_scale=None):
    """q [B, Sq, H, D], k/v [B, Sk, H, D] (last dim contiguous), seqused_k [B] int32: the number of valid leading keys of
    each sample. Returns [B, Sq, H, D]."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return fa3_fwd(q.contiguous(), k.contiguous(), v.contiguous(), seqused_k, softmax_scale)[0]


# Packed (varlen) form: q [Tq, H, D] and k/v [Tk, H, D] hold the sequences back to back; cu_seqlens_q/k [B + 1] int32
# their offsets; max_seqlen_q/k bound the longest sequence (fixed bounds keep the compiled graphs static: the kernel's grid
# covers max_seqlen, and blocks past a sequence's end exit at once).
@torch.library.custom_op("gpic::fa3_varlen_fwd", mutates_args=(), device_types="cuda")
def fa3_varlen_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
                   max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_ = _lib()._flash_attn_forward(
        q, k, v, k_new=None, v_new=None, qv=None, out=None, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_k_new=None, seqused_q=None, seqused_k=None, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        page_table=None, kv_batch_idx=None, leftpad_k=None, rotary_cos=None, rotary_sin=None, seqlens_rotary=None,
        q_descale=None, k_descale=None, v_descale=None, softmax_scale=softmax_scale, causal=False)
    return out, lse


@fa3_varlen_fwd.register_fake
def _(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale):
    return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)


@torch.library.custom_op("gpic::fa3_varlen_bwd", mutates_args=(), device_types="cuda")
def fa3_varlen_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                   lse: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int,
                   max_seqlen_k: int, softmax_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.zeros_like(k), torch.zeros_like(v)
    _lib()._flash_attn_backward(
        dout, q, k, v, out, lse, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, sequed_q=None, sequed_k=None,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, dq=dq, dk=dk, dv=dv, softmax_scale=softmax_scale,
        causal=False)
    return dq, dk, dv


@fa3_varlen_bwd.register_fake
def _(dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_varlen(ctx, inputs, output):
    q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k)
    ctx.max_q, ctx.max_k, ctx.softmax_scale = max_q, max_k, softmax_scale


def _backward_varlen(ctx, dout, dlse):
    q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
    dq, dk, dv = fa3_varlen_bwd(dout.contiguous(), q, k, v, out, lse, cu_q, cu_k, ctx.max_q, ctx.max_k, ctx.softmax_scale)
    return dq, dk, dv, None, None, None, None, None


fa3_varlen_fwd.register_autograd(_backward_varlen, setup_context=_setup_varlen)


def fa3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=None):
    """Packed attention: q [Tq, H, D], k/v [Tk, H, D]; sequence b is rows cu[b]:cu[b+1]. Returns [Tq, H, D]."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return fa3_varlen_fwd(q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q),
                          int(max_seqlen_k), softmax_scale)[0]
