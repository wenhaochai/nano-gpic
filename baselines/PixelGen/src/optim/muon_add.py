"""MFU lever MA (GPIC_MFU_MUON_ADD=1, src/optim/muon.py add_units): Muon's per-part weight updates in one Triton launch."""
import torch
import triton
import triton.language as tl

BLOCK = 4096


@triton.jit
def _add_units_kernel(DST, SRC, N, ALPHA, BLOCK: tl.constexpr):
    k = tl.program_id(0)
    b = tl.program_id(1)
    n = tl.load(N + k)
    if b * BLOCK < n:
        d = tl.load(DST + k).to(tl.pointer_type(tl.float32))
        s = tl.load(SRC + k).to(tl.pointer_type(tl.bfloat16))
        a = tl.load(ALPHA + k)
        i = b.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        w = tl.load(d + i, mask=m)
        o = tl.load(s + i, mask=m).to(tl.float32)
        tl.store(d + i, w + a * o, mask=m)


def add_units_triton(dsts, srcs, alphas):
    """dst += alpha * src, fp32 dst, bf16 src, both contiguous, one launch over every pair."""
    dev = dsts[0].device
    for d, s in zip(dsts, srcs):
        assert d.dtype == torch.float32 and s.dtype == torch.bfloat16 and d.is_contiguous() and s.is_contiguous() \
            and d.numel() == s.numel(), (d.dtype, s.dtype, d.shape, s.shape)
    n = [d.numel() for d in dsts]
    tab = torch.tensor([[d.data_ptr() for d in dsts], [s.data_ptr() for s in srcs], n], dtype=torch.int64).pin_memory()
    al = torch.tensor(alphas, dtype=torch.float32).pin_memory()
    tab, al = tab.to(dev, non_blocking=True), al.to(dev, non_blocking=True)
    _add_units_kernel[(len(dsts), triton.cdiv(max(n), BLOCK))](tab[0], tab[1], tab[2], al, BLOCK=BLOCK, num_warps=8)


# ---- GPIC_MUON_FUSE=1 (MoE line, 2026-11-14, w2722's profile: the optimizer's element-wise passes over an MoE model's 8 x larger shards) ----
# Under FSDP2 the Muon step passes over each rank's shard five times before and after Newton-Schulz: the momentum lerp_ (reads g and m,
# writes m), the Nesterov lerp (reads g and m, writes an fp32 copy), its cast to bf16, the weight decay mul_ (reads and writes w) and the
# add (reads w and the update, writes w). Two launches do the same arithmetic with half the memory traffic:
#   nesterov_units_triton   m <- lerp(m, g, 1 - beta); u <- bf16(lerp(g, m, beta)) (or bf16(m) without Nesterov)
#   decay_add_units_triton  w <- w * (1 - lr wd) + alpha * o
# Each value is computed with the operations ATen uses, in ATen's order (lerp: start + weight * (end - start) for a small weight, end -
# (end - start) * (1 - weight) for a large one, each as one fused multiply-add; the decay's product rounded to fp32, then one fused
# multiply-add), so the results are the same bits (tools/mfu/tests/test_muon_fuse.py: 0 of 16.7M values differ, on random data at two
# scales; a plain w * f + a * o, which the compiler contracts the other way, differs in a quarter of them).

@triton.jit
def _nesterov_kernel(G, M, U, N, w1, c2, NEST: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0)
    b = tl.program_id(1)
    n = tl.load(N + k)
    if b * BLOCK < n:
        g_ptr = tl.load(G + k).to(tl.pointer_type(tl.float32))
        m_ptr = tl.load(M + k).to(tl.pointer_type(tl.float32))
        u_ptr = tl.load(U + k).to(tl.pointer_type(tl.bfloat16))
        i = b.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        msk = i < n
        g = tl.load(g_ptr + i, mask=msk)
        m = tl.load(m_ptr + i, mask=msk)
        m2 = tl.math.fma(g - m, w1 + tl.zeros([BLOCK], tl.float32), m)            # lerp_(m, g, 1 - beta): m + (1 - beta) (g - m)
        tl.store(m_ptr + i, m2, mask=msk)
        if NEST:
            u = tl.math.fma(g - m2, c2 + tl.zeros([BLOCK], tl.float32), m2)       # lerp(g, m2, beta): m2 - (m2 - g) (1 - beta)
        else:
            u = m2
        tl.store(u_ptr + i, u.to(tl.bfloat16), mask=msk)


@triton.jit
def _decay_add_units_kernel(DST, SRC, N, ALPHA, f, BLOCK: tl.constexpr):
    k = tl.program_id(0)
    b = tl.program_id(1)
    n = tl.load(N + k)
    if b * BLOCK < n:
        d = tl.load(DST + k).to(tl.pointer_type(tl.float32))
        s = tl.load(SRC + k).to(tl.pointer_type(tl.bfloat16))
        a = tl.load(ALPHA + k)
        i = b.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        w = tl.load(d + i, mask=m)
        o = tl.load(s + i, mask=m).to(tl.float32)
        wf = w * f                                                                # mul_(1 - lr wd), rounded to fp32
        tl.store(d + i, tl.math.fma(o, a + tl.zeros([BLOCK], tl.float32), wf), mask=m)


def _f32(v):
    return float(torch.tensor(v, dtype=torch.float32))


def nesterov_units_triton(grads, bufs, outs, beta, nesterov):
    """bufs <- lerp(bufs, grads, 1 - beta) in place; outs <- bf16(grads.lerp(bufs, beta)) with nesterov, else bf16(bufs).
    fp32 grads and bufs, bf16 outs, all contiguous and of one size per triple; one launch."""
    dev = grads[0].device
    for g, m, u in zip(grads, bufs, outs):
        assert g.dtype == torch.float32 and m.dtype == torch.float32 and u.dtype == torch.bfloat16 and g.is_contiguous() \
            and m.is_contiguous() and u.is_contiguous() and g.numel() == m.numel() == u.numel(), (g.dtype, m.dtype, u.dtype, g.shape, m.shape)
    n = [g.numel() for g in grads]
    if max(n) == 0:
        return
    tab = torch.tensor([[g.data_ptr() for g in grads], [m.data_ptr() for m in bufs], [u.data_ptr() for u in outs], n],
                       dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
    # ATen's scalar weights: fp32(1 - beta) for the in-place lerp; for the Nesterov lerp fp32(1) - fp32(beta), an fp32 subtraction
    w1 = _f32(1 - beta)
    c2 = float(torch.tensor(1.0, dtype=torch.float32) - torch.tensor(beta, dtype=torch.float32))
    _nesterov_kernel[(len(grads), triton.cdiv(max(n), BLOCK))](tab[0], tab[1], tab[2], tab[3], w1, c2, NEST=bool(nesterov), BLOCK=BLOCK,
                                                             num_warps=8)


def decay_add_units_triton(dsts, srcs, alphas, factor):
    """dst <- dst * factor + alpha * src (fp32 dst, bf16 src, both contiguous), one launch over every pair: mul_(factor) on every dst, then
    add_units, in the same bits. The dsts must cover everything the decay applies to."""
    dev = dsts[0].device
    for d, s in zip(dsts, srcs):
        assert d.dtype == torch.float32 and s.dtype == torch.bfloat16 and d.is_contiguous() and s.is_contiguous() \
            and d.numel() == s.numel(), (d.dtype, s.dtype, d.shape, s.shape)
    n = [d.numel() for d in dsts]
    tab = torch.tensor([[d.data_ptr() for d in dsts], [s.data_ptr() for s in srcs], n], dtype=torch.int64).pin_memory()
    al = torch.tensor(alphas, dtype=torch.float32).pin_memory()
    tab, al = tab.to(dev, non_blocking=True), al.to(dev, non_blocking=True)
    _decay_add_units_kernel[(len(dsts), triton.cdiv(max(n), BLOCK))](tab[0], tab[1], tab[2], al, _f32(factor), BLOCK=BLOCK, num_warps=8)
