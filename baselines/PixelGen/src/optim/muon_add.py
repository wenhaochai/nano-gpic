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
