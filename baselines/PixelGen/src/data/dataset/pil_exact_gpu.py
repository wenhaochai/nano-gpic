"""MFU lever C (docs/mfu.md): the two resize passes of a batch's region crops on the GPU, bit for bit Pillow's.

pack(items) (CPU, in the collate) turns the workers' (src, passes) pairs (pil_exact.plan_region_crop) into one batch:
the sources back to back as bytes, their offsets and shapes, and each sample's two passes (axis, xmin, kk padded to K
taps with zero weights). run(job) (GPU, in the training step) runs pass 1 into a scratch buffer and pass 2 into
[B, 256, 256, 3] uint8, one Triton launch per pass: out = clamp((2^21 + sum_j in[xmin + j] * kk[j]) >> 22, 0, 255) in
int32, the integer arithmetic of Pillow's ImagingResample{Horizontal,Vertical}_8bpc, so the result is exact.
"""
import numpy as np
import torch
import triton
import triton.language as tl

OUT = 256


@triton.jit
def _pass_kernel(src_ptr, src_off_ptr, out_ptr, out_off_ptr, xmin_ptr, kk_ptr,
                 n_i_ptr, s_in_i_ptr, s_in_a_ptr, len_a_ptr, s_out_i_ptr, s_out_o_ptr, p,
                 K: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_O: tl.constexpr, NOUT: tl.constexpr):
    """One pass for sample b: out[i, o, c] = clip8(2^21 + sum_j in[i, xmin[o] + j, c] * kk[o, j]); i runs over the axis
    the pass keeps, o over the OUT outputs along the axis it resamples. Strides in bytes of uint8 RGB (3 per pixel)."""
    b = tl.program_id(0)
    ib = tl.program_id(1)
    ob = tl.program_id(2)
    n_i = tl.load(n_i_ptr + b)
    if ib * BLOCK_I >= n_i:
        return
    i = ib * BLOCK_I + tl.arange(0, BLOCK_I)
    o = ob * BLOCK_O + tl.arange(0, BLOCK_O)
    mi = i < n_i
    s_in_i = tl.load(s_in_i_ptr + b)
    s_in_a = tl.load(s_in_a_ptr + b)
    len_a = tl.load(len_a_ptr + b)
    s_out_i = tl.load(s_out_i_ptr + b)
    s_out_o = tl.load(s_out_o_ptr + b)
    src = src_ptr + tl.load(src_off_ptr + b)
    dst = out_ptr + tl.load(out_off_ptr + b)
    tab = (b * 2 + p) * NOUT   # a constexpr argument: a compiled kernel cannot read a module global (the interpreter can)
    xmin = tl.load(xmin_ptr + tab + o)
    acc0 = tl.full([BLOCK_I, BLOCK_O], 1 << 21, tl.int32)
    acc1 = tl.full([BLOCK_I, BLOCK_O], 1 << 21, tl.int32)
    acc2 = tl.full([BLOCK_I, BLOCK_O], 1 << 21, tl.int32)
    for j in tl.static_range(K):
        k = tl.load(kk_ptr + (tab + o) * K + j)
        a = tl.minimum(xmin + j, len_a - 1)
        base = src + i[:, None].to(tl.int64) * s_in_i + a[None, :].to(tl.int64) * s_in_a
        m = mi[:, None]
        acc0 += tl.load(base + 0, mask=m, other=0).to(tl.int32) * k[None, :]
        acc1 += tl.load(base + 1, mask=m, other=0).to(tl.int32) * k[None, :]
        acc2 += tl.load(base + 2, mask=m, other=0).to(tl.int32) * k[None, :]
    out = dst + i[:, None].to(tl.int64) * s_out_i + o[None, :].to(tl.int64) * s_out_o
    m = mi[:, None]
    tl.store(out + 0, tl.minimum(tl.maximum(acc0 >> 22, 0), 255).to(tl.uint8), mask=m)
    tl.store(out + 1, tl.minimum(tl.maximum(acc1 >> 22, 0), 255).to(tl.uint8), mask=m)
    tl.store(out + 2, tl.minimum(tl.maximum(acc2 >> 22, 0), 255).to(tl.uint8), mask=m)


# The same passes split by axis so that each kernel reads memory along its contiguous axis (the generic kernel above read
# the vertical pass across rows, one row per lane, and recomputed 64-bit addresses for every tap; 1.0 ms per pass for 128
# crops, w1866). The arithmetic is the generic kernel's, term by term in the same order, so the result is the same bit for bit.
@triton.jit
def _rows_kernel(src_ptr, src_off_ptr, out_ptr, out_off_ptr, xmin_ptr, kk_ptr, sel_ptr, n_i_ptr, s_in_ptr, len_a_ptr,
                 s_out_ptr, p, K: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_Q: tl.constexpr, NOUT: tl.constexpr):
    """The vertical pass of sample sel[pid 0] (it resamples the rows): out[o, q] = clip8(2^21 + sum_j in[xmin[o] + j, q] *
    kk[o, j]), q over the bytes of a row (columns x 3 channels: the channels are filtered alike, so a row is one byte vector)."""
    b = tl.load(sel_ptr + tl.program_id(0))
    qb = tl.program_id(1)
    ob = tl.program_id(2)
    n_q = tl.load(n_i_ptr + b).to(tl.int32) * 3
    if qb * BLOCK_Q >= n_q:
        return
    q = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    o = ob * BLOCK_O + tl.arange(0, BLOCK_O)
    mq = (q < n_q)[None, :]
    s_in = tl.load(s_in_ptr + b).to(tl.int32)
    len_a = tl.load(len_a_ptr + b).to(tl.int32)
    s_out = tl.load(s_out_ptr + b).to(tl.int32)
    src = src_ptr + tl.load(src_off_ptr + b)
    dst = out_ptr + tl.load(out_off_ptr + b)
    tab = (b * 2 + p) * NOUT
    xmin = tl.load(xmin_ptr + tab + o)
    acc = tl.full([BLOCK_O, BLOCK_Q], 1 << 21, tl.int32)
    for j in tl.static_range(K):
        k = tl.load(kk_ptr + (tab + o) * K + j)
        a = tl.minimum(xmin + j, len_a - 1)
        v = tl.load(src + (a * s_in)[:, None] + q[None, :], mask=mq, other=0)
        acc += v.to(tl.int32) * k[:, None]
    tl.store(dst + (o * s_out)[:, None] + q[None, :], tl.minimum(tl.maximum(acc >> 22, 0), 255).to(tl.uint8), mask=mq)


@triton.jit
def _cols_kernel(src_ptr, src_off_ptr, out_ptr, out_off_ptr, xmin_ptr, kk_ptr, sel_ptr, n_i_ptr, s_in_ptr, len_a_ptr,
                 s_out_ptr, p, K: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_Q: tl.constexpr, NOUT: tl.constexpr):
    """The horizontal pass of sample sel[pid 0] (it resamples the columns): out[r, 3 o + c] = clip8(2^21 + sum_j
    in[r, 3 (xmin[o] + j) + c] * kk[o, j]), over the 3 x NOUT output bytes of a row (BLOCK_Q divides 3 x NOUT)."""
    b = tl.load(sel_ptr + tl.program_id(0))
    rb = tl.program_id(1)
    qb = tl.program_id(2)
    n_r = tl.load(n_i_ptr + b).to(tl.int32)
    if rb * BLOCK_R >= n_r:
        return
    r = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    q = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    o = q // 3
    c = q - 3 * o
    mr = (r < n_r)[:, None]
    s_in = tl.load(s_in_ptr + b)
    len_a = tl.load(len_a_ptr + b).to(tl.int32)
    s_out = tl.load(s_out_ptr + b).to(tl.int32)
    src = src_ptr + tl.load(src_off_ptr + b)
    dst = out_ptr + tl.load(out_off_ptr + b)
    tab = (b * 2 + p) * NOUT
    xm = tl.load(xmin_ptr + tab + o)
    row = src + r.to(tl.int64)[:, None] * s_in
    acc = tl.full([BLOCK_R, BLOCK_Q], 1 << 21, tl.int32)
    for j in tl.static_range(K):
        k = tl.load(kk_ptr + (tab + o) * K + j)
        a = tl.minimum(xm + j, len_a - 1)
        v = tl.load(row + (3 * a + c)[None, :], mask=mr, other=0)
        acc += v.to(tl.int32) * k[None, :]
    tl.store(dst + (r * s_out)[:, None] + q[None, :], tl.minimum(tl.maximum(acc >> 22, 0), 255).to(tl.uint8), mask=mr)


def _geometry(hw, axis, p):
    """Per-sample loop sizes and byte strides of pass p: input hw[:, p] (rows, cols), output hw[:, p + 1]."""
    rows, cols = hw[:, p, 0].astype(np.int64), hw[:, p, 1].astype(np.int64)
    ax = axis[:, p]
    s_row, s_col = cols * 3, np.full_like(cols, 3)
    orows, ocols = hw[:, p + 1, 0].astype(np.int64), hw[:, p + 1, 1].astype(np.int64)
    so_row, so_col = ocols * 3, np.full_like(ocols, 3)
    # along axis 1 (columns) the pass keeps the rows; along axis 0 it keeps the columns
    geo = np.stack([np.where(ax == 1, rows, cols),          # n_i: positions of the kept axis
                    np.where(ax == 1, s_row, s_col),        # input stride along the kept axis
                    np.where(ax == 1, s_col, s_row),        # input stride along the resampled axis
                    np.where(ax == 1, cols, rows),          # input length along the resampled axis
                    np.where(ax == 1, so_row, so_col),      # output stride along the kept axis
                    np.where(ax == 1, so_col, so_row)])     # output stride along the resampled axis
    return geo, orows * ocols * 3


def pack(items, K=None):
    """items: [(src HxWx3 uint8, [(axis, xmin, xlen, kk), (axis, xmin, xlen, kk)])] -> a dict of CPU tensors (pinned by
    the DataLoader with the rest of the batch) and Python ints: everything run() needs, so the step makes no host copy."""
    B = len(items)
    K = K or max(p[3].shape[1] for _, ps in items for p in ps)
    srcs = [np.ascontiguousarray(s) for s, _ in items]
    off = np.zeros(B, dtype=np.int64)
    off[1:] = np.cumsum([s.nbytes for s in srcs])[:-1]
    flat = torch.from_numpy(np.concatenate([s.reshape(-1) for s in srcs]))
    xmin = np.zeros((B, 2, OUT), dtype=np.int32)
    kk = np.zeros((B, 2, OUT, K), dtype=np.int32)
    hw = np.zeros((B, 3, 2), dtype=np.int64)   # (rows, cols) of the source, after pass 1, after pass 2
    axis = np.zeros((B, 2), dtype=np.int64)
    for b, (s, ps) in enumerate(items):
        shp = list(s.shape[:2])
        hw[b, 0] = shp
        for p, (ax, xm, xl, k) in enumerate(ps):
            assert len(xm) == OUT
            xmin[b, p], axis[b, p] = xm, ax
            kk[b, p, :, :k.shape[1]] = k
            shp[ax] = OUT
            hw[b, p + 1] = shp
    g1, mid_bytes = _geometry(hw, axis, 0)
    g2, _ = _geometry(hw, axis, 1)
    mid_off = np.zeros(B, dtype=np.int64)
    mid_off[1:] = np.cumsum(mid_bytes)[:-1]
    sel, sel_slices, sel_max = [], [], []   # per pass: the samples it resamples vertically, then horizontally (_rows/_cols_kernel)
    for p, g in enumerate((g1, g2)):
        v, h = np.nonzero(axis[:, p] == 0)[0], np.nonzero(axis[:, p] == 1)[0]
        sel_slices.append([(len(sel), len(v)), (len(sel) + len(v), len(h))])
        sel_max.append([int(3 * g[0][v].max()) if len(v) else 0, int(g[0][h].max()) if len(h) else 0])
        sel += [int(x) for x in v] + [int(x) for x in h]
    return {"sel": torch.tensor(sel, dtype=torch.int32), "sel_slices": sel_slices, "sel_max": sel_max,
            "src": flat, "src_off": torch.from_numpy(off), "xmin": torch.from_numpy(xmin), "kk": torch.from_numpy(kk),
            "geo": torch.from_numpy(np.stack([g1, g2])),     # [2 passes, 6, B] int64
            "mid_off": torch.from_numpy(mid_off), "out_off": torch.arange(B, dtype=torch.int64) * (OUT * OUT * 3),
            "K": int(K), "B": B, "mid_bytes": int(mid_bytes.sum()), "n_i_max": [int(g1[0].max()), int(g2[0].max())],
            "hw": torch.from_numpy(hw), "axis": torch.from_numpy(axis)}   # for lever E2's per-sample repacking


def run(job):
    """The batch's crops, [B, 256, 256, 3] uint8, on the device of job's tensors."""
    dev, B = job["src"].device, job["B"]
    mid = torch.empty(job["mid_bytes"], dtype=torch.uint8, device=dev)
    out = torch.empty(B, OUT, OUT, 3, dtype=torch.uint8, device=dev)
    geo = job["geo"]
    passes = [(job["src"], job["src_off"], mid, job["mid_off"]), (mid, job["mid_off"], out, job["out_off"])]
    if "sel" not in job:   # a job built elsewhere (lever E2's repacking): the generic kernel
        BI, BO = 32, 64
        for p, (s_ptr, s_off, d_ptr, d_off) in enumerate(passes):
            g = geo[p]
            grid = (B, triton.cdiv(job["n_i_max"][p], BI), OUT // BO)
            _pass_kernel[grid](s_ptr, s_off, d_ptr, d_off, job["xmin"], job["kk"], g[0], g[1], g[2], g[3], g[4], g[5], p,
                               K=job["K"], BLOCK_I=BI, BLOCK_O=BO, NOUT=OUT)
        return out
    sel, K = job["sel"], job["K"]
    RO, RQ, CR, CQ = 32, 128, 16, 256   # rows kernel: 32 output rows x 128 bytes; cols kernel: 16 rows x 256 output bytes
    for p, (s_ptr, s_off, d_ptr, d_off) in enumerate(passes):
        g = geo[p]   # n_i, s_in_i, s_in_a, len_a, s_out_i, s_out_o (_geometry)
        (v0, nv), (h0, nh) = job["sel_slices"][p]
        qmax, rmax = job["sel_max"][p]
        if nv:   # vertical: the row stride is s_in_a in, s_out_o out
            _rows_kernel[(nv, triton.cdiv(qmax, RQ), OUT // RO)](s_ptr, s_off, d_ptr, d_off, job["xmin"], job["kk"],
                                                                 sel[v0:v0 + nv], g[0], g[2], g[3], g[5], p,
                                                                 K=K, BLOCK_O=RO, BLOCK_Q=RQ, NOUT=OUT)
        if nh:   # horizontal: the row stride is s_in_i in, s_out_i out
            _cols_kernel[(nh, triton.cdiv(rmax, CR), 3 * OUT // CQ)](s_ptr, s_off, d_ptr, d_off, job["xmin"], job["kk"],
                                                                     sel[h0:h0 + nh], g[0], g[1], g[3], g[4], p,
                                                                     K=K, BLOCK_R=CR, BLOCK_Q=CQ, NOUT=OUT)
    return out
