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
    the DataLoader with the rest of the batch) and Python ints: everything run() needs, so the step makes no host copy.
    With GPU decoding (gpic_dataset._GPU_DECODE) an item can also be ("jpeg", file bytes, (H, W), rows or None, passes):
    its source, rows r0:r1 of the decoded image, is decoded by run() on the GPU. The source buffer holds the CPU items
    first (job["src"], their bytes) and then the JPEG items, job["src_bytes"] in all; job["jpeg"] is a Host with the files
    back to back in one tensor (it crosses from the worker through shared memory, not pickled through a pipe)."""
    B = len(items)
    K = K or max(it[-1][p][3].shape[1] for it in items for p in range(2))
    pix = [(b, np.ascontiguousarray(it[0])) for b, it in enumerate(items) if not (isinstance(it[0], str) and it[0] == "jpeg")]
    jpg = [(b, it) for b, it in enumerate(items) if isinstance(it[0], str) and it[0] == "jpeg"]
    off = np.zeros(B, dtype=np.int64)
    shapes = [None] * B
    pos = 0
    for b, src in pix:
        off[b], shapes[b] = pos, list(src.shape[:2])
        pos += src.nbytes
    pix_bytes = pos
    index, start = [], 0
    for b, (_, data, (H, W), rows, _) in jpg:
        r0, r1 = (0, H) if rows is None else rows
        off[b], shapes[b] = pos, [r1 - r0, W]
        index.append((start, len(data), int(H), int(W), int(r0), int(r1), int(pos)))
        start += len(data)
        pos += (r1 - r0) * W * 3
    jpeg = Host(torch.from_numpy(np.frombuffer(b"".join(it[1] for _, it in jpg), dtype=np.uint8).copy()), index) if jpg else None
    flat = torch.from_numpy(np.concatenate([src.reshape(-1) for _, src in pix])) if pix else torch.empty(0, dtype=torch.uint8)
    xmin = np.zeros((B, 2, OUT), dtype=np.int32)
    kk = np.zeros((B, 2, OUT, K), dtype=np.int32)
    hw = np.zeros((B, 3, 2), dtype=np.int64)   # (rows, cols) of the source, after pass 1, after pass 2
    axis = np.zeros((B, 2), dtype=np.int64)
    for b, it in enumerate(items):
        ps = it[-1]
        shp = list(shapes[b])
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
            "hw": torch.from_numpy(hw), "axis": torch.from_numpy(axis),   # for lever E2's per-sample repacking
            "jpeg": jpeg, "pix_bytes": int(pix_bytes), "src_bytes": int(pos)}


class Host:
    """The JPEG files of a batch for nvJPEG, which reads them from host memory: blob (uint8 CPU tensor, the files back to back)
    and index [(start, length, H, W, r0, r1, offset in the source buffer)]. It has no .to(), so Lightning's device transfer
    leaves it on the host, and no .pin_memory(), so the DataLoader's pin thread leaves it as it is."""

    def __init__(self, blob, index):
        self.blob, self.index = blob, index


def _decode(datas, dev):
    """nvJPEG on dev for a list of JPEG files (uint8 CPU tensors) -> [3, H, W] uint8 tensors. A file nvJPEG refuses (none of 8k
    training files in tools/mfu/jpeg_gpu_bench.py) is decoded by Pillow on the host instead, as the CPU path would."""
    from torchvision.io import decode_jpeg, ImageReadMode
    try:
        return decode_jpeg(datas, device=dev, mode=ImageReadMode.RGB)
    except RuntimeError as e:
        import io
        from PIL import Image
        out = []
        for d in datas:
            try:
                out.append(decode_jpeg([d], device=dev, mode=ImageReadMode.RGB)[0])
            except RuntimeError:
                print(f"[gpu_decode] nvJPEG refused a {d.numel()}-byte file ({e}); Pillow decodes it", flush=True)
                img = np.asarray(Image.open(io.BytesIO(d.numpy().tobytes())).convert("RGB"))
                out.append(torch.from_numpy(img).permute(2, 0, 1).to(dev))
        return out


_DECODE_STREAM = {}


def _decode_on_side(datas, dev):
    """nvJPEG with its output buffers allocated on a stream of their own, plus an event after the decode. torchvision
    (0.22 decode_jpegs_cuda.cpp) allocates the outputs on the caller's current stream but writes them on its own stream
    without waiting for the caller's stream: a block that the caller's stream freed while its kernels still use it (the
    step before is still running on the GPU) can be handed to nvJPEG and overwritten. In the training step that trained
    to NaN at step 12 (w1896, GPIC_MFU_GPU_DECODE=1). This stream runs no kernels of its own, and the caller keeps the
    images alive for its copies (record_stream), so no block of its pool is in use when nvJPEG gets it."""
    side = _DECODE_STREAM.get(dev)
    if side is None:
        side = _DECODE_STREAM[dev] = torch.cuda.Stream(dev)
    side.synchronize()   # nothing is pending on it (no kernels run there); a guard
    with torch.cuda.stream(side):
        imgs = _decode(datas, dev)
        done = torch.cuda.Event()
        done.record(side)
    return imgs, done


def _sources(job, dev):
    """The batch's source buffer on dev, built on the caller's current stream: the CPU items' bytes, then the JPEG items
    decoded by nvJPEG (rows r0:r1, HWC)."""
    jp = job.get("jpeg")
    if jp is None:
        return job["src"]
    cur = torch.cuda.current_stream(dev) if torch.device(dev).type == "cuda" else None
    if cur is not None:
        imgs, done = _decode_on_side([jp.blob[s:s + n] for s, n, *_ in jp.index], dev)
    else:   # CPU tests
        imgs, done = _decode([jp.blob[s:s + n] for s, n, *_ in jp.index], dev), None
    src = torch.empty(job["src_bytes"], dtype=torch.uint8, device=dev)
    if job["pix_bytes"]:
        src[:job["pix_bytes"]].copy_(job["src"], non_blocking=True)
    if done is not None:
        cur.wait_event(done)
    for img, (_, _, H, W, r0, r1, off) in zip(imgs, jp.index):
        assert tuple(img.shape) == (3, H, W), (tuple(img.shape), H, W)
        src[off:off + (r1 - r0) * W * 3].view(r1 - r0, W, 3).copy_(img[:, r0:r1].permute(1, 2, 0))
        if cur is not None:
            img.record_stream(cur)   # its block goes back to the decode stream's pool only after these copies ran
    return src


def prepare(batch, dev, stream):
    """GPU decoding a step ahead (GPIC_MFU_GPU_DECODE=2, DecodeAhead's thread): the batch's source buffer built on dev in
    stream (copy of the CPU items, nvJPEG, row copies); job["src"] becomes it, with an event that run() makes its stream
    wait for."""
    job = batch[0]
    if not isinstance(job, dict) or job.get("jpeg") is None:
        return batch
    with torch.cuda.stream(stream):
        src = _sources(job, dev)
        ready = torch.cuda.Event()
        ready.record(stream)
    out = list(batch)
    out[0] = dict(job, src=src, jpeg=None, ready=ready)
    return type(batch)(out)


class DecodeAhead:
    """Wraps the training loader (GPIC_MFU_GPU_DECODE=2): each batch's JPEGs are decoded by one background thread on a side
    stream while the step before it trains, so the training thread never waits for nvJPEG (decode_jpeg blocks its caller
    until its stream is done: torchvision 0.22 decode_jpegs_cuda.cpp). The wrapped loader is still pulled on the training
    thread (the shared bucket's all-reduce stays in order on every rank)."""

    def __init__(self, loader):
        self.loader, self.pool, self.stream, self.dev = loader, None, None, None

    def __iter__(self):
        if self.pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self.dev = torch.device("cuda", torch.cuda.current_device())
            self.stream = torch.cuda.Stream(self.dev)
            self.pool = ThreadPoolExecutor(1, thread_name_prefix="gpu_decode", initializer=torch.cuda.set_device, initargs=(self.dev,))
        return _AheadIter(self)

    def __getattr__(self, name):   # everything else (e.g. the dataset) from the wrapped loader
        return getattr(self.__dict__["loader"], name)


class _AheadIter:
    def __init__(self, parent):
        from collections import deque
        self.p, self.it, self.q, self.done = parent, None, deque(), False

    def __iter__(self):
        return self

    def __next__(self):
        p = self.p
        if self.it is None:
            self.it = iter(p.loader)
        while not self.done and len(self.q) < 2:   # the batch returned now and the next one, whose decode runs meanwhile
            try:
                b = next(self.it)
            except StopIteration:
                self.done = True
                break
            self.q.append(p.pool.submit(prepare, b, p.dev, p.stream))
        if not self.q:
            raise StopIteration
        return self.q.popleft().result()


def run(job):
    """The batch's crops, [B, 256, 256, 3] uint8, on the device of job's tensors."""
    dev, B = job["src_off"].device, job["B"]
    ready = job.get("ready")
    if ready is not None:   # built a step ahead on DecodeAhead's stream: wait for it there, and keep its memory until used here
        cur = torch.cuda.current_stream(dev)
        cur.wait_event(ready)
        job["src"].record_stream(cur)
    src = _sources(job, dev)
    mid = torch.empty(job["mid_bytes"], dtype=torch.uint8, device=dev)
    out = torch.empty(B, OUT, OUT, 3, dtype=torch.uint8, device=dev)
    geo = job["geo"]
    passes = [(src, job["src_off"], mid, job["mid_off"]), (mid, job["mid_off"], out, job["out_off"])]
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
