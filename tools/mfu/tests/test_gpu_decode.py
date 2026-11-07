#!/usr/bin/env python3
"""GPU JPEG decoding (GPIC_MFU_GPU_DECODE=1; src/data/dataset/pil_exact.py draw_region_crop, gpic_dataset._plan_jpeg,
pil_exact_gpu.pack / run). CPU (TRITON_INTERPRET=1) or one GPU.
1. plan_region_crop, now built on draw_region_crop, equals its previous code (git HEAD~ version, inlined below) bit for bit:
   the same sources and passes on images that need BOX halvings, that run the vertical pass first, and plain ones.
2. On real GPIC JPEGs and PNGs, the worker's GPU-decoding item (header only) carries the same plan as the CPU path from the
   same random state, and its halved items equal the CPU path's sources.
3. A mixed batch (CPU items and undecoded JPEGs) through pack + run equals the all-CPU batch bit for bit when the JPEGs are
   decoded with Pillow (the plumbing: offsets, rows, layout); on the GPU, with nvJPEG, the crops' pixel difference is reported.
   On the CPU (Triton's interpreter is slow) the batch is 11 items: 6 undecoded JPEGs, 3 halved JPEGs, 2 on the CPU path.
4. GPU: DecodeAhead (GPIC_MFU_GPU_DECODE=2: a background thread decodes on a side stream) gives the synchronous path's
   crops bit for bit, over three batches from pinned host tensors as the DataLoader gives them.
5. GPU: decode_jpeg releases the GIL: a pure-Python loop on the main thread keeps its speed while a thread decodes.
6. GPU: the allocation race (w1896: GPIC_MFU_GPU_DECODE=1 trained to NaN at step 12). A buffer is freed while a kernel that
   reads it is still queued behind a GPU sleep, then a batch is decoded: torchvision's own call may hand the freed block to
   nvJPEG, which overwrites it before the reader runs (reported); the decode on its own stream (_decode_on_side) must not.
usage: python test_gpu_decode.py"""
import io, math, os, random, sys, tarfile, threading, time
import numpy as np
import torch
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"
os.environ.update(GPIC_REGION_CROP="1", GPIC_MFU_GPU_RESIZE="1", GPIC_MFU_GPU_DECODE="1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from PIL import Image  # noqa: E402
from src.data.dataset import pil_exact, pil_exact_gpu  # noqa: E402
from src.data.dataset.gpic_dataset import GPICWebDataset  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ok = True


def old_plan(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):   # the code before draw_region_crop
    coeffs = pil_exact.coeffs
    min_s, max_s = math.ceil(image_size / max_crop_frac), math.ceil(image_size / min_crop_frac)
    smaller = random.randrange(min_s, max_s + 1)
    while min(*pil_image.size) >= 2 * smaller:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    W, H = pil_image.size
    scale = smaller / min(W, H)
    out_w, out_h = round(W * scale), round(H * scale)
    crop_y = random.randrange(out_h - image_size + 1)
    crop_x = random.randrange(out_w - image_size + 1)
    sx, sy = W / out_w, H / out_h
    x0, x1 = crop_x * sx, min(W, (crop_x + image_size) * sx)
    y0, y1 = crop_y * sy, min(H, (crop_y + image_size) * sy)
    arr = np.asarray(pil_image, dtype=np.uint8)
    if (H - out_h) > 0 and (H - out_h) > (W - out_w) * 2:
        return arr, [(0,) + coeffs(H, y0, y1, image_size), (1,) + coeffs(W, x0, x1, image_size)]
    support = 2 * max(1.0, sy)
    r0, r1 = max(0, int(y0 - support) - 1), min(H, int(y1 + support) + 2)
    return arr[r0:r1], [(1,) + coeffs(W, x0, x1, image_size), (0,) + coeffs(r1 - r0, y0 - r0, y1 - r0, image_size)]


def same_plan(a, b):
    (sa, pa), (sb, pb) = a, b
    return np.array_equal(sa, sb) and all(x[0] == y[0] and all(np.array_equal(u, v) for u, v in zip(x[1:], y[1:]))
                                          for x, y in zip(pa, pb))


# 1. the refactor
rng = np.random.default_rng(0)
n1 = 0
for W, H in [(640, 427), (1500, 1000), (2100, 1300), (300, 1200), (1200, 300), (512, 512), (330, 1000), (4000, 3000)]:
    img = Image.fromarray(rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8))
    for seed in range(5):
        random.seed(seed); a = old_plan(img, 256)
        random.seed(seed); b = pil_exact.plan_region_crop(img, 256)
        ok &= same_plan(a, b); n1 += 1
print(f"1. plan_region_crop equals its previous code on {n1} (image, seed) pairs: {ok}", flush=True)

# 2. the worker's items on real images
ds = object.__new__(GPICWebDataset)
ds.is_val = False
items_gpu, items_cpu, n_j, n_h, n_o = [], [], 0, 0, 0
halved_idx, png_idx = set(), set()
good2 = True
with tarfile.open("/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_00000.tar") as t:
    for m in t:
        ext = m.name.rsplit(".", 1)[-1]
        if ext not in ("jpg", "png"):
            continue
        data = t.extractfile(m).read()
        seed = len(items_cpu)
        random.seed(seed)
        cpu = pil_exact.plan_region_crop(Image.open(io.BytesIO(data)).convert("RGB"), 256)
        random.seed(seed)
        it = ds._plan_jpeg(data) if ext == "jpg" else None
        n_o_before = n_o
        if it is None:   # the CPU path draws itself
            random.seed(seed)
            it = pil_exact.plan_region_crop(Image.open(io.BytesIO(data)).convert("RGB"), 256)
            n_o += 1
        if ext == "png":
            png_idx.add(len(items_cpu))
        if isinstance(it[0], str):
            n_j += 1
            _, raw, (H, W), rows, passes = it
            arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            src = arr if rows is None else arr[rows[0]:rows[1]]
            good2 &= arr.shape[:2] == (H, W) and same_plan((src, passes), cpu)
        else:
            if ext == "jpg" and n_o_before == n_o:
                n_h += 1
                halved_idx.add(len(items_cpu))
            good2 &= same_plan(it, cpu)
        items_gpu.append(it); items_cpu.append(cpu)
        if len(items_cpu) >= 96:
            break
ok &= good2
print(f"2. worker items from the header ({n_j} undecoded JPEGs, {n_h} decoded here (halved), {n_o} on the CPU path): "
      f"the CPU path's plan from the same random state: {good2}", flush=True)

# 3. pack + run
def to_dev(job):   # as Lightning's transfer: tensors to the device; the Host (JPEG files) and the event stay
    return {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in job.items()}


if DEV == "cpu":   # a small batch: 6 undecoded JPEGs, 3 halved JPEGs, 2 PNGs (CPU path), in shard order
    want, pick = {"jpeg": 6, "halved": 3, "png": 2}, []
    for i, it in enumerate(items_gpu):
        k = "jpeg" if isinstance(it[0], str) else ("halved" if i in halved_idx else "png" if i in png_idx else None)
        if k and want[k] > 0:
            want[k] -= 1
            pick.append(i)
    items_gpu, items_cpu = [items_gpu[i] for i in pick], [items_cpu[i] for i in pick]

ref = pil_exact_gpu.run(to_dev(pil_exact_gpu.pack(items_cpu))).cpu()
import torchvision.io as tvio
real_decode = tvio.decode_jpeg
def pil_decode(datas, device="cpu", mode=None):   # Pillow's pixels, as CHW tensors on the device
    return [torch.from_numpy(np.asarray(Image.open(io.BytesIO(d.numpy().tobytes())).convert("RGB"))).permute(2, 0, 1)
            .contiguous().to(device) for d in datas]
tvio.decode_jpeg = pil_decode
mixed = pil_exact_gpu.run(to_dev(pil_exact_gpu.pack(items_gpu))).cpu()
tvio.decode_jpeg = real_decode
plumb = torch.equal(mixed, ref)
ok &= plumb
n3 = sum(isinstance(it[0], str) for it in items_gpu)
line = (f"3. mixed batch of {len(items_gpu)} ({n3} undecoded JPEGs) through pack + run, JPEGs decoded with Pillow: equal to the "
        f"all-CPU batch: {plumb}")
if DEV == "cuda":
    nv = pil_exact_gpu.run(to_dev(pil_exact_gpu.pack(items_gpu))).cpu()
    d = (nv.to(torch.int16) - ref.to(torch.int16)).abs()
    line += (f"; with nvJPEG: mean |diff| {d.float().mean():.3f} of 255, {(d > 0).float().mean():.1%} of values differ, "
             f"max {int(d.max())}")
print(line, flush=True)

if DEV == "cuda":
    # 4. DecodeAhead against the synchronous path
    def pinned(job):
        return {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in job.items()}
    subsets = [items_gpu[i::3] for i in range(3)]
    sync_out = [pil_exact_gpu.run(to_dev(pinned(pil_exact_gpu.pack(sub)))).cpu() for sub in subsets]
    loader = [(pinned(pil_exact_gpu.pack(sub)), None, {}) for sub in subsets]
    outs = [pil_exact_gpu.run(to_dev(b[0])).cpu() for b in pil_exact_gpu.DecodeAhead(loader)]
    good4 = len(outs) == 3 and all(torch.equal(a, b) for a, b in zip(outs, sync_out))
    ok &= good4
    print(f"4. DecodeAhead over 3 batches: the synchronous path's crops bit for bit: {good4}", flush=True)

    # 5. the GIL during decode_jpeg
    jp = pil_exact_gpu.pack(items_gpu)["jpeg"]
    datas = [jp.blob[s:s + n] for s, n, *_ in jp.index]
    dev = torch.device("cuda", 0)
    def spin(n=3_000_000):
        t, x = time.perf_counter(), 0
        for i in range(n):
            x += i
        return time.perf_counter() - t
    alone = min(spin() for _ in range(3))
    stop, n_dec = threading.Event(), [0, 0.0]
    def bg():
        torch.cuda.set_device(dev)
        while not stop.is_set():
            t = time.perf_counter()
            pil_exact_gpu._decode(datas, dev)
            n_dec[0] += 1
            n_dec[1] += time.perf_counter() - t
    th = threading.Thread(target=bg)
    th.start()
    time.sleep(1.0)
    shared = min(spin() for _ in range(3))
    stop.set()
    th.join()
    print(f"5. a pure-Python loop: {1e3 * alone:.0f} ms alone, {1e3 * shared:.0f} ms while a thread decodes {len(datas)}-file "
          f"batches with nvJPEG ({1e3 * n_dec[1] / max(1, n_dec[0]):.1f} ms per batch, {n_dec[0]} batches): "
          f"{'the GIL is released' if shared < 1.3 * alone else 'WARNING: the decode holds the GIL'}", flush=True)
if DEV == "cuda":
    # 6. the allocation race
    def race(on_side):
        torch.cuda.synchronize()
        H0, W0 = jp.index[0][2], jp.index[0][3]
        buf = torch.full((3 * H0 * W0,), 7, dtype=torch.uint8, device=dev)   # the size of the first decoded image
        torch.cuda._sleep(300_000_000)               # about 0.2 s of GPU time queued before the reader
        total = buf.sum(dtype=torch.int64)           # the reader, queued behind the sleep
        del buf                                      # free for the allocator now; its reader has not run
        if on_side:
            imgs, done = pil_exact_gpu._decode_on_side(datas, dev)
            torch.cuda.current_stream().wait_event(done)
        else:
            imgs = pil_exact_gpu._decode(datas, dev)
        torch.cuda.synchronize()
        return int(total) == 7 * 3 * H0 * W0
    raw = [race(False) for _ in range(3)]
    side = [race(True) for _ in range(3)]
    good6 = all(side)
    ok &= good6
    print(f"6. a freed buffer whose reader is still queued: torchvision's call on the current stream kept it intact in "
          f"{sum(raw)} of 3 tries (False = the race reproduced); on the decode stream in {sum(side)} of 3: {good6}", flush=True)

print("GPU_DECODE_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
