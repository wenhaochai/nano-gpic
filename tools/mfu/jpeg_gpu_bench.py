#!/usr/bin/env python3
"""GPU JPEG decoding for the data path (owner 2026-11-07: 「可以用GPU解码」): torchvision.io.decode_jpeg on the GPU (nvJPEG)
against Pillow (libjpeg-turbo) on real GPIC images. Per batch of B: wall time, the calling process's CPU time (Huffman
decoding on the CPU would show here), the GPU time; images Pillow decodes but the GPU cannot; pixel differences against
Pillow's RGB decode. Run one copy per GPU at once (all 8 decode engines busy, as in training).
usage: python jpeg_gpu_bench.py <shard index> [B]"""
import io, sys, tarfile, time, statistics as st
import numpy as np
import torch, torchvision
from torchvision.io import decode_jpeg, ImageReadMode
from PIL import Image

idx = int(sys.argv[1]); B = int(sys.argv[2]) if len(sys.argv) > 2 else 128
path = f"/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_{idx:05d}.tar"
jpgs = []
with tarfile.open(path) as t:
    for m in t:
        if m.name.endswith(".jpg"):
            jpgs.append(t.extractfile(m).read())
        if len(jpgs) >= 8 * B:
            break
dev = "cuda"
data = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in jpgs]
# Pillow, one thread
t0 = time.perf_counter()
pil = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in jpgs[:B]]
t_pil = (time.perf_counter() - t0) / B * 1000
# GPU, batched; images it rejects are counted
bad = 0
ok_idx = []
for i in range(0, len(data), B):
    try:
        decode_jpeg(data[i:i + B], device=dev, mode=ImageReadMode.RGB)
        ok_idx += list(range(i, i + B))
    except Exception:
        for j in range(i, min(i + B, len(data))):
            try:
                decode_jpeg(data[j], device=dev, mode=ImageReadMode.RGB); ok_idx.append(j)
            except Exception:
                bad += 1
torch.cuda.synchronize()
good = [data[j] for j in ok_idx]
walls, cpus, gpus = [], [], []
for rep in range(5):
    for i in range(0, len(good) - B + 1, B):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        c0, w0 = time.process_time(), time.perf_counter()
        e0.record()
        out = decode_jpeg(good[i:i + B], device=dev, mode=ImageReadMode.RGB)
        e1.record()
        c1 = time.process_time()
        torch.cuda.synchronize()
        w1 = time.perf_counter()
        if rep:
            walls.append((w1 - w0) / B * 1000); cpus.append((c1 - c0) / B * 1000); gpus.append(e0.elapsed_time(e1) / B)
gpu0 = decode_jpeg([data[j] for j in range(B) if j in set(ok_idx)], device=dev, mode=ImageReadMode.RGB)
diffs = []
for k, j in enumerate([j for j in range(B) if j in set(ok_idx)][:64]):
    g = gpu0[k].permute(1, 2, 0).cpu().numpy().astype(np.int16)
    p = pil[j].astype(np.int16)
    if g.shape != p.shape:
        diffs.append((999, 1.0)); continue
    d = np.abs(g - p)
    diffs.append((int(d.max()), float((d > 0).mean())))
mx = max(d[0] for d in diffs); frac = st.mean(d[1] for d in diffs)
mean_abs = st.mean(float(np.abs(gpu0[k].permute(1, 2, 0).cpu().numpy().astype(np.int16) - pil[j].astype(np.int16)).mean())
                   for k, j in enumerate([j for j in range(B) if j in set(ok_idx)][:64]) if gpu0[k].shape[1:] == pil[j].shape[:2])
print(f"GPU {torch.cuda.current_device()} shard {idx} B={B} torchvision {torchvision.__version__}: Pillow {t_pil:.2f} ms/img (one thread); "
      f"GPU decode per image: wall {st.median(walls):.3f} ms, calling process CPU {st.median(cpus):.3f} ms, GPU {st.median(gpus):.3f} ms; "
      f"rejected by the GPU {bad} of {len(data)}; vs Pillow: max |diff| {mx}, pixels differing {frac:.1%}, mean |diff| {mean_abs:.3f}", flush=True)
