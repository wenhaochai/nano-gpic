#!/usr/bin/env python3
"""Lever C's resize passes on one GPU: the generic kernel (_pass_kernel) against the axis-split kernels (_rows_kernel,
_cols_kernel) on a real batch of 128 crops from one shard, the same job: equal outputs bit for bit, and the time of
both passes per batch (CUDA events, median of 50 runs after 10 warm-up runs).
usage: python resize_bench.py [shard index] [batch]"""
import io, os, random, sys, tarfile

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("GPIC_REGION_CROP", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))
from src.data.dataset import pil_exact, pil_exact_gpu as G  # noqa: E402

shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
B = int(sys.argv[2]) if len(sys.argv) > 2 else 128
items = []
with tarfile.open(f"/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_{shard:05d}.tar") as t:
    for m in t:
        if not m.name.endswith((".jpg", ".png")):
            continue
        random.seed(len(items))
        items.append(pil_exact.plan_region_crop(Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB"), 256))
        if len(items) == B:
            break
job = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in G.pack(items).items()}
generic = {k: v for k, v in job.items() if k != "sel"}   # run() falls back to the generic kernel without "sel"


def timed(j, n=50, warm=10):
    for _ in range(warm):
        G.run(j)
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        G.run(j)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


same = torch.equal(G.run(job), G.run(generic))
tg, ts = timed(generic), timed(job)
nv = [job["sel_slices"][p][0][1] for p in range(2)]
print(f"shard {shard}, {B} crops (vertical first in pass 1: {nv[0]}), K {job['K']}: both passes {tg:.3f} ms generic, "
      f"{ts:.3f} ms split (x{tg / ts:.1f}); equal bit for bit: {same}")
print("RESIZE_BENCH " + ("PASS" if same else "FAIL"))
sys.exit(0 if same else 1)
