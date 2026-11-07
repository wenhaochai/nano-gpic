#!/usr/bin/env python3
"""MFU lever C: the batched Triton passes (src/data/dataset/pil_exact_gpu.py) against the numpy reference and Pillow, on
real images. On a CPU node it runs Triton's interpreter (TRITON_INTERPRET=1, set here when no GPU is visible); on a GPU
it runs the compiled kernel. Exact equality required. usage: python test_pil_exact_gpu.py [n images] [batch]"""
import io, os, random, sys, tarfile
import torch
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
from src.data.dataset.augmentation import random_crop_arr_region
from src.data.dataset.pil_exact import plan_region_crop, run_passes_np
from src.data.dataset import pil_exact_gpu as G


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items, refs = [], []
    with tarfile.open("/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_00123.tar") as t:
        for m in t:
            if not m.name.endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            img = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
            seed = len(items) * 7919 + 1
            random.seed(seed)
            refs.append(random_crop_arr_region(img, 256))
            random.seed(seed)
            items.append(plan_region_crop(img, 256))
            if len(items) >= n:
                break
    bad = 0
    for i in range(0, n, bs):
        chunk = items[i:i + bs]
        job = G.pack(chunk)
        job = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in job.items()}
        out = G.run(job).cpu().numpy()
        for j, (src, ps) in enumerate(chunk):
            ref_np = run_passes_np(np.ascontiguousarray(src), ps)
            ok = np.array_equal(out[j], refs[i + j]) and np.array_equal(ref_np, refs[i + j])
            bad += not ok
            if not ok and bad <= 3:
                d = np.abs(out[j].astype(int) - refs[i + j].astype(int))
                print(f"MISMATCH sample {i + j}: {(d > 0).mean():.3%} differ, max {d.max()}", flush=True)
    print(f"{'ok' if bad == 0 else 'FAIL'} ({dev}{', Triton interpreter' if dev == 'cpu' else ''}): {n} images in batches "
          f"of {bs}, {bad} mismatches against Pillow")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
