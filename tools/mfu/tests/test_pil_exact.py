#!/usr/bin/env python3
"""MFU lever C, CPU test: plan_region_crop + run_passes_np (src/data/dataset/pil_exact.py) against random_crop_arr_region
(Pillow) on real GPIC images: the same output bit for bit and the same random state after each image. Also reports the
share of images with BOX halvings, the passes' kernel sizes and the CPU time of both.
usage (vis node): python test_pil_exact.py [n images] [shard ...]"""
import io, os, random, sys, tarfile, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
from src.data.dataset.augmentation import random_crop_arr_region
from src.data.dataset.pil_exact import plan_region_crop, run_passes_np


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    shards = sys.argv[2:] or ["00011", "00412", "00777"]
    done = bad = boxed = 0
    ks = []
    t_pil = t_plan = t_run = 0.0
    for sh in shards:
        with tarfile.open(f"/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_{sh}.tar") as t:
            for m in t:
                if not m.name.endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                img = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
                seed = hash(m.name) & 0xffffffff
                random.seed(seed)
                a = time.perf_counter()
                ref = random_crop_arr_region(img, 256)
                b = time.perf_counter()
                st_ref = random.getstate()
                random.seed(seed)
                src, passes = plan_region_crop(img, 256)
                c = time.perf_counter()
                got = run_passes_np(np.ascontiguousarray(src), passes)
                d = time.perf_counter()
                t_pil, t_plan, t_run = t_pil + b - a, t_plan + c - b, t_run + d - c
                if random.getstate() != st_ref or got.shape != ref.shape or not np.array_equal(got, ref):
                    bad += 1
                    if bad <= 5:
                        diff = np.abs(got.astype(int) - ref.astype(int)) if got.shape == ref.shape else None
                        print(f"MISMATCH {m.name} size {img.size}: shapes {got.shape} {ref.shape}, "
                              f"{'' if diff is None else f'{(diff > 0).mean():.4%} differ, max {diff.max()}'}", flush=True)
                boxed += min(img.size) >= 512
                ks += [p[3].shape[1] for p in passes]
                done += 1
                if done % (n // len(shards)) == 0:
                    break
    print(f"{'ok' if bad == 0 else 'FAIL'}: {done} images, {bad} mismatches; images with a BOX halving (min side >= 512): "
          f"{boxed / done:.1%}; kernel sizes {sorted(set(ks))}; ms per image: Pillow crop {1e3 * t_pil / done:.2f}, "
          f"plan (coefficients) {1e3 * t_plan / done:.2f}, numpy passes {1e3 * t_run / done:.2f}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
