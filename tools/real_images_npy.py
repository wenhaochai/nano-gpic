"""Write N real GPIC images (center-crop 256, the baseline's val_transform crop) to an NHWC uint8 .npy.

Used to sanity-check gpic-eval and the reference stats: real held-out images should score a low FD-DINOv2
against test_stats.npz. Also reports single-process decode+crop throughput (images/s).
usage: python real_images_npy.py <tar_glob> <N> <out.npy>
"""
import glob, io, json, sys, tarfile, time
import numpy as np
from PIL import Image

sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/gpic/baselines/PixelGen")
from src.data.dataset.augmentation import center_crop_arr

pattern, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
arr = np.zeros((n, 256, 256, 3), np.uint8)
i, t0, tdec = 0, time.time(), 0.0
for path in sorted(glob.glob(pattern)):
    with tarfile.open(path) as tar:
        for m in tar:
            if not m.isfile() or m.name.endswith(".json"):
                continue
            raw = tar.extractfile(m).read()
            t1 = time.time()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            arr[i] = center_crop_arr(img, 256)
            tdec += time.time() - t1
            i += 1
            if i == n:
                break
    if i == n:
        break
np.save(out, arr[:i])
print(f"wrote {i} images to {out}; decode+crop {i/tdec:.1f} img/s/process; wall {time.time()-t0:.0f}s")
