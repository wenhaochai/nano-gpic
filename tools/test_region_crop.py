#!/usr/bin/env python3
"""random_crop_arr_region against random_crop_arr on real images of one GPIC shard, on one thread: speed, the RNG state after all draws, the values that differ, and hard shapes.
usage: test_region_crop.py <images>     Run on a vis node from baselines/PixelGen with PYTHONPATH=. and OMP_NUM_THREADS=1."""
import io
import random
import sys
import tarfile
import time

import numpy as np
from PIL import Image

from src.data.dataset.augmentation import random_crop_arr, random_crop_arr_region

N = int(sys.argv[1])
raw = []
with tarfile.open("/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_00000.tar") as tar:
    for m in tar:
        if m.name.endswith(".jpg"):
            raw.append(tar.extractfile(m).read())
            if len(raw) == N:
                break
imgs = []
for b in raw:
    im = Image.open(io.BytesIO(b))
    im.load()
    imgs.append(im.convert("RGB"))


def compare(images, seed):
    random.seed(seed)
    t0 = time.perf_counter()
    a = [np.asarray(random_crop_arr(im, 256)) for im in images]
    t_a = time.perf_counter() - t0
    state = random.getstate()
    random.seed(seed)
    t0 = time.perf_counter()
    b = [random_crop_arr_region(im, 256) for im in images]
    t_b = time.perf_counter() - t0
    assert state == random.getstate(), "the random draws differ"
    assert all(x.shape == y.shape == (256, 256, 3) and y.dtype == np.uint8 for x, y in zip(a, b))
    px = sum(int((x != y).sum()) for x, y in zip(a, b))
    mx = max(int(np.abs(x.astype(int) - y.astype(int)).max()) for x, y in zip(a, b))
    return 1e3 * t_a / len(images), 1e3 * t_b / len(images), 100 * px / (len(images) * 256 * 256 * 3), mx


for seed in range(3):
    print("real images, seed %d: official %.2f ms, region %.2f ms, values that differ %.4f%%, largest difference %d" % ((seed,) + compare(imgs, seed)), flush=True)
base = imgs[0]
rng = np.random.default_rng(0)
hard = []
for w, h in [(256, 256), (257, 255), (64, 64), (1, 1), (3000, 2000), (2000, 3000), (4000, 83), (83, 4000), (639, 83), (83, 639), (512, 512), (513, 1025), (1025, 513), (288, 288), (320, 320), (321, 700), (700, 321), (257, 1200), (1200, 257)]:
    hard.append(Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)))
    hard.append(base.resize((w, h), Image.BILINEAR))
worst = 0
for seed in range(40):
    _, _, share, mx = compare(hard, seed)
    worst = max(worst, mx)
print("hard shapes: all run, largest difference over 40 seeds", worst, "(noise images test the clip at 0 and 255)")
