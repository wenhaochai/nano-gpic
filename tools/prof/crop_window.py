#!/usr/bin/env python3
"""random_crop_arr, as it is and computing only the crop window: the time of each, and whether the pixels are the same.

BICUBIC resampling in PIL costs about the number of output pixels. The official crop resamples the whole image and then cuts 256 x 256
from it. PIL can resample a box of the source, which has the same filter centres up to rounding. usage: crop_window.py <shard.tar>
<images> [more shards...]
Measured on 4000 ImageNet images (Pillow 13.0.0): 3.14 ms -> 1.71 ms, but 3951 images differ (2.2% of channel values by 1, a few by more,
max 23). So this path is not the same pixels, and production does not use it.
"""
import io
import math
import random
import sys
import tarfile
import time

import numpy as np
from PIL import Image


def official(pil_image, image_size=256, min_crop_frac=0.8, max_crop_frac=1.0):
    t = time.perf_counter()
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
    halvings = 0
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
        halvings += 1
    t_box = time.perf_counter() - t
    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    t_bic = time.perf_counter() - t - t_box
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    out = Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
    return out, t_box, t_bic, time.perf_counter() - t, halvings


def windowed(pil_image, image_size=256, min_crop_frac=0.8, max_crop_frac=1.0):
    t = time.perf_counter()
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    W, H = pil_image.size
    scale = smaller_dim_size / min(W, H)
    w, h = round(W * scale), round(H * scale)
    # the same draws in the same order as the official code
    crop_y = random.randrange(h - image_size + 1)
    crop_x = random.randrange(w - image_size + 1)
    sx, sy = W / w, H / h
    box = (crop_x * sx, crop_y * sy, min((crop_x + image_size) * sx, W), min((crop_y + image_size) * sy, H))
    out = pil_image.resize((image_size, image_size), resample=Image.BICUBIC, box=box)
    return out, time.perf_counter() - t


def main():
    shards, n = sys.argv[1:2] + sys.argv[3:], int(sys.argv[2])
    rows, sizes, seen, differ, maxdiff, diffpix = [], [], 0, 0, 0, 0
    for shard in shards:
        with tarfile.open(shard) as tar:
            for m in tar:
                if seen >= n:
                    break
                ext = m.name.rsplit(".", 1)[-1].lower()
                if ext not in ("jpg", "jpeg", "png", "webp"):
                    continue
                data = tar.extractfile(m).read()
                t0 = time.perf_counter()
                im = Image.open(io.BytesIO(data))
                im.load()
                t_decode = time.perf_counter() - t0
                t0 = time.perf_counter()
                rgb = im.convert("RGB")
                t_convert = time.perf_counter() - t0
                seed = seen * 7919 + 13
                random.seed(seed)
                a, t_box, t_bic, t_official, halvings = official(rgb)
                random.seed(seed)
                b, t_window = windowed(rgb)
                t0 = time.perf_counter()
                arr = np.asarray(a, dtype=np.uint8)
                t_asarray = time.perf_counter() - t0
                x, y = arr, np.asarray(b, dtype=np.uint8)
                if not np.array_equal(x, y):
                    differ += 1
                    d = np.abs(x.astype(int) - y.astype(int))
                    maxdiff = max(maxdiff, int(d.max()))
                    diffpix += int((d > 0).sum())
                rows.append((t_decode, t_convert, t_box, t_bic, t_official, t_window, t_asarray, len(data), halvings))
                sizes.append(rgb.size)
                seen += 1
    r = np.array(rows)
    names = ["decode", "convert RGB", "BOX halving", "BICUBIC whole", "official crop", "window crop", "asarray"]
    print(f"{seen} images; {differ} with any different pixel (max difference {maxdiff}, {diffpix} channel values)")
    for i, name in enumerate(names):
        print(f"{name:15s} mean {1e3 * r[:, i].mean():6.3f} ms  median {1e3 * np.median(r[:, i]):6.3f} ms  p90 {1e3 * np.percentile(r[:, i], 90):6.3f} ms")
    print(f"JPEG bytes: mean {r[:, 7].mean() / 1e3:.1f} KB; halvings: {np.bincount(r[:, 8].astype(int)).tolist()}")
    s = np.array(sizes)
    print(f"image width median {np.median(s[:, 0]):.0f}, height median {np.median(s[:, 1]):.0f}, short side p10 {np.percentile(s.min(1), 10):.0f}, p90 {np.percentile(s.min(1), 90):.0f}")


if __name__ == "__main__":
    main()
