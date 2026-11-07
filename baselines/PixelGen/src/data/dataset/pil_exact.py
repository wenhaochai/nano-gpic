"""MFU lever C (docs/mfu.md): Pillow's 8-bit separable resample, bit for bit, with the resize work moved to the GPU.

Pillow 13 (src/libImaging/Resample.c) computes each output pixel of one pass as
    clip8((2^21 + sum_x in[xmin + x] * kk[x]) >> 22),  kk = (int)(+-0.5 + w * 2^22),
with double-precision filter weights w normalized per output pixel (precompute_coeffs, normalize_coeffs_8bpc) and int32
sums. The integer part is exact on any device, so the GPU result equals Pillow's whenever the coefficient tables do.
`coeffs` computes them with the same IEEE double operations in the same order (numpy, no FMA), from the box taken as
single-precision floats as Pillow's binding does. `plan_region_crop` runs random_crop_arr_region's logic and random draws
(augmentation.py) but returns its two resize passes instead of running them; the BOX halvings of large images stay in
Pillow on the CPU (they are rare and cheap). `run_passes_np` is the CPU reference; src/data/dataset/pil_exact_gpu.py runs
the same passes on the GPU.
"""
import math
import random

import numpy as np

PRECISION_BITS = 22


def _bicubic(x):
    """Pillow's bicubic_filter (a = -0.5), elementwise, the same operations in the same order."""
    a = -0.5
    x = np.abs(x)
    out = np.zeros_like(x)
    m1 = x < 1.0
    xm = x[m1]
    out[m1] = ((a + 2.0) * xm - (a + 3.0)) * xm * xm + 1
    m2 = (~m1) & (x < 2.0)
    xm = x[m2]
    out[m2] = (((xm - 5) * xm + 8) * xm - 4) * a
    return out


def coeffs(in_size, in0, in1, out_size, support_base=2.0, filt=_bicubic):
    """precompute_coeffs + normalize_coeffs_8bpc. Returns xmin [out] int32, xlen [out] int32, kk [out, ksize] int32."""
    in0, in1 = np.float32(in0), np.float32(in1)
    scale = float(np.float32(in1 - in0)) / out_size          # (double)(in1 - in0) / outSize, in1 - in0 in float
    filterscale = scale if scale >= 1.0 else 1.0
    support = support_base * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    inv = 1.0 / filterscale
    xx = np.arange(out_size, dtype=np.float64)
    center = float(in0) + (xx + 0.5) * scale
    xmin = np.trunc(center - support + 0.5).astype(np.int64)
    xmin = np.maximum(xmin, 0)
    xmax = np.trunc(center + support + 0.5).astype(np.int64)
    xmax = np.minimum(xmax, in_size) - xmin
    x = np.arange(ksize)[None, :]
    valid = x < xmax[:, None]
    arg = ((x + xmin[:, None]).astype(np.float64) - center[:, None] + 0.5) * inv
    w = np.where(valid, filt(arg), 0.0)
    ww = np.zeros(out_size)
    for j in range(ksize):   # the C loop's order: ww += w over x = 0 .. xmax - 1
        ww = ww + np.where(valid[:, j], w[:, j], 0.0)
    k = np.where(valid & (ww != 0.0)[:, None], w / np.where(ww != 0.0, ww, 1.0)[:, None], np.where(valid, w, 0.0))
    scaled = k * (1 << PRECISION_BITS)
    kk = np.where(k < 0, np.trunc(-0.5 + scaled), np.trunc(0.5 + scaled)).astype(np.int64)
    kk = np.where(valid, kk, 0)
    return xmin.astype(np.int32), xmax.astype(np.int32), kk.astype(np.int32)


def run_pass_np(img, axis, xmin, xlen, kk):
    """One pass on an HxWxC uint8 array along axis (1 = horizontal: columns, 0 = vertical: rows)."""
    a = np.moveaxis(img, axis, 0).astype(np.int64)            # [in, other, C]
    out = np.full((len(xmin),) + a.shape[1:], 1 << (PRECISION_BITS - 1), dtype=np.int64)
    for j in range(kk.shape[1]):
        idx = np.minimum(xmin + j, a.shape[0] - 1)
        out += a[idx] * kk[:, j][:, None, None]
    out = np.clip(out >> PRECISION_BITS, 0, 255).astype(np.uint8)
    return np.moveaxis(out, 0, axis)


def plan_region_crop(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """random_crop_arr_region (augmentation.py) up to its two resizes: the same random draws, the same BOX halvings (in
    Pillow), then the source rows it keeps and its two passes. Returns (src uint8 HxWx3, passes [(axis, xmin, xlen, kk),
    (axis, xmin, xlen, kk)]); run_passes on src gives random_crop_arr_region's output bit for bit."""
    from PIL import Image
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    W, H = pil_image.size
    scale = smaller_dim_size / min(W, H)
    out_w, out_h = round(W * scale), round(H * scale)
    crop_y = random.randrange(out_h - image_size + 1)
    crop_x = random.randrange(out_w - image_size + 1)
    sx, sy = W / out_w, H / out_h
    x0, x1 = crop_x * sx, min(W, (crop_x + image_size) * sx)
    y0, y1 = crop_y * sy, min(H, (crop_y + image_size) * sy)
    arr = np.asarray(pil_image, dtype=np.uint8)
    if (H - out_h) > 0 and (H - out_h) > (W - out_w) * 2:
        # vertical pass on the whole image (box y0..y1), then horizontal on the 256 rows (box x0..x1)
        v = coeffs(H, y0, y1, image_size)
        h = coeffs(W, x0, x1, image_size)
        return arr, [(0,) + v, (1,) + h]
    support = 2 * max(1.0, sy)
    r0, r1 = max(0, int(y0 - support) - 1), min(H, int(y1 + support) + 2)
    h = coeffs(W, x0, x1, image_size)
    v = coeffs(r1 - r0, y0 - r0, y1 - r0, image_size)
    return arr[r0:r1], [(1,) + h, (0,) + v]


def run_passes_np(src, passes):
    out = src
    for axis, xmin, xlen, kk in passes:
        out = run_pass_np(out, axis, xmin, xlen, kk)
    return out
