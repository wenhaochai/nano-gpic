"""tools/dino_gpu_resize.py (gpic-28 2026-11-09, the optimization scan): an exact GPU copy of gpic_eval's _dino_transform for 256x256 uint8 images: Pillow's 8-bit bicubic resize to 224x224 (horizontal
pass, then vertical, each clip8((2^21 + sum in * kk) >> 22) with Pillow's integer coefficients from pil_exact.coeffs), then
ToTensor (float32 / 255) and Normalize (float32 sub, div). The sums are integers below 2^53, so float64 matmuls give them exactly."""
import math

import numpy as np
import torch

# Pillow's coefficients: precompute_coeffs + normalize_coeffs_8bpc, verbatim from src/data/dataset/pil_exact.py (MFU lever C)
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


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_W = {}


def _weights(n_in, n_out, device):
    key = (n_in, n_out, str(device))
    if key not in _W:
        xmin, xlen, kk = coeffs(n_in, 0.0, float(n_in), n_out)
        W = np.zeros((n_out, n_in), dtype=np.float64)
        for o in range(n_out):
            for j in range(int(xlen[o])):
                W[o, xmin[o] + j] += kk[o, j]
        _W[key] = torch.from_numpy(W).to(device)
    return _W[key]


def _clip8(s):
    return torch.floor((s + 2.0 ** 21) / 2.0 ** 22).clamp_(0, 255)


def preprocess(x, size=224):
    """x: [B, H, W, 3] uint8 (any device) -> [B, 3, size, size] float32, as _dino_transform image by image."""
    B, H, Wd, C = x.shape
    Wh, Wv = _weights(Wd, size, x.device), _weights(H, size, x.device)
    a = x.to(torch.float64)
    a = _clip8(torch.einsum("bhwc,ow->bhoc", a, Wh))           # horizontal pass: [B, H, size, C], uint8 values
    a = _clip8(torch.einsum("bhwc,oh->bowc", a, Wv))           # vertical pass: [B, size, size, C]
    lut = _lut(x.device)                                         # [3, 256]: ToTensor + Normalize of each value, made by torchvision on the CPU
    v = a.to(torch.long).permute(0, 3, 1, 2)                     # [B, 3, size, size]
    return torch.stack([lut[c][v[:, c]] for c in range(3)], dim=1)


_L = {}


def _lut(device):
    """CUDA divides by a Python scalar as a multiply by its reciprocal (1 ulp off the CPU's division), so the float steps are
    a table: torchvision's ToTensor + Normalize applied on the CPU to an image holding every value 0..255 in every channel."""
    key = str(device)
    if key not in _L:
        import torchvision.transforms as TF
        from PIL import Image
        img = Image.fromarray(np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2))   # 1 x 256 x 3
        t = TF.Compose([TF.ToTensor(), TF.Normalize(MEAN, STD)])(img)                                  # [3, 1, 256]
        _L[key] = t[:, 0, :].contiguous().to(device)
    return _L[key]
