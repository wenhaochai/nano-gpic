#!/usr/bin/env python3
"""MFU lever P (GPIC_MFU_PATCHIFY=1, src/models/transformer/JiT_T2I.py patchify / unpatchify) against F.unfold / F.fold:
values bit for bit, the patchified tensor's strides, and the gradients of both directions, in fp32 and bf16, for square
and non-square images. CPU, or a GPU when there is one (CUDA's per-sample im2col is the kernel the lever replaces).
usage: python test_patchify.py"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.models.transformer.JiT_T2I import patchify, unpatchify  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
ok = True
for dt in (torch.float32, torch.bfloat16):
    for (B, C, H, W, p) in [(4, 3, 256, 256, 16), (2, 3, 256, 192, 16), (3, 5, 64, 96, 8)]:
        g = torch.Generator(device="cpu").manual_seed(B * H + W)
        x = torch.randn(B, C, H, W, generator=g).to(dev, dt).requires_grad_()
        x2 = x.detach().clone().requires_grad_()
        a = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)
        b = patchify(x2, p)
        same_fwd = torch.equal(a, b) and a.stride() == b.stride() and a.shape == b.shape
        u = torch.randn(a.shape, generator=g).to(dev, dt)
        (a * u).sum().backward()
        (b * u).sum().backward()
        same_bwd = torch.equal(x.grad, x2.grad)

        z = torch.randn(B, (H // p) * (W // p), C * p * p, generator=g).to(dev, dt).requires_grad_()
        z2 = z.detach().clone().requires_grad_()
        f = F.fold(z.transpose(1, 2).contiguous(), output_size=(H, W), kernel_size=p, stride=p)
        h = unpatchify(z2, H, W, p)
        same_fold = torch.equal(f, h) and h.is_contiguous() and f.shape == h.shape
        v = torch.randn(f.shape, generator=g).to(dev, dt)
        (f * v).sum().backward()
        (h * v).sum().backward()
        same_fold_bwd = torch.equal(z.grad, z2.grad)
        good = same_fwd and same_bwd and same_fold and same_fold_bwd
        ok &= good
        print(f"{dev} {str(dt):15s} B{B} C{C} {H}x{W} p{p}: patchify values+strides {same_fwd}, its gradient {same_bwd}; "
              f"unpatchify values {same_fold}, its gradient {same_fold_bwd}", flush=True)
print("PATCHIFY_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
