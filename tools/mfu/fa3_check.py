#!/usr/bin/env python3
"""Fail-fast check of the FA3 custom ops (src/models/layers/fa3_op.py) on one GPU, about 20 s: forward and backward at one
image-block shape against SDPA's math backend with the same key mask, eager and compiled with a dynamic T. Exit 1 on an
error or an error above 2e-2. Probe jobs run it first, so a broken op stops the job before its GPUs are booked."""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.environ.get("PIXGEN_SRC", os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))
from src.models.layers.fa3_op import fa3_attention

dev = "cuda"
ok = True
fc = torch.compile(lambda q, k, v, s: fa3_attention(q, k, v, s), dynamic=True)
for name, f in (("eager", fa3_attention), ("compiled", fc)):
    for T in (24, 56):
        B, N, H, D = 4, 256, 4, 64
        lens = torch.tensor([T, T - 3, 1, T - 7], device=dev, dtype=torch.int32)
        n_valid = lens + N
        Nk = N + T
        mask = torch.arange(Nk, device=dev)[None] < n_valid[:, None]
        q, k, v = (torch.randn(B, s, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for s in (N, Nk, Nk))
        do = torch.randn(B, N, H, D, device=dev, dtype=torch.bfloat16)
        o = f(q, k, v, n_valid)
        o.backward(do)
        qr, kr, vr = (t.detach().float().transpose(1, 2).requires_grad_() for t in (q, k, v))
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=mask[:, None, None, :])
        ref.backward(do.float().transpose(1, 2))
        errs = [(o.float() - ref.transpose(1, 2)).abs().max().item()] + \
               [(t.grad.float() - r.grad.transpose(1, 2)).abs().max().item() for t, r in ((q, qr), (k, kr), (v, vr))]
        pad = k.grad[~mask].abs().max().item() if (~mask).any() else 0.0
        good = max(errs) < 2e-2 and pad == 0.0
        ok &= good
        print(f"fa3 {name} T={T}: max err out/dq/dk/dv {['%.1e' % e for e in errs]}, padded-key grad {pad:.1e}: {'ok' if good else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
