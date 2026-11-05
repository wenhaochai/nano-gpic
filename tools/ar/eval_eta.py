#!/usr/bin/env python3
"""The mask-state jump process with stochasticity eta on saved weights of tools/ar/overfit_ar.py, no training: AR_JiT.sample_eta. Metrics as in tools/ar/refine_ar.py, plus effective images, exp(entropy) of the samples per training image.
usage: eval_eta.py <weights .pt> <N> <out .json> <eta> <sampling seed> [--head-depth D] [--head-width W] [--samples S] [--ar-steps K]    Run from baselines/PixelGen with PYTHONPATH=. and one visible GPU.
"""
import argparse
import io
import json
import tarfile
import time

import numpy as np
import torch
from PIL import Image

from src.data.dataset.augmentation import center_crop_arr
from src.models.transformer.AR_JiT import ARJiT

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("N", type=int)
ap.add_argument("out")
ap.add_argument("eta", type=float)
ap.add_argument("seed", type=int)
ap.add_argument("--head-depth", type=int, default=3)
ap.add_argument("--head-width", type=int, default=1024)
ap.add_argument("--samples", type=int, default=None)
ap.add_argument("--ar-steps", type=int, default=64)
A = ap.parse_args()
N = A.N
SAMPLES = A.samples or 8 * N
WC = "/scratch/gpfs/GROUP/USER"
dev = "cuda"
torch.manual_seed(A.seed)
imgs = []
with tarfile.open(f"{WC}/data/gpic/train/gpic_train_00000.tar") as tar:
    for m in tar:
        if m.name.endswith(".jpg"):
            arr = center_crop_arr(Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB"), 256)
            imgs.append(torch.from_numpy(np.asarray(arr).copy()).permute(2, 0, 1).float() / 127.5 - 1)
            if len(imgs) == N:
                break
imgs = torch.stack(imgs).to(dev)
unfold = lambda g: torch.nn.functional.unfold(g, kernel_size=16, stride=16).transpose(1, 2)
true = unfold(imgs)
null = torch.load(f"{WC}/data/gpic/train_emb_qwen3_1.7b/null.pt", map_location=dev, weights_only=True).float()
B, T = 32, 8
y = torch.zeros(B, T, 2048, device=dev, dtype=torch.bfloat16)
y[:, 0] = null[0, 0].bfloat16()
mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
mask[:, 0] = True
sd = torch.load(A.weights, map_location="cpu", weights_only=True)
assert any("mask_token" in k for k in sd), "this script is for the random order"
net = ARJiT(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256, head_width=A.head_width, head_depth=A.head_depth, head_steps=32, order="random", ar_steps=A.ar_steps)
net.load_state_dict(sd, strict=True)
net = net.to(dev).eval().requires_grad_(False)
net.compile()
t0 = time.time()
gen, draws = [], []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(0, SAMPLES, B):
        g, d = net.sample_eta(y, mask, eta=A.eta)
        gen.append(g.float())
        draws.append(d)
gen = torch.cat(gen)[:SAMPLES]
sample_s = time.time() - t0
d = torch.cat([torch.cdist(gen[i:i + 64].flatten(1), imgs.flatten(1)) for i in range(0, len(gen), 64)]) / (3 * 256 * 256) ** 0.5
nn, idx = d.min(1)
near = nn < 0.1
tokens = unfold(gen)
best, who = [], []
for i in range(0, len(tokens), 32):
    per = ((tokens[i:i + 32, None] - true[None]) ** 2).mean(-1).sqrt()   # [s, N, 256]
    share, at = (per < 0.2).float().mean(-1).max(dim=1)
    best.append(share)
    who.append(at)
best, who = torch.cat(best), torch.cat(who)
ok = best >= 0.9


def effective(labels):
    c = torch.bincount(labels, minlength=N).float()
    q = c[c > 0] / c.sum()
    return (-(q * q.log()).sum()).exp().item()


res = dict(eta=A.eta, ar_steps=A.ar_steps, seed=A.seed, samples=len(gen), sample_s=sample_s, head_draws_per_image=float(np.mean(draws)), nn_rms=nn.mean().item(), near=near.float().mean().item(), covered=idx[near].unique().numel() / N, distinct=idx.unique().numel() / N, effective_images=effective(idx), never_drawn=1 - idx.unique().numel() / N, best_share=best.mean().item(), near_robust=ok.float().mean().item(), covered_robust=who[ok].unique().numel() / N, stray=(1 - best[ok]).mean().item() if ok.any() else None)
json.dump(res, open(A.out, "w"), indent=1)
torch.save(dict(nn=nn.cpu(), idx=idx.cpu(), best=best.cpu(), who=who.cpu()), A.out.replace(".json", "_per_sample.pt"))
print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
