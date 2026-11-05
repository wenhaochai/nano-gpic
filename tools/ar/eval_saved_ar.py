#!/usr/bin/env python3
"""Sampler settings on saved weights of tools/ar/overfit_ar.py, with no training (w1133 saved them). The first metrics are the overfit test's own. The robust ones explain them: in w1133 the MAR-order samples look like one training image with a few stray patches, which raise an rms over the whole image.
  best_share   for each sample, the largest share of its 256 patches within rms 0.2 of one training image. Different images' patches at one position lie much further apart (median rms 0.66, 10th percentile 0.34), and a collage of two images gets a low share, which a median would hide (review 2026-11-04)
  near_robust  share of samples with best_share >= 0.9 (at most 25 stray patches); covered_robust likewise, as covered
  stray        over those samples: 1 - best_share
usage: eval_saved_ar.py <weights .pt> <N> <out .json> <ar_steps> <temperature> [samples]    Run from baselines/PixelGen with PYTHONPATH=. and one visible GPU.
"""
import io
import json
import sys
import tarfile
import time

import numpy as np
import torch
from PIL import Image

from src.data.dataset.augmentation import center_crop_arr
from src.models.transformer.AR_JiT import ARJiT

W, N, OUT, STEPS, TEMP = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
SAMPLES = int(sys.argv[6]) if len(sys.argv) > 6 else 8 * N
WC = "/scratch/gpfs/GROUP/USER"
dev = "cuda"
torch.manual_seed(7)
imgs = []
with tarfile.open(f"{WC}/data/gpic/train/gpic_train_00000.tar") as tar:
    for m in tar:
        if m.name.endswith(".jpg"):
            arr = center_crop_arr(Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB"), 256)
            imgs.append(torch.from_numpy(np.asarray(arr).copy()).permute(2, 0, 1).float() / 127.5 - 1)
            if len(imgs) == N:
                break
imgs = torch.stack(imgs).to(dev)
true = torch.nn.functional.unfold(imgs, kernel_size=16, stride=16).transpose(1, 2)   # [N, 256, 768]
null = torch.load(f"{WC}/data/gpic/train_emb_qwen3_1.7b/null.pt", map_location=dev, weights_only=True).float()
B, T = 32, 8
y = torch.zeros(B, T, 2048, device=dev, dtype=torch.bfloat16)
y[:, 0] = null[0, 0].bfloat16()
mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
mask[:, 0] = True
sd = torch.load(W, map_location="cpu", weights_only=True)
order = "random" if any("mask_token" in k for k in sd) else "raster"
net = ARJiT(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256, head_width=1024, head_depth=3, head_steps=32, order=order, ar_steps=STEPS, temperature=TEMP)
# weights of the compiled model carry no prefix when saved from state_dict of the uncompiled module; accept both
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
missing, unexpected = net.load_state_dict(sd, strict=False)
assert not unexpected and all(m in ("pos_embed",) for m in missing), (missing, unexpected)
net = net.to(dev).eval().requires_grad_(False)
net.compile()
t0 = time.time()
gen = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(0, SAMPLES, B):
        gen.append(net.ar_sample(y, mask).float())
gen = torch.cat(gen)[:SAMPLES]
sample_s = time.time() - t0
d = torch.cat([torch.cdist(gen[i:i + 64].flatten(1), imgs.flatten(1)) for i in range(0, len(gen), 64)]) / (3 * 256 * 256) ** 0.5
nn, idx = d.min(1)
res = dict(ar_steps=STEPS, temperature=TEMP, samples=len(gen), sample_s=sample_s, nn_rms=nn.mean().item(), near=(nn < 0.1).float().mean().item(), covered=idx[nn < 0.1].unique().numel() / N, distinct=idx.unique().numel() / N)
patches = torch.nn.functional.unfold(gen, kernel_size=16, stride=16).transpose(1, 2)   # [S, 256, 768]
best, who = [], []
for i in range(0, len(patches), 32):
    per = ((patches[i:i + 32, None] - true[None]) ** 2).mean(-1).sqrt()   # [s, N, 256]
    share, at = (per < 0.2).float().mean(-1).max(dim=1)
    best.append(share)
    who.append(at)
best, who = torch.cat(best), torch.cat(who)
ok = best >= 0.9
res.update(best_share=best.mean().item(), near_robust=ok.float().mean().item(), covered_robust=who[ok].unique().numel() / N, stray=(1 - best[ok]).mean().item() if ok.any() else None, share_quantiles=[round(v, 3) for v in best.quantile(torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev)).tolist()])
json.dump(res, open(OUT, "w"), indent=1)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
