#!/usr/bin/env python3
"""What is the distribution at one step, for a real model? No training and no new head: from one x_t, draw many samples of x with ancestral steps of the point model.

The point model gives x_hat = E[x | x_t, y], one image. From x_t at t, one ancestral chain on the 50-step grid gives one sample of the model's p(x | x_t, y):
    x_{s'} = s' x_hat + (1 - s') (rho eps_hat + sqrt(1 - rho^2) xi),   eps_hat = (x_s - s x_hat) / (1 - s),   rho^2 = lambda(s) / lambda(s'),   lambda(s) = (s / (1 - s))^2.
This is DDPM's posterior step on the flow's path, and it ends at s' = 1 with x = x_hat. It makes no change to the model.

Over images and noise, for an exact model: E ||x - x_hat||^2 = E Var[x | x_t, y], so the real error equals the spread of the samples. If the real error is above the spread, the model does not know how wrong it is.
For each t: err = rms(x_hat - x), spread = rms of the per-pixel std of the K samples, mean_gap = rms(mean of the samples - x_hat) (a check of the chains; small K adds spread / sqrt(K)).

usage: posterior_mc.py <checkpoint> <out dir> <shard index> <shards> <images> <K>     Run from baselines/PixelGen with PYTHONPATH=. and one visible GPU.
"""
import io
import json
import os
import sys
import tarfile

import numpy as np
import torch
from PIL import Image

from src.data.dataset.augmentation import center_crop_arr
from src.models.transformer.JiT_T2I import JiT_T2I

CKPT, OUT, SHARD, SHARDS, IMAGES, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
WC = "/scratch/gpfs/GROUP/USER"
TS = [0.02, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
DEMOS = 16   # the first images are also kept as pictures
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(1000 + SHARD)
dev = "cuda"

sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
sd = {k[len("ema_denoiser."):]: v for k, v in sd.items() if k.startswith("ema_denoiser.")}
net = JiT_T2I(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256)
missing, unexpected = net.load_state_dict(sd, strict=False)
assert not unexpected and all("pos_embed" in m for m in missing), (missing, unexpected)
net = net.to(dev).eval().requires_grad_(False)

shard = "gpic_train_00000"
with np.load(f"{WC}/data/gpic/train_emb_qwen3_1.7b/{shard}.idx.npz") as z:
    index = dict(zip(z["keys"].tolist(), zip(z["offsets"].tolist(), z["lengths"].tolist())))
fd = os.open(f"{WC}/data/gpic/train_emb_qwen3_1.7b/{shard}.emb", os.O_RDONLY)
samples, seen = [], {}
with tarfile.open(f"{WC}/data/gpic/train/{shard}.tar") as tar:
    for m in tar:
        key, _, ext = m.name.rpartition(".")
        if ext in ("jpg", "json"):
            seen.setdefault(key, {})[ext] = tar.extractfile(m).read()
            if len(seen[key]) == 2 and key in index:
                samples.append((key, seen.pop(key)))
                if len(samples) == IMAGES:
                    break


def embed(key):
    """The caption's real tokens, raw bf16 from the cache, and their number."""
    off, n = index[key]
    raw = os.pread(fd, n * 4096, off * 4096)
    assert len(raw) == n * 4096
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(n, 2048), n


def jpeg(a):
    """[3, H, W] in [-1, 1] to a uint8 HWC array."""
    return ((a.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def predict(x, s, y, mask):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return net(x, torch.full((x.shape[0],), s, device=dev, dtype=torch.float32), y, y_mask=mask).float()


@torch.no_grad()
def chain(x_t, t, y, mask):
    """K ancestral chains from one x_t, on the 50-step grid."""
    x = x_t.repeat(K, 1, 1, 1)
    yy, mm = y.repeat(K, 1, 1), mask.repeat(K, 1)
    grid = [t + i / 50 for i in range(round((1 - t) * 50) + 1)]
    grid[-1] = 1.0
    lam = lambda s: (s / (1 - s)) ** 2
    for s, s2 in zip(grid[:-1], grid[1:]):
        hat = predict(x, s, yy, mm)
        if s2 >= 1:
            x = hat
            break
        eps = (x - s * hat) / (1 - s)
        rho2 = lam(s) / lam(s2)
        x = s2 * hat + (1 - s2) * (rho2 ** 0.5 * eps + (1 - rho2) ** 0.5 * torch.randn_like(x))
    return x


rows = {t: dict(err2=0.0, spread2=0.0, gap2=0.0, n=0) for t in TS}
demos = []
for i, (key, parts) in enumerate(samples):
    if i % SHARDS != SHARD:
        continue
    arr = center_crop_arr(Image.open(io.BytesIO(parts["jpg"])).convert("RGB"), 256)
    x = torch.from_numpy(np.asarray(arr).copy()).permute(2, 0, 1).float().div(127.5).sub(1).to(dev)[None]
    emb, n = embed(key)
    y = emb[None].to(dev)
    mask = torch.ones(1, n, dtype=torch.bool, device=dev)
    meta = json.loads(parts["json"])
    demo = dict(key=key, caption=meta.get("caption"), caption_type=meta.get("caption_type"), license=meta.get("license"), attribution=meta.get("attribution"), image=jpeg(x[0]), levels={})
    for t in TS:
        eps = torch.randn_like(x)
        x_t = t * x + (1 - t) * eps
        hat = predict(x_t, t, y, mask)
        s = chain(x_t, t, y, mask)
        var = s.var(0, unbiased=True)
        r = rows[t]
        r["err2"] += (hat - x).pow(2).mean().item()
        r["spread2"] += var.mean().item()
        r["gap2"] += (s.mean(0, keepdim=True) - hat).pow(2).mean().item()
        r["n"] += 1
        if i < DEMOS:
            mean = s.mean(0)
            std = var.mean(0).sqrt()
            # the top directions of the samples: what varies together
            c = (s - mean).flatten(1)
            _, sv, vh = torch.linalg.svd(c.float(), full_matrices=False)
            pcs = vh[:2].view(2, 3, 256, 256) * sv[:2, None, None, None] / (K - 1) ** 0.5
            demo["levels"][t] = dict(x_t=jpeg(x_t[0]), hat=jpeg(hat[0]), mean=jpeg(mean), std=std.cpu().numpy().astype(np.float32), samples=np.stack([jpeg(a) for a in s[:8]]),
                                     pc_plus=np.stack([jpeg(mean + 2 * p) for p in pcs]), pc_minus=np.stack([jpeg(mean - 2 * p) for p in pcs]),
                                     share=(sv[:8].pow(2) / sv.pow(2).sum()).cpu().numpy().astype(np.float32), err=(hat - x).pow(2).mean().sqrt().item(), spread=var.mean().sqrt().item())
    if i < DEMOS:
        demos.append(demo)
    print(i, key, flush=True)
json.dump({str(t): r for t, r in rows.items()}, open(f"{OUT}/shard{SHARD}.json", "w"))
torch.save(demos, f"{OUT}/demos{SHARD}.pt")
print("done shard", SHARD, flush=True)
