#!/usr/bin/env python3
"""Overfit test of the autoregressive model on the real backbone (d640, 9 + 2 blocks), on one GPU, with no data workers and no Lightning; the model and the trainer are the real ones. Unconditional (the null caption), N real images. Same images and metrics as tools/dist_step/overfit_code.py, so the point model's numbers of w1095 are the reference.

  nn_rms    root mean square per pixel from each sample to its nearest training image, in [-1, 1] units
  near      share of samples with nn_rms below 0.1
  covered   share of training images that are the nearest of a sample with nn_rms below 0.1
  distinct  share of training images that are the nearest of any sample
  late      the same distances on the last quarter of the patches (rows 192 to 255), where the image is already decided: nn_late to the nearest training image of the whole sample, tf_late when each patch is drawn given the true patches before it (teacher forcing). tf_late far below nn_late means that errors grow along the chain. The reverse does not hold on early patches: a good model may draw another image's patch there, so only the late rows are read this way.
Samples: 128 at the checks, 8 N (at least 64) at the last step. Raw weights.
Rescue, at the last step: the first k true patches of a training image are given, and the chain draws the rest. late_source is the rms on the last quarter of the rows to that image, late_nearest to the nearest training image of the whole result, and late_ok the share of samples with late_nearest below 0.1 (the distance has two modes, so this share is steadier than a mean of 128). head_steps_128 repeats k = 0 with four times the head's steps. The reading is in docs/autoregressive.md.
row_rms, at the last step: the rms of each of the 16 patch rows to the nearest training image of the whole sample, and row_ok the share under 0.1. High from row 0: blending starts at the first draws. Low and then rising: the chain drifts.
The final weights are saved as <name>.pt, for later tests of the sampler.
usage: overfit_ar.py <N> <out dir> [steps] [seed] [--context-noise S] [--head-width W] [--head-depth D] [--name TAG]    Run from baselines/PixelGen with PYTHONPATH=. and one visible GPU.
"""
import argparse
import glob
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
import torch
from PIL import Image

from src.data.dataset.augmentation import center_crop_arr
from src.diffusion.flow_matching.training_ar import ARTrainer
from src.models.transformer.AR_JiT import ARJiT

ap = argparse.ArgumentParser()
ap.add_argument("N", type=int)
ap.add_argument("out")
ap.add_argument("steps", type=int, nargs="?", default=24000)
ap.add_argument("seed", type=int, nargs="?", default=0)
ap.add_argument("--context-noise", type=float, default=0.0)
ap.add_argument("--head-width", type=int, default=1024)
ap.add_argument("--head-depth", type=int, default=3)
ap.add_argument("--name", default=None)
ap.add_argument("--order", default="raster", choices=["raster", "random"])
ap.add_argument("--few-known", type=float, default=0.0)
ARGS = ap.parse_args()
N, OUT, STEPS, SEED = ARGS.N, ARGS.out, ARGS.steps, ARGS.seed
CHECKS = sorted({s for s in (1000, 4000, 8000, 16000, 24000, 32000, 40000) if s < STEPS} | {STEPS})
WC = "/scratch/gpfs/GROUP/USER"
TAG = ARGS.name or f"n{N}" + (f"_s{SEED}" if SEED else "")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)
dev = "cuda"


def load_images(n):
    out = []
    with tarfile.open(f"{WC}/data/gpic/train/gpic_train_00000.tar") as tar:
        for m in tar:
            if m.name.endswith(".jpg"):
                arr = center_crop_arr(Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB"), 256)
                out.append(torch.from_numpy(np.asarray(arr).copy()).permute(2, 0, 1).float() / 127.5 - 1)
                if len(out) == n:
                    break
    return torch.stack(out).to(dev)


imgs = load_images(N)
assert (torch.nn.functional.unfold(imgs[:1], kernel_size=16, stride=16)[0, :, 0] == imgs[0, :, :16, :16].flatten()).all(), "first_patches relies on this order"
null = torch.load(glob.glob(f"{WC}/data/gpic/train_emb_qwen3_1.7b/null.pt")[0], map_location=dev, weights_only=True).float()
B, T = 32, 8
y = torch.zeros(B, T, 2048, device=dev, dtype=torch.bfloat16)
y[:, 0] = null[0, 0].bfloat16()
mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
mask[:, 0] = True
net = ARJiT(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256, head_width=ARGS.head_width, head_depth=ARGS.head_depth, head_steps=32, context_noise=ARGS.context_noise, order=ARGS.order).to(dev)
net.compile()
tr = ARTrainer(P_mean=-0.8, P_std=1.6, t_eps=0.05, few_known_p=ARGS.few_known, null_condition_p=0.0).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, fused=True)


def measure(g):
    d = torch.cat([torch.cdist(g[i:i + 64].flatten(1), imgs.flatten(1)) for i in range(0, len(g), 64)]) / (3 * 256 * 256) ** 0.5
    nn, idx = d.min(1)
    late = ((g[:, :, 192:] - imgs[idx][:, :, 192:]) ** 2).mean(dim=(1, 2, 3)).sqrt()
    rows_rms = ((g - imgs[idx]) ** 2).mean(dim=(1, 3))   # [samples, 256 pixel rows]
    rows_rms = rows_rms.unfold(1, 16, 16).mean(-1).sqrt()   # [samples, 16 patch rows]
    return dict(nn_rms=nn.mean().item(), nn_rms_p90=nn.quantile(0.9).item(), near=(nn < 0.1).float().mean().item(), covered=idx[nn < 0.1].unique().numel() / N, distinct=idx.unique().numel() / N, pixel_std=g.std(0).mean().item(), nn_late=late.mean().item(), row_rms=[round(v, 4) for v in rows_rms.mean(0).tolist()], row_ok=[round(v, 3) for v in (rows_rms < 0.1).float().mean(0).tolist()])


@torch.no_grad()
def sample(n):
    res = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(0, n, B):
            res.append(net.ar_sample(y, mask).float())
    return torch.cat(res)[:n]


@torch.no_grad()
def teacher_forced():
    """Each patch drawn given the true patches before it: the chain with no chance for errors to grow."""
    x = imgs[torch.randint(N, (B,), device=dev)]
    tokens = torch.nn.functional.unfold(x, kernel_size=16, stride=16).transpose(1, 2)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        context = tokens + net.context_noise * torch.randn_like(tokens)
        kw = {}
        if net.order == "random":   # 30% known, the lowest masked share in training; the late rows are read on the masked tokens
            known = torch.rand(B, net.num_patches, device=dev) < 0.3
            kw["known"] = known
        z = net(context, y, y_mask=mask, **kw).float()
        drawn = net.draw_patch(z.reshape(-1, z.shape[-1])).view(tokens.shape)
        if net.order == "random":
            w = (~known[:, 192:, None]).float()
            return (((drawn[:, 192:] - tokens[:, 192:]) ** 2 * w).sum(dim=(1, 2)) / (w.sum(dim=(1, 2)) * 768).clamp_min(1)).sqrt().mean().item()
    return ((drawn[:, 192:] - tokens[:, 192:]) ** 2).mean(dim=(1, 2)).sqrt().mean().item()


@torch.no_grad()
def rescue(k, batches=4):
    """The chain from k true patches: rms on the last quarter of the rows to the source image, to the nearest training image of the whole result, and the share of the latter under 0.1."""
    src, nearest = [], []
    for _ in range(batches):
        x = imgs[torch.randint(N, (B,), device=dev)]
        tokens = torch.nn.functional.unfold(x, kernel_size=16, stride=16).transpose(1, 2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = net.ar_sample(y, mask, prefix=tokens[:, :k]).float()
        src.append(((g[:, :, 192:] - x[:, :, 192:]) ** 2).mean(dim=(1, 2, 3)).sqrt())
        idx = torch.cdist(g.flatten(1), imgs.flatten(1)).argmin(1)
        nearest.append(((g[:, :, 192:] - imgs[idx][:, :, 192:]) ** 2).mean(dim=(1, 2, 3)).sqrt())
    near_all = torch.cat(nearest)
    return torch.cat(src).mean().item(), near_all.mean().item(), (near_all < 0.1).float().mean().item()


@torch.no_grad()
def first_patches():
    """Each training image's first patch against draws of the first patch: is the first draw on a true first patch?"""
    true = imgs[:, :, :16, :16].flatten(1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        kw = {"known": torch.zeros(B, net.num_patches, dtype=torch.bool, device=dev)} if net.order == "random" else {}
        z = net(torch.zeros(B, net.num_patches, net.patch_dim, device=dev), y, y_mask=mask, **kw)[:, 0].float()
        draws = torch.cat([net.draw_patch(z) for _ in range(16)])
    d = torch.cdist(draws, true).min(1)[0] / 768 ** 0.5
    between = torch.cdist(true, true)
    between.fill_diagonal_(float("inf"))
    return dict(first_draw_rms=d.mean().item(), first_draw_near=(d < 0.1).float().mean().item(), first_true_gap=(between.min(1)[0] / 768 ** 0.5).median().item())


rows = []
t0 = time.time()
ema_fm = None
for step in range(1, STEPS + 1):
    for group in opt.param_groups:
        group["lr"] = 1e-4 * min(1.0, step / 500)
    x = imgs[torch.randint(N, (B,), device=dev)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = tr._impl_trainstep(net, None, None, x, y, {"y_mask": mask})
    res["loss"].backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, foreach=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 50 == 0:
        fm = res["fm_loss"].item()
        ema_fm = fm if ema_fm is None else 0.9 * ema_fm + 0.1 * fm
    if step in CHECKS:
        train_s = time.time() - t0
        g = sample(max(64, 8 * N) if step == STEPS else 128)
        row = dict(step=step, seconds=train_s, sample_s=time.time() - t0 - train_s, fm=ema_fm, tf_late=teacher_forced(), **measure(g))
        rows.append(row)
        json.dump(rows, open(f"{OUT}/{TAG}.json", "w"))
        print(TAG, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if not isinstance(v, list)}, flush=True)
diag = dict(first_patches())
for k in (0, 1, 4, 16, 64, 128, 192):
    diag[f"late_source_k{k}"], diag[f"late_nearest_k{k}"], diag[f"late_ok_k{k}"] = rescue(k)
net.head_steps = 128
diag["late_source_k0_head_steps_128"], diag["late_nearest_k0_head_steps_128"], diag["late_ok_k0_head_steps_128"] = rescue(0, batches=2)
net.head_steps = 32
json.dump(diag, open(f"{OUT}/{TAG}_rescue.json", "w"), indent=1)
print(TAG, "rescue", {k: round(v, 4) for k, v in diag.items()}, flush=True)
torch.save({k: v.cpu() for k, v in net.state_dict().items()}, f"{OUT}/{TAG}.pt")
g = sample(32)[:16]
grid = ((g.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
Image.fromarray(np.concatenate([np.concatenate(list(grid[i:i + 8]), 1) for i in (0, 8)], 0)).save(f"{OUT}/{TAG}.png")
print(TAG, "done", round(time.time() - t0), "s", flush=True)
