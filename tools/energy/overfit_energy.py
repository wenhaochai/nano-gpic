#!/usr/bin/env python3
"""Overfit test E (docs/energy_head.md) on the real model (d640, 9 + 2 blocks), on one GPU, with no data workers and no Lightning. From tools/dist_step/overfit_code.py.

Arms:
  point     the official model and loss (JiTTrainer), sampled with the official Euler grid and t_eps: the control
  energy    JiT_T2I(head_width=352): the backbone runs once per x_t, EnergyHead draws M = 2 samples; ImplicitJiTTrainer (whole-image energy score, beta = 1, lambda = 1)
  implicit  the implicit model already in the repo, JiT_T2I(xi_dim=64, xi_block=6), M = 2, the same loss
energy and implicit sample with ImplicitSampler's exact step and fresh noise at each call. All arms run the same number of steps, so implicit spends about 32% more FLOPs.

Unconditional (the null caption), N real images. Metrics, at 50 and 8 calls, from the raw weights, on 8 N samples (2048 at N = 256):
  nn_rms    root mean square per pixel from each sample to its nearest training image, in [-1, 1] units; mean and 90th percentile
  near      share of samples with nn_rms below 0.1
  covered   share of training images that are the nearest of a sample with nn_rms below 0.1
  distinct  share of training images that are the nearest of any sample (no threshold)
  effective_images  exp(entropy) of the share of samples whose nearest image is each training image (no threshold; N if spread evenly, 1 if all on one image)
At the end, the nearest image and distance of every sample go to <tag>_nearest.pt, and the weights to <tag>.pt.
usage: overfit_energy.py <point|energy|implicit> <N> <out dir> [steps] [seed]    Run from baselines/PixelGen with PYTHONPATH=. and one visible GPU.
"""
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
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.flow_matching.training_implicit import ImplicitJiTTrainer
from src.diffusion.flow_matching.training_JiT import JiTTrainer, constant
from src.models.transformer.JiT_T2I import JiT_T2I

ARM, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
assert ARM in ("point", "energy", "implicit"), ARM
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 24000
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
EVALS = sorted({s for s in (1000, 2000, 4000, 8000, 16000) if s < STEPS} | {STEPS})
WC = "/scratch/gpfs/GROUP/USER"
TAG = f"n{N}_{ARM}_s{SEED}"
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
null = torch.load(glob.glob(f"{WC}/data/gpic/train_emb_qwen3_1.7b/null.pt")[0], map_location=dev, weights_only=True).float()
B, T = 32, 8
y = torch.zeros(B, T, 2048, device=dev, dtype=torch.bfloat16)
y[:, 0] = null[0, 0].bfloat16()
mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
mask[:, 0] = True
kw = {"point": {}, "energy": dict(head_width=352), "implicit": dict(xi_dim=64, xi_block=6)}[ARM]
net = JiT_T2I(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256, **kw).to(dev)
net.compile()
tkw = dict(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05, null_condition_p=0.0)
tr = (JiTTrainer(**tkw) if ARM == "point" else ImplicitJiTTrainer(**tkw, samples=2, beta=1.0)).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, fused=True)


@torch.no_grad()
def sample(n, calls):
    """point: the official Euler step with t_eps. energy, implicit: ImplicitSampler's exact step, no t_eps; the last call returns x_hat."""
    res = []
    ts = torch.cat([torch.linspace(0, 1 - 1 / calls, calls), torch.ones(1)]).to(dev)
    floor = 0.05 if ARM == "point" else 0.0
    for i in range(0, n, B):
        x = torch.randn(B, 3, 256, 256, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for j in range(calls):
                t = ts[j].repeat(B)
                pred = net(x, t, y, y_mask=mask).float()
                x = x + (ts[j + 1] - ts[j]) * (pred - x) / (1 - t.view(-1, 1, 1, 1)).clamp_min(floor)
        res.append(x.float())
    return torch.cat(res)[:n]


def nearest(g):
    d = torch.cat([torch.cdist(g[i:i + 64].flatten(1), imgs.flatten(1)) for i in range(0, len(g), 64)]) / (3 * 256 * 256) ** 0.5
    return d.min(1)


def measure(g):
    nn, idx = nearest(g)
    p = torch.bincount(idx, minlength=N).float() / len(idx)
    entropy = -(p[p > 0] * p[p > 0].log()).sum()
    return dict(nn_rms=nn.mean().item(), nn_rms_p90=nn.quantile(0.9).item(), near=(nn < 0.1).float().mean().item(), covered=idx[nn < 0.1].unique().numel() / N,
                distinct=idx.unique().numel() / N, effective_images=entropy.exp().item(), pixel_std=g.std(0).mean().item())


rows = []
t0 = time.time()
ema = {}
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
        for k in ("loss", "to_data", "between"):
            if k in res:
                v = res[k].item()
                ema[k] = v if k not in ema else 0.9 * ema[k] + 0.1 * v
    if step in EVALS:
        n = max(64, 8 * N)
        row = dict(step=step, seconds=time.time() - t0, **{f"train_{k}": v for k, v in ema.items()})
        g50, g8 = sample(n, 50), sample(n, 8)
        row.update(measure(g50))
        row.update({f"c8_{k}": v for k, v in measure(g8).items()})
        one = measure(sample(64, 1))
        row.update(one_call_nn_rms=one["nn_rms"], one_call_near=one["near"])
        rows.append(row)
        json.dump(rows, open(f"{OUT}/{TAG}.json", "w"))
        print(TAG, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
(d50, i50), (d8, i8) = nearest(g50), nearest(g8)
torch.save(dict(calls50_index=i50.cpu(), calls50_dist=d50.cpu(), calls8_index=i8.cpu(), calls8_dist=d8.cpu()), f"{OUT}/{TAG}_nearest.pt")
torch.save({k: v.cpu() for k, v in net.state_dict().items()}, f"{OUT}/{TAG}.pt")
grid = ((g50[:16].clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
Image.fromarray(np.concatenate([np.concatenate(list(grid[i:i + 8]), 1) for i in (0, 8)], 0)).save(f"{OUT}/{TAG}.png")
print(TAG, "done", round(time.time() - t0), "s", flush=True)
