#!/usr/bin/env python3
"""Overfit test of plan A on the real model (d640, 9 + 2 blocks), on one GPU, with no data workers and no Lightning. It uses the real model and the real trainer's loss.

Unconditional (the null caption), N real images. With N = 1 the posterior is the image, so nothing is left for z to carry: this checks the plumbing only (both arms must return the image).
With N = 32 or 256 the posterior has many modes, and the question is how many images the samples cover at each step. This test is one-sided. A code wins easily by remembering which image it
is, so a pass does not show that plan A helps on new images. A failure does show a fault in the design or the code.

Metrics, at 50 Euler calls with the official grid and t_eps, from the raw weights (the official EMA of 0.9999 hardly moves in so few steps):
  nn_rms    root mean square per pixel from each sample to its nearest training image, in [-1, 1] units; mean and 90th percentile
  near      share of samples with nn_rms below 0.1
  covered   share of training images that are the nearest of a sample with nn_rms below 0.1
  distinct  share of training images that are the nearest of any sample (no threshold)
The learning rate warms up over 500 steps. Production has 4000 warm-up steps (lite_ladder.yaml); this test has fewer steps in all.
usage: overfit_code.py <N> <R, 0 for the point model> <out dir> [steps]    Run from baselines/PixelGen with PYTHONPATH=. and one visible GPU.
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
from src.diffusion.flow_matching.training_code import CodeJiTTrainer
from src.diffusion.flow_matching.training_JiT import constant
from src.models.transformer.JiT_T2I import JiT_T2I

N, RATE, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 24000
EVALS = sorted({s for s in (1000, 2000, 4000, 8000, 16000) if s < STEPS} | {STEPS})
WC = "/scratch/gpfs/GROUP/USER"
TAG = f"n{N}_r{RATE:g}"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
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
kw = dict(code_dim=64, code_rate=RATE) if RATE else {}
net = JiT_T2I(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256, **kw).to(dev)
net.compile()
tr = CodeJiTTrainer(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05, null_condition_p=0.0).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, fused=True)


@torch.no_grad()
def sample(n, calls):
    res = []
    ts = torch.cat([torch.linspace(0, 1 - 1 / calls, calls), torch.ones(1)]).to(dev)
    for i in range(0, n, B):
        x = torch.randn(B, 3, 256, 256, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            extra = {"z": net.code_flow(torch.randn(B, net.code_dim, device=dev), y, mask)} if RATE else {}
            for j in range(calls):
                t = ts[j].repeat(B)
                pred = net(x, t, y, y_mask=mask, **extra).float()
                x = x + (ts[j + 1] - ts[j]) * (pred - x) / (1 - t.view(-1, 1, 1, 1)).clamp_min(0.05)
        res.append(x.float())
    return torch.cat(res)[:n]


def measure(g):
    d = torch.cat([torch.cdist(g[i:i + 64].flatten(1), imgs.flatten(1)) for i in range(0, len(g), 64)]) / (3 * 256 * 256) ** 0.5
    nn, idx = d.min(1)
    return dict(nn_rms=nn.mean().item(), nn_rms_p90=nn.quantile(0.9).item(), near=(nn < 0.1).float().mean().item(), covered=idx[nn < 0.1].unique().numel() / N, distinct=idx.unique().numel() / N, pixel_std=g.std(0).mean().item())


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
    if step in EVALS:
        row = dict(step=step, seconds=time.time() - t0, fm=ema_fm, **measure(sample(max(64, 8 * N), 50)))
        one = measure(sample(64, 1))
        row.update(one_call_nn_rms=one["nn_rms"], one_call_near=one["near"])
        if RATE:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                zs = net.code_encode(imgs[:B] if N >= B else imgs[torch.randint(N, (B,), device=dev)])
            row["z_var"] = zs.var(0).mean().item()
        rows.append(row)
        json.dump(rows, open(f"{OUT}/{TAG}.json", "w"))
        print(TAG, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
g = sample(16, 50)
grid = ((g.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
Image.fromarray(np.concatenate([np.concatenate(list(grid[i:i + 8]), 1) for i in (0, 8)], 0)).save(f"{OUT}/{TAG}.png")
print(TAG, "done", round(time.time() - t0), "s", flush=True)
