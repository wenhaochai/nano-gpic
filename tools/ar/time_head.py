#!/usr/bin/env python3
"""Where the time of autoregressive sampling goes, and whether one compiled graph per draw (HeadFlow) helps. Saved weights, the null caption, batch 32, 64 steps, one GPU, no training.
For the loop and for HeadFlow, in this order: compile time (first batch); then wall time per batch over 4 batches, with no extra syncs; then 2 batches with a sync around each backbone call and each draw, which gives the split. Then the same seed through both, to compare the images.
usage: time_head.py <weights .pt> <out .json> <seed>    Run from baselines/PixelGen with PYTHONPATH=. and one visible GPU.
"""
import json
import os
import sys
import time

import torch

from src.models.transformer.AR_JiT import ARJiT

W, OUT, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
WC = "/scratch/gpfs/GROUP/USER"
dev = "cuda"
null = torch.load(f"{WC}/data/gpic/train_emb_qwen3_1.7b/null.pt", map_location=dev, weights_only=True).float()
B, T = 32, 8
y = torch.zeros(B, T, 2048, device=dev, dtype=torch.bfloat16)
y[:, 0] = null[0, 0].bfloat16()
mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
mask[:, 0] = True
net = ARJiT(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256, head_width=1024, head_depth=3, head_steps=32, order="random", ar_steps=64)
net.load_state_dict(torch.load(W, map_location="cpu", weights_only=True), strict=True)
net = net.to(dev).eval().requires_grad_(False)
net.compile()
out = dict(seed=SEED)


def one(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return net._sample_random(y, mask).float()


images = {}
keep_draw = ARJiT.draw_patch
for name, flag in (("loop", "0"), ("flow", "1")):
    os.environ["GPIC_AR_HEAD_FLOW"] = flag
    torch.manual_seed(SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    one()
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    walls = []
    for _ in range(4):
        t0 = time.perf_counter()
        one()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
    # the split: a sync at each boundary, so the sum is an upper bound of the wall above
    spent = dict(draw=0.0, draws=0)
    def timed_draw(self, z):
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = keep_draw(self, z)
        torch.cuda.synchronize()
        spent["draw"] += time.perf_counter() - t
        spent["draws"] += 1
        return res
    ARJiT.draw_patch = timed_draw
    splits = []
    for _ in range(2):
        spent.update(draw=0.0, draws=0)
        t0 = time.perf_counter()
        one()
        torch.cuda.synchronize()
        splits.append(dict(wall=time.perf_counter() - t0, draw=spent["draw"], draws=spent["draws"]))
    ARJiT.draw_patch = keep_draw
    images[name] = one(seed=SEED + 1)
    torch.cuda.synchronize()
    out[name] = dict(first_batch_s=compile_s, wall_per_batch=walls, median_wall=sorted(walls)[len(walls) // 2], split=splits)
    print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
d = (images["loop"] - images["flow"]).abs()
out["same_seed"] = dict(max_abs=d.max().item(), mean_abs=d.mean().item(), rms_per_pixel_range=((d ** 2).mean().sqrt() / 2).item())
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out["same_seed"]), flush=True)
