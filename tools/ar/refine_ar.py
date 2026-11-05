#!/usr/bin/env python3
"""Can drawing some tokens again remove the stray patches of the MAR-order model? Saved weights of tools/ar/overfit_ar.py, no training, 64 steps. All arms start from the same samples, so the comparison is paired.
  base      the samples as drawn
  first16   the 16 tokens drawn first, drawn again given the other 240, in one step
  first64   the same with 64
  sweep8    8 steps over a random partition of all tokens into 8 groups of 32: each group is drawn again given the rest
Metrics of each arm: tools/ar/overfit_ar.py's (near, covered, distinct, nn_rms) and tools/ar/eval_saved_ar.py's robust ones, with each sample's values in <out>_per_sample.pt, and each arm's paired difference from base under "paired". Also, for the base samples, the share of stray patches by draw rank: a patch is stray if its rms to the image with the largest best_share is above 0.2.
usage: refine_ar.py <weights .pt> <N> <out .json> <sampling seed>    Run from baselines/PixelGen with PYTHONPATH=. and one visible GPU.
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


def paired(per_sample, n_images, resamples=1000):
    """Each arm minus base, on the same samples, with the standard error from resampling the samples: {arm: {metric: [difference, standard error]}}."""
    names = ("near", "covered", "near_robust", "covered_robust")

    def stats(p, i):
        near, good = p["nn"][i] < 0.1, p["best"][i] >= 0.9
        return torch.tensor([near.float().mean().item(), p["idx"][i][near].unique().numel() / n_images, good.float().mean().item(), p["who"][i][good].unique().numel() / n_images])

    count = len(per_sample["base"]["nn"])
    everyone = torch.arange(count)
    picks = torch.randint(0, count, (resamples, count), generator=torch.Generator().manual_seed(0))
    base_now = stats(per_sample["base"], everyone)
    base_boot = [stats(per_sample["base"], i) for i in picks]
    res = {}
    for arm, p in per_sample.items():
        if arm != "base":
            se = torch.stack([stats(p, i) - b for i, b in zip(picks, base_boot)]).std(0)
            now = stats(p, everyone) - base_now
            res[arm] = {m: [now[j].item(), se[j].item()] for j, m in enumerate(names)}
    return res


W, N, OUT, SEED = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
SAMPLES = 8 * N
WC = "/scratch/gpfs/GROUP/USER"
dev = "cuda"
torch.manual_seed(SEED)
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
assert any("mask_token" in k for k in sd), "this script is for the random order"
net = ARJiT(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256, head_width=1024, head_depth=3, head_steps=32, order="random", ar_steps=64)
net.load_state_dict(sd, strict=True)
net = net.to(dev).eval().requires_grad_(False)
net.compile()
unfold = lambda g: torch.nn.functional.unfold(g, kernel_size=16, stride=16).transpose(1, 2)


def score(tokens):
    g = net.unpatchify(tokens, 256, 256)
    d = torch.cat([torch.cdist(g[i:i + 64].flatten(1), imgs.flatten(1)) for i in range(0, len(g), 64)]) / (3 * 256 * 256) ** 0.5
    nn, idx = d.min(1)
    res = dict(nn_rms=nn.mean().item(), near=(nn < 0.1).float().mean().item(), covered=idx[nn < 0.1].unique().numel() / N, distinct=idx.unique().numel() / N)
    best, who, stray_flags = [], [], []
    for i in range(0, len(tokens), 32):
        per = ((tokens[i:i + 32, None] - true[None]) ** 2).mean(-1).sqrt()   # [s, N, 256]
        share, at = (per < 0.2).float().mean(-1).max(dim=1)
        best.append(share)
        who.append(at)
        stray_flags.append(per[torch.arange(len(at)), at] >= 0.2)
    best, who, flags = torch.cat(best), torch.cat(who), torch.cat(stray_flags)
    ok = best >= 0.9
    res.update(best_share=best.mean().item(), near_robust=ok.float().mean().item(), covered_robust=who[ok].unique().numel() / N, stray=(1 - best[ok]).mean().item() if ok.any() else None)
    return res, flags, ok, dict(nn=nn.cpu(), idx=idx.cpu(), best=best.cpu(), who=who.cpu())


t0 = time.time()
tokens, orders = [], []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(0, SAMPLES, B):
        g, order = net._sample_random(y, mask, return_order=True)
        tokens.append(unfold(g.float()))
        orders.append(order)
tokens, orders = torch.cat(tokens)[:SAMPLES], torch.cat(orders)[:SAMPLES]
out = dict(samples=len(tokens), seed=SEED, sample_s=time.time() - t0)
per_sample = {}
out["base"], flags, ok, per_sample["base"] = score(tokens)
rank = torch.zeros_like(orders)
rank.scatter_(1, orders, torch.arange(255, -1, -1, device=dev).repeat(len(orders), 1))   # 0 = drawn first
edges = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]
out["stray_by_rank"] = {f"{a}-{b - 1}": flags[(rank >= a) & (rank < b)].float().mean().item() for a, b in edges}
out["stray_by_rank_near_robust_only"] = {f"{a}-{b - 1}": flags[ok][((rank >= a) & (rank < b))[ok]].float().mean().item() for a, b in edges}
out["stray_share_rank_below"] = {str(k): ((flags[ok] & (rank[ok] < k)).sum() / flags[ok].sum().clamp_min(1)).item() for k in (16, 64)}   # over samples with best_share >= 0.9: the share of all stray patches with a draw rank below k


def redraw_all(tokens, picks):
    """picks: [S, 256] bool with the same count in each row. Runs in batches of B."""
    res = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(tokens), B):
            res.append(net.redraw(tokens[i:i + B], picks[i:i + B], y, mask))
    return torch.cat(res)


for k in (16, 64):
    t1 = time.time()
    picks = rank < k
    assert (picks.sum(1) == k).all()
    out[f"first{k}"], _, _, per_sample[f"first{k}"] = score(redraw_all(tokens, picks))
    out[f"first{k}"]["seconds"] = time.time() - t1
t1 = time.time()
current = tokens
perm = torch.rand(tokens.shape[0], 256, device=dev).argsort(dim=1)
for g in range(8):
    picks = torch.zeros(tokens.shape[0], 256, dtype=torch.bool, device=dev)
    picks.scatter_(1, perm[:, g * 32:(g + 1) * 32], True)
    current = redraw_all(current, picks)
out["sweep8"], _, _, per_sample["sweep8"] = score(current)
out["sweep8"]["seconds"] = time.time() - t1
json.dump(out, open(OUT, "w"), indent=1)
torch.save(per_sample, OUT.replace(".json", "_per_sample.pt"))
out["paired"] = paired(per_sample, N)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out)[:3000], flush=True)
