#!/usr/bin/env python3
"""Does the model follow a few known tokens, and are its draws real patches? Saved weights of tools/ar/overfit_ar.py, no training.
For each k in 0, 1, 2, 4, 8, 16, 32, 64, 128: give the model k true tokens of a training image at random positions, draw all the others in one step, and compare them, position by position, with the training images' patches.
  follow    share of drawn tokens within rms 0.2 of the given image's patch at that position. k = 0 gives no image, so it is the chance level
  ideal     the same share if the draw came from a training image chosen uniformly among those whose patches lie within 0.2 of all the k known tokens
  manifold  share of drawn tokens within 0.2 of any training image's patch at that position
  own_rms, any_rms  the medians of the two distances
At k = 0 also which training image each draw lies nearest to:
  effective_per_position  exp(entropy) of the image counts at each position, mean over positions
  effective_all           the same over all positions together
  baseline_*              the same statistics for true patches of uniformly chosen images, plus noise of the same median rms as any_rms
  *_hit                   the same, only for draws within 0.2 of a training patch, against a baseline with as many draws. An error of another shape than the baseline's white noise can favour central patches, and these counts do not read that as a preference (review 2026-11-04)
<out>_counts0.pt holds the counts; <out>_samples.png holds 16 samples, each row as drawn followed by a row with the first 64 tokens drawn again.
usage: follow_ar.py <weights .pt> <N> <out .json> <sampling seed> [head depth]    Run from baselines/PixelGen with PYTHONPATH=. and one visible GPU.
"""
import io
import json
import sys
import tarfile
import time

import numpy as np
import torch


def close_matrix(true, limit):
    """[N, N, tokens] bool: patch j of images a and b lie within rms limit of each other."""
    return torch.stack([((true[a:a + 1] - true) ** 2).mean(-1).sqrt() < limit for a in range(len(true))])


def distances(drawn, true):
    """[B, tokens, P] against [N, tokens, P] gives [B, N, tokens]: the rms at each position."""
    return ((drawn[:, None] - true[None]) ** 2).mean(-1).sqrt()


def effective(counts):
    """exp(entropy) over the last dimension."""
    q = counts / counts.sum(-1, keepdim=True).clamp_min(1)
    return (-(q * q.clamp_min(1e-30).log()).sum(-1)).exp()


def add_counts(counts, who, keep=None):
    T = who.shape[1]
    pos, images = torch.arange(T, device=who.device).repeat(len(who)), who.flatten()
    if keep is not None:
        pos, images = pos[keep.flatten()], images[keep.flatten()]
    counts.index_put_((pos, images), torch.ones(images.numel(), device=who.device), accumulate=True)


def measure(draw, true, ks, repeats, batch, limit=0.2):
    """draw(tokens [B, T, P], pick [B, T] bool) returns the tokens with the picked ones drawn again. Each image is the given image repeats times at each k. Returns {k: stats} and two counts [T, N] at k = 0: all draws, and only the draws within limit of a training patch."""
    N, T = true.shape[:2]
    assert N % batch == 0
    dev = true.device
    close = close_matrix(true, limit)
    res, counts, hits = {}, torch.zeros(T, N, device=dev), torch.zeros(T, N, device=dev)
    for k in ks:
        hit_own = hit_any = total = 0
        ideal, own_vals, any_vals = [], [], []
        for _ in range(repeats):
            perm = torch.randperm(N, device=dev)
            for i in range(0, N, batch):
                idx = perm[i:i + batch]
                chosen = torch.rand(batch, T, device=dev).argsort(1)[:, :k]
                known = torch.zeros(batch, T, dtype=torch.bool, device=dev).scatter(1, chosen, True)
                pick = ~known
                out = draw(true[idx], pick)
                assert (out[known] == true[idx][known]).all(), "a known token changed"
                per = distances(out, true)
                own = per[torch.arange(batch, device=dev), idx]
                nearest, who = per.min(1)
                hit_own += (own[pick] < limit).sum().item()
                hit_any += (nearest[pick] < limit).sum().item()
                total += pick.sum().item()
                own_vals.append(own[pick].cpu())
                any_vals.append(nearest[pick].cpu())
                for a, c, p in zip(idx, chosen, pick):
                    agree = close[a][:, c].all(-1)   # the images that agree with all k known tokens; the given image is one of them
                    ideal.append(close[a][agree][:, p].float().mean().item())
                if k == 0:
                    add_counts(counts, who)
                    add_counts(hits, who, nearest < limit)
        res[k] = dict(follow=hit_own / total, ideal=float(np.mean(ideal)), manifold=hit_any / total, own_rms=torch.cat(own_vals).median().item(), any_rms=torch.cat(any_vals).median().item(), draws=total)
    return res, counts, hits


def baseline(true, sigma, draws_per_position, batch):
    """The k = 0 counts of a draw that picks a uniform image at each position and adds noise of rms sigma."""
    N, T, P = true.shape
    dev = true.device
    counts = torch.zeros(T, N, device=dev)
    for _ in range(0, draws_per_position, batch):
        images = torch.randint(0, N, (batch, T), device=dev)
        drawn = true[images, torch.arange(T, device=dev)] + sigma * torch.randn(batch, T, P, device=dev)
        add_counts(counts, distances(drawn, true).min(1)[1])
    return counts


def main():
    from PIL import Image

    from src.data.dataset.augmentation import center_crop_arr
    from src.models.transformer.AR_JiT import ARJiT

    W, N, OUT, SEED = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    HEAD_DEPTH = int(sys.argv[5]) if len(sys.argv) > 5 else 3
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
    unfold = lambda g: torch.nn.functional.unfold(g, kernel_size=16, stride=16).transpose(1, 2)
    true = unfold(imgs)   # [N, 256, 768]
    null = torch.load(f"{WC}/data/gpic/train_emb_qwen3_1.7b/null.pt", map_location=dev, weights_only=True).float()
    B, T = 32, 8
    y = torch.zeros(B, T, 2048, device=dev, dtype=torch.bfloat16)
    y[:, 0] = null[0, 0].bfloat16()
    mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
    mask[:, 0] = True
    sd = torch.load(W, map_location="cpu", weights_only=True)
    assert any("mask_token" in k for k in sd), "this script is for the random order"
    net = ARJiT(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256, head_width=1024, head_depth=HEAD_DEPTH, head_steps=32, order="random", ar_steps=64)
    net.load_state_dict(sd, strict=True)
    net = net.to(dev).eval().requires_grad_(False)
    net.compile()

    def draw(tokens, pick):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return net.redraw(tokens, pick, y, mask)

    t0 = time.time()
    res, counts, hits = measure(draw, true, (0, 1, 2, 4, 8, 16, 32, 64, 128), repeats=4, batch=B)
    out = dict(seed=SEED, seconds=time.time() - t0, k={str(k): v for k, v in res.items()})
    per_position = int(counts[0].sum().item())
    base = baseline(true, res[0]["any_rms"], per_position, B)
    out["draw0"] = dict(draws_per_position=per_position, effective_per_position=effective(counts).mean().item(), effective_all=effective(counts.sum(0)).item(), baseline_per_position=effective(base).mean().item(), baseline_all=effective(base.sum(0)).item())
    hits_per_position = int(hits.sum(1).float().mean().item())
    base_hit = baseline(true, res[0]["any_rms"], hits_per_position, B)
    out["draw0"].update(hits_per_position=hits_per_position, effective_per_position_hit=effective(hits).mean().item(), effective_all_hit=effective(hits.sum(0)).item(), baseline_per_position_hit=effective(base_hit).mean().item(), baseline_all_hit=effective(base_hit.sum(0)).item())
    json.dump(out, open(OUT, "w"), indent=1)
    torch.save(dict(per_position=counts.cpu(), baseline=base.cpu(), hits=hits.cpu(), baseline_hit=base_hit.cpu()), OUT.replace(".json", "_counts0.pt"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        g, order = net._sample_random(y, mask, return_order=True)
        tokens = unfold(g.float())
        rank = torch.zeros_like(order)
        rank.scatter_(1, order, torch.arange(255, -1, -1, device=dev).repeat(B, 1))
        after = net.unpatchify(net.redraw(tokens, rank < 64, y, mask), 256, 256)
    to_row = lambda x: np.concatenate(list(((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()), axis=1)
    Image.fromarray(np.concatenate([to_row(g[:8]), to_row(after[:8]), to_row(g[8:16]), to_row(after[8:16])], axis=0)).save(OUT.replace(".json", "_samples.png"))
    print(json.dumps(out)[:2500], flush=True)


if __name__ == "__main__":
    main()
