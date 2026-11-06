#!/usr/bin/env python3
"""A point model's own x-prediction error per t (docs/gmflow_hn.md, H1): MSE_x(t) = E ||x - x_hat(x_t, t)||^2 / D and s_x(t) = sqrt(MSE_x(t)), on held-out images.

Data: the GPIC validation split (data/gpic/val; training reads only data/gpic/train), the first N images of gpic_val_00000.tar that have a caption, centre-cropped
(the val transform) to 256, with their captions. Model: the checkpoint's EMA weights in fp32 under bf16 autocast, as in sampling; captions through Qwen3TextEncoder
in the masked mode of lite_ladder_sample.yaml (each batch padded to its longest caption, a multiple of 8); no CFG. Noise: fresh per (image, t), from a generator
seeded by the image's index, so the result does not depend on the sharding. t: 64 points uniform in [0.01, 0.99].

Run: one process per GPU on disjoint images (image i goes to shard i % shards), each writes its sums; then --merge averages over all images:
  calibrate_spread.py run <ckpt> <out dir> --shard i --shards 8 [--images 4096]
  calibrate_spread.py merge <out dir> <table.json>
The table: {"t": [...], "s_x": [...], "mse_x": [...], "se_mse_x": [...], "n_images": N, "ckpt": path, "data": ..., "seed": ...}. GMODESampler(spread_table=...) reads it.
"""
import argparse
import glob
import io
import json
import math
import os
import tarfile

import numpy as np
import torch

WC = "/scratch/gpfs/GROUP/USER"
VAL_TAR = f"{WC}/data/gpic/val/gpic_val_00000.tar"
T_GRID = [0.01 + (0.99 - 0.01) * i / 63 for i in range(64)]


def load_val(n, res=256, tar_path=VAL_TAR):
    """The first n (key, image (3, res, res) in [-1, 1], caption) of the tar, in tar order, as the val transform makes them."""
    from PIL import Image
    from src.data.dataset.augmentation import center_crop_arr
    out, seen = [], {}
    with tarfile.open(tar_path) as tar:
        for m in tar:
            key, _, ext = m.name.rpartition(".")
            if ext not in ("jpg", "json"):
                continue
            seen.setdefault(key, {})[ext] = tar.extractfile(m).read()
            if len(seen[key]) < 2:
                continue
            parts = seen.pop(key)
            meta = json.loads(parts["json"])
            if "caption" not in meta:
                continue
            arr = np.asarray(center_crop_arr(Image.open(io.BytesIO(parts["jpg"])).convert("RGB"), res)).copy()
            out.append((key, torch.from_numpy(arr).permute(2, 0, 1).float().div(127.5).sub(1), meta["caption"]))
            if len(out) == n:
                break
    assert len(out) == n, f"{tar_path} has only {len(out)} captioned images"
    return out


def noise_for(index, shape, seed, device):
    """The noise of image `index` at every t: (len(T), *shape), from its own generator."""
    g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + index)
    return torch.randn((len(T_GRID), *shape), generator=g, device=device)


@torch.no_grad()
def error_sums(predict, images, indices, ts, seed, batch):
    """Per t: the sum over images of the per-image MSE_x and of its square. predict(x_t, t (B,), rows) -> x_hat (B, C, H, W); rows index into `images`.
    images (N, C, H, W) on the device; indices: the global image indices (they seed the noise). Returns (sum (T,), sum of squares (T,)) in fp64."""
    dev = images.device
    s1 = torch.zeros(len(ts), dtype=torch.float64)
    s2 = torch.zeros(len(ts), dtype=torch.float64)
    for b0 in range(0, len(images), batch):
        rows = list(range(b0, min(b0 + batch, len(images))))
        x = images[rows].float()
        eps = torch.stack([noise_for(indices[r], x.shape[1:], seed, dev) for r in rows], 1)   # (T, B, C, H, W)
        for ti, t in enumerate(ts):
            x_t = t * x + (1 - t) * eps[ti]
            hat = predict(x_t, torch.full((len(rows),), t, device=dev, dtype=torch.float32), rows).float()
            mse = (x - hat).square().flatten(1).mean(1).double().cpu()
            s1[ti] += mse.sum()
            s2[ti] += mse.square().sum()
    return s1, s2


def build_net(ckpt, d, nb, nt, device):
    from src.models.transformer.JiT_T2I import JiT_T2I
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    sd = {k[len("ema_denoiser."):]: v for k, v in sd.items() if k.startswith("ema_denoiser.")}
    assert sd["final_layer.linear.weight"].shape[0] == 768, "a point model (gm_k = 0) checkpoint is needed"
    net = JiT_T2I(input_size=256, in_channels=3, num_groups=d // 64, hidden_size=d, num_blocks=nb, num_text_blocks=nt, patch_size=16, txt_embed_dim=2048,
                  txt_max_length=320, bottleneck_dim=256)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected and all("pos_embed" in m for m in missing), (missing, unexpected)   # pos_embed is a fixed buffer, rebuilt at init
    return net.to(device).float().eval().requires_grad_(False)


def cmd_run(a):
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    data = load_val(a.images)
    mine = [i for i in range(a.images) if i % a.shards == a.shard]
    net = build_net(a.ckpt, a.d, a.nb, a.nt, dev)
    from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder
    te = Qwen3TextEncoder(weight_path="Qwen/Qwen3-1.7B", embed_dim=2048, max_length=320, text_mode="masked", pad_multiple=8, attn_implementation="sdpa").to(dev).eval()
    images = torch.stack([data[i][1] for i in mine]).to(dev)
    caps = [data[i][2] for i in mine]
    cache = {}

    def predict(x_t, t, rows):
        key = (rows[0], len(rows))
        if key not in cache:   # one caption batch per image batch, encoded once, reused at every t
            meta = {}
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):   # as under Lightning's bf16-mixed
                y, _ = te([caps[r] for r in rows], meta)
            cache.clear()
            cache[key] = (y, meta["y_mask"])
        y, mask = cache[key]
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            return net(x_t, t, y, y_mask=mask)

    s1, s2 = error_sums(predict, images, mine, T_GRID, a.seed, a.batch)
    json.dump(dict(t=T_GRID, sum_mse=s1.tolist(), sum_sq=s2.tolist(), n=len(mine), ckpt=os.path.abspath(a.ckpt), images=a.images, seed=a.seed, data=VAL_TAR,
                   shard=a.shard, shards=a.shards), open(os.path.join(a.out, f"shard{a.shard}.json"), "w"))
    print(f"shard {a.shard}/{a.shards}: {len(mine)} images; s_x at t = {T_GRID[0]:.2f}, 0.5, {T_GRID[-1]:.2f}: "
          f"{math.sqrt(s1[0] / len(mine)):.4f} {math.sqrt(s1[31] / len(mine)):.4f} {math.sqrt(s1[-1] / len(mine)):.4f}", flush=True)


def merge(shard_dicts):
    """Sum the shards' sums; s_x = sqrt(mean MSE_x); se_mse_x = the standard error of that mean over images."""
    d0 = shard_dicts[0]
    for d in shard_dicts:
        assert d["t"] == d0["t"] and d["ckpt"] == d0["ckpt"] and d["seed"] == d0["seed"] and d["shards"] == d0["shards"], "shards of different runs"
    assert sorted(d["shard"] for d in shard_dicts) == list(range(d0["shards"])), "a shard is missing or repeated"
    n = sum(d["n"] for d in shard_dicts)
    s1 = np.sum([d["sum_mse"] for d in shard_dicts], 0)
    s2 = np.sum([d["sum_sq"] for d in shard_dicts], 0)
    mse = s1 / n
    se = np.sqrt(np.maximum(s2 / n - mse ** 2, 0) / max(n - 1, 1))
    return dict(t=d0["t"], s_x=np.sqrt(mse).tolist(), mse_x=mse.tolist(), se_mse_x=se.tolist(), n_images=int(n), ckpt=d0["ckpt"], data=d0["data"], seed=d0["seed"])


def cmd_merge(a):
    files = sorted(glob.glob(os.path.join(a.out, "shard*.json")))
    table = merge([json.load(open(f)) for f in files])
    json.dump(table, open(a.table, "w"), indent=1)
    print(f"{a.table}: {table['n_images']} images; s_x(0.01) {table['s_x'][0]:.4f}, s_x(0.99) {table['s_x'][-1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("ckpt"); r.add_argument("out")
    r.add_argument("--shard", type=int, default=0); r.add_argument("--shards", type=int, default=1)
    r.add_argument("--images", type=int, default=4096); r.add_argument("--batch", type=int, default=32); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--d", type=int, default=640); r.add_argument("--nb", type=int, default=9); r.add_argument("--nt", type=int, default=2)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    m = sub.add_parser("merge")
    m.add_argument("out"); m.add_argument("table")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_merge(a)


if __name__ == "__main__":
    main()
