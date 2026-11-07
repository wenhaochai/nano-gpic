#!/usr/bin/env python3
"""Offline held-out loss of the point-model ladders (owner 2026-11-06: 'add a held-out loss evaluation').

For one ms checkpoint (ms-step=NNNNNN.ckpt of a point-model run: JiT_T2I, K = 0, flow matching, x-prediction, loss in
v space) and for both its EMA weights (ema_denoiser.*) and its raw weights (denoiser.*), the training objective on fixed
data, with the definition of FixedValLoss (gpic-wt/muon/baselines/PixelGen/src/callbacks/val_loss.py):
- loss: JiTTrainer's (training_JiT.py): x_t = t x + (1 - t) e, v = (x - x_t) / max(1 - t, t_eps), the network's x
  prediction turned into v the same way, weight constant, squared error averaged over C, H, W per image; eager forward
  (net.forward, no torch.compile) under bf16 autocast, the loss in fp32, sums in fp64;
- t: NUM_T = 16 levels at the quantiles (k + 0.5) / 16 of the training t distribution: sigmoid(P_mean + P_std z_q),
  then time_shift_fn; the mean over levels estimates the expected training loss;
- noise: one fixed noise per (image, level): image j of a set draws randn((16, C, H, W)) from a CPU generator seeded
  SEED * 1_000_003 + j, independent of the batch, the GPU and the number of GPUs; SEED = 0 for both sets, so image j of
  the two sets shares its noise and levels (common random numbers for the train-val gap);
- data: (a) held-out: the first N_val samples of the GPIC validation shards gpic_val_*.tar (shard order, tar order)
  with their own captions; (b) train: the first N_train samples of gpic_train_00787.tar (TRAIN_SHARD below); images
  center-cropped to 256 (val_transform), no caption dropout;
- text: as in training (CachedTextConditioner): Qwen3-1.7B outputs of the real tokens, zeros on padding, a [B, T] mask,
  T = the batch's longest caption rounded up to 8, at most 320. Train captions take their outputs from the training
  cache itself (train_emb_qwen3_1.7b); held-out captions are encoded once with tools/encode_captions.encode_shard, the
  function that built that cache (same tokenizer, truncation, length-sorted batches of 32 padded to a multiple of 8,
  SDPA, bf16 autocast). `prep` also re-encodes the train captions and compares them with the cache.
Images are batched in order of caption length, so a batch pads little; the order changes only rounding.

Subcommands (run from anywhere; the repo's baselines/PixelGen and tools/ go on sys.path):
  prep  --n-val N --n-train N        reads, crops and encodes the data once -> OUT_DIR/data_v1_val<N>_train<N>_00787.pt
  list  --out results.jsonl [--ckpt ...]   prints the task list (JSON lines): every ms checkpoint of the ladders that
                                     is older than --min-age seconds and lacks a line for some weights in the output
  work  --out results.jsonl --data <prep file> (--tasks file --counter file | --ckpt ...) [--gpu i]
                                     computes the tasks; with --tasks, workers share the list through a lockf counter
                                     (each takes the next task); appends one JSON line per (checkpoint, weights)
  bench --d 1792 --nb 18 --nt 4      seconds per forward of a random model, and the implied minutes per checkpoint
Idempotent: a (checkpoint, weights, N_val, N_train, num_t, seed, data file) that has a line in the output is skipped.
"""
import argparse
import fcntl
import glob
import hashlib
import io
import json
import math
import os
import queue
import re
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(REPO, "baselines", "PixelGen"), os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

WC = "/scratch/gpfs/GROUP/USER"
RUNS = os.path.join(WC, "runs", "gpic-speed")
VAL_GLOB = os.path.join(WC, "data/gpic/val/gpic_val_*.tar")
# A shard that every run of both ladders has trained on before its first ms checkpoint: in GPIC Lite's pattern
# {00000..00800} and in GPIC Full's {00000..07999}; position 175 in Full's first permutation (seed 20261106), so in the
# first 256 shards that all 8 ranks x 32 workers start at step 0 (each full_ladder/exp_ladder_*/shards_started_rank*.txt
# logs it at the run's first second); in the Lite runs' first permutations at positions 402 (d640), 660 (d768), 25 (d896),
# 235 (d1024), 29 (d1280) of 801, and the first ms checkpoint of each Lite run comes after 3.39, 2.15, 1.46, 1.02 and
# 0.57 epochs (d1280: 454 shards started). Of the 4 Lite shards in Full's first 256 that d1280 starts in its first 256,
# 00787 comes first for d1280, the Lite run with the fewest epochs.
TRAIN_SHARD = os.path.join(WC, "data/gpic/train/gpic_train_00787.tar")
TRAIN_EMB = os.path.join(WC, "data/gpic/train_emb_qwen3_1.7b")
OUT_DIR = os.path.join(WC, "tmp", "heldout_loss")
NUM_T, SEED, BATCH = 16, 0, 128
SPLITS = ("val", "train", "train_reenc")
IMAGE, PAD_MULTIPLE, TEXT_MAX, EMB_DIM = 256, 8, 320, 2048
IMAGE_SUFFIXES = ("jpg", "jpeg", "png", "webp")
WEIGHTS = (("ema", "ema_denoiser."), ("raw", "denoiser."))
GLOBAL_BATCH = 256   # images per optimizer step in both ladders (8 GPUs x 32)
# training GFLOP per image (3 x forward) of sizes that docs/ladder_runs.json does not list: 3 x the forward GFLOP of
# tools/submit_full_ladder.sh (branch ladder-full), the same formula (tools/ladder_flops.py) at the same mean T
FULL_ONLY = {"d1536": (1536, 16, 4, 3 * 341.6), "d1792": (1792, 18, 4, 3 * 518.3)}
CKPT_RE = re.compile(r"ms-step=(\d+)\.ckpt")
RUN_RE = re.compile(r"exp_ladder_(d\d+)")


# ---------------------------------------------------------------------------------------------------------------- data
def _tar_groups(path):
    """(key, {suffix: bytes}) in tar order, grouped as webdataset groups a tar (base_plus_ext, lower-case suffixes)."""
    from webdataset.tariterators import base_plus_ext
    key, cur = None, {}
    with tarfile.open(path) as tar:
        for m in tar:
            if not m.isfile():
                continue
            k, ext = base_plus_ext(m.name)
            if k is None:
                continue
            if k != key or ext.lower() in cur:
                if key is not None:
                    yield key, cur
                key, cur = k, {}
            cur[ext.lower()] = tar.extractfile(m).read()
    if key is not None:
        yield key, cur


def read_samples(paths, n):
    """The first n samples with a caption, a caption type and an image (the training pipeline's filters), in shard and
    tar order; returns (samples, number of samples skipped before the n-th)."""
    out, skipped = [], 0
    for path in paths:
        for key, d in _tar_groups(path):
            img = next((e for e in IMAGE_SUFFIXES if e in d), None)
            meta = json.loads(d["json"]) if "json" in d else {}
            if img is None or "caption" not in meta or "caption_type" not in meta:
                skipped += 1
                continue
            out.append(dict(key=key, shard=os.path.basename(path), image=d[img], caption=meta["caption"]))
            if len(out) == n:
                return out, skipped
    raise RuntimeError(f"only {len(out)} samples in {len(paths)} shards, {n} asked")


def center_crop_u8(b):
    """val_transform's first step (center_crop_arr to 256) on encoded image bytes, as an HWC uint8 array."""
    from PIL import Image
    from src.data.dataset.augmentation import center_crop_arr
    return np.asarray(center_crop_arr(Image.open(io.BytesIO(b)).convert("RGB"), IMAGE), dtype=np.uint8)


def to_input(u8):
    """[B, H, W, 3] uint8 -> the denoiser's x: val_transform's ToTensor + Normalize(0.5, 0.5), the same fp32 ops (/255,
    -0.5, /0.5; tools/test_heldout_loss.py checks bit equality), then PixelAE(scale 1, shift 0), the identity."""
    return u8.permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.5)


def text_batch(split, idx, device):
    """The cached path's batch (GPICWebDataset._collate_cached): real-token outputs, zeros after, mask = lengths,
    T = the longest caption rounded up to pad_multiple, at most text_max_length."""
    lens = [int(split["lens"][i]) for i in idx]
    T = min(-(-max(lens) // split["pad_multiple"]) * split["pad_multiple"], split["text_max_length"])
    emb = split["emb"]
    y = torch.zeros((len(idx), T, emb.shape[1]), dtype=emb.dtype, device=device)
    mask = torch.zeros((len(idx), T), dtype=torch.bool, device=device)
    for r, (i, n) in enumerate(zip(idx, lens)):
        o = int(split["offs"][i])
        y[r, :n] = emb[o:o + n]
        mask[r, :n] = True
    return y, mask


def data_path(n_val, n_train):
    return os.path.join(OUT_DIR, f"data_v1_val{n_val}_train{n_train}_{os.path.basename(TRAIN_SHARD)[11:16]}.pt")


def pack_split(samples_or_none, img, embs, extra):
    lens = torch.tensor([len(e) for e in embs], dtype=torch.int64)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)[:-1]])
    d = dict(img=img, emb=torch.cat(embs).contiguous(), lens=lens, offs=offs, pad_multiple=PAD_MULTIPLE,
             text_max_length=TEXT_MAX)
    if samples_or_none is not None:
        d.update(keys=[s["key"] for s in samples_or_none], shards=[s["shard"] for s in samples_or_none],
                 captions=[s["caption"] for s in samples_or_none])
    d.update(extra)
    return d


def cmd_prep(a):
    path = a.data
    if os.path.exists(path):
        print(f"[prep] {path} exists", flush=True)
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    t0 = time.time()
    val, val_skip = read_samples(sorted(glob.glob(VAL_GLOB)), a.n_val)
    tr, tr_skip = read_samples([TRAIN_SHARD], a.n_train)
    print(f"[prep] read {len(val)} val samples ({val_skip} skipped) and {len(tr)} train samples ({tr_skip} skipped) "
          f"in {time.time() - t0:.0f} s", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.procs) as ex:   # before any CUDA call: the pool forks
        imgs = [torch.from_numpy(np.stack(list(ex.map(center_crop_u8, [s["image"] for s in ss], chunksize=16))))
                for ss in (val, tr)]
    print(f"[prep] cropped in {time.time() - t0:.0f} s", flush=True)
    # train captions: the training cache's own outputs (what training read for these samples)
    name = os.path.basename(TRAIN_SHARD)[:-len(".tar")]
    with np.load(os.path.join(TRAIN_EMB, name + ".idx.npz")) as z:
        index = dict(zip(z["keys"].tolist(), zip(z["offsets"].tolist(), z["lengths"].tolist())))
    cached = []
    with open(os.path.join(TRAIN_EMB, name + ".emb"), "rb") as f:
        for s in tr:
            off, n = index[s["key"]]
            raw = os.pread(f.fileno(), n * EMB_DIM * 2, off * EMB_DIM * 2)
            assert len(raw) == n * EMB_DIM * 2, (s["key"], len(raw), n)
            cached.append(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(n, EMB_DIM))
    # held-out captions: encoded as tools/encode_captions.py built the cache
    from encode_captions import encode_shard
    from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder
    torch.set_num_threads(1)
    te = Qwen3TextEncoder("Qwen/Qwen3-1.7B", embed_dim=EMB_DIM, max_length=TEXT_MAX, text_mode="masked",
                          pad_multiple=PAD_MULTIPLE, attn_implementation="sdpa").to("cuda").eval()
    enc = lambda ss: encode_shard(te, te.tokenizer, [s["key"] for s in ss], [s["caption"] for s in ss], 32,
                                  PAD_MULTIPLE, TEXT_MAX, "cuda")[0]
    val_emb = enc(val)
    # check of the text path: the same function on the train captions against the training cache
    tr_emb = enc(tr)
    assert [len(e) for e in tr_emb] == [len(e) for e in cached], "token lengths differ from the training cache"
    a32, b32 = torch.cat(tr_emb).float(), torch.cat(cached).float()
    diff = a32 - b32
    check = dict(tokens=int(a32.shape[0]), frac_equal=float((diff == 0).float().mean()),
                 rel_rms=float(diff.pow(2).mean().sqrt() / b32.pow(2).mean().sqrt()), max_abs=float(diff.abs().max()),
                 max_abs_cache=float(b32.abs().max()))
    print(f"[prep] train captions re-encoded vs the training cache: {check}", flush=True)
    # bf16 compute through 28 layers: on 512 of these captions (vis2 A40, 2026-11-07) the cache and a re-encode are each
    # 1.5% (RMS, relative) from an fp32 encoding of one caption at a time, and 1.1% from each other; bf16 rounding of the
    # outputs alone is 0.17%. A different tokenizer, truncation, padding or mask would give an error of order 1.
    assert check["rel_rms"] < 3e-2, check
    del te
    out = dict(version=1, n_val=a.n_val, n_train=a.n_train, val_glob=VAL_GLOB, train_shard=TRAIN_SHARD,
               train_emb=TRAIN_EMB, text_check=check, skipped=dict(val=val_skip, train=tr_skip),
               created=time.strftime("%Y-%m-%dT%H:%M:%S"), code=_git_head(),
               val=pack_split(val, imgs[0], val_emb, dict(text="Qwen3-1.7B, tools/encode_captions.encode_shard")),
               train=pack_split(tr, imgs[1], cached, dict(text=f"training cache {TRAIN_EMB}")),
               # the train images with the captions encoded on the held-out path: work --splits train_reenc measures
               # how much the encoding (cache vs this run's encode) moves the loss
               train_reenc=pack_split(tr, imgs[1], tr_emb, dict(text="Qwen3-1.7B, tools/encode_captions.encode_shard")))
    tmp = path + f".tmp{os.getpid()}"
    torch.save(out, tmp)
    os.replace(tmp, path)
    print(f"[prep] wrote {path} ({os.path.getsize(path) / 2**30:.2f} GiB) in {time.time() - t0:.0f} s; mean tokens "
          f"val {out['val']['lens'].float().mean():.1f} train {out['train']['lens'].float().mean():.1f}", flush=True)


def load_data(path, device):
    d = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    for name in SPLITS:
        if name not in d:
            continue
        s = d[name]
        s["img"], s["emb"] = s["img"].to(device), s["emb"].to(device)
        s["lens"], s["offs"] = s["lens"].tolist(), s["offs"].tolist()
        s["order"] = sorted(range(len(s["lens"])), key=lambda i: (s["lens"][i], i))   # by caption length
    d["tag"] = os.path.basename(path)[:-len(".pt")]
    return d


# --------------------------------------------------------------------------------------------------------------- loss
def levels(P_mean, P_std, timeshift, num_t=NUM_T):
    """FixedValLoss's levels: the quantiles (k + 0.5) / num_t of the training t (float64, then float32, then the shift)."""
    from src.diffusion.flow_matching.training_JiT import time_shift_fn
    q = (torch.arange(num_t, dtype=torch.float64) + 0.5) / num_t
    base_t = torch.sigmoid(P_mean + P_std * torch.special.ndtri(q)).float()
    return time_shift_fn(base_t, timeshift)


def image_noise(seed, j, num_t, shape, out=None):
    """Image j's noise at every level, FixedValLoss's draw: randn((num_t, *shape)) from a CPU generator seeded
    seed * 1_000_003 + j."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + j)
    return torch.randn((num_t, *shape), generator=g, out=out)


def level_loss(net, tr, x, y, y_mask, t, noise, autocast):
    """Per-image loss at one level t (0-d tensor): FixedValLoss._loss's body, which is JiTTrainer._impl_trainstep's
    with t and the noise given."""
    t = t.expand(x.shape[0])
    alpha, sigma = tr.scheduler.alpha(t), tr.scheduler.sigma(t)
    x_t = alpha * x + noise * sigma
    v_t = (x - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(tr.t_eps)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        pred = net.forward(x_t, t, y, y_mask=y_mask)
    out = (pred.float() - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(tr.t_eps)
    loss = tr.loss_weight_fn(alpha, sigma) * (out - v_t) ** 2
    return loss.mean(dim=(1, 2, 3))


def _noise_feed(batches, seed, num_t, shape, threads, pin):
    """Yields each batch's noise [B, num_t, *shape] in order, generated one batch ahead in a background thread, the
    images of a batch on `threads` threads (each image has its own generator, so the values do not depend on them).
    Three buffers, allocated once (1.6 GB each at 128 images): one in the queue, one being filled, one the consumer
    holds; the consumer must be done with a buffer (a synchronous copy) before it asks for the next one."""
    from concurrent.futures import ThreadPoolExecutor
    q = queue.Queue(maxsize=1)
    stop = threading.Event()
    B = max(len(i) for i in batches)
    bufs = [torch.empty((B, num_t, *shape), pin_memory=pin) for _ in range(min(3, len(batches)))]

    def produce():
        try:
            with ThreadPoolExecutor(threads) as pool:
                for b, idx in enumerate(batches):
                    buf = bufs[b % len(bufs)][:len(idx)]
                    list(pool.map(lambda r: image_noise(seed, idx[r], num_t, shape, out=buf[r]), range(len(idx))))
                    while not stop.is_set():
                        try:
                            q.put(buf, timeout=1)
                            break
                        except queue.Full:
                            pass
                    if stop.is_set():
                        return
            q.put(None)
        except BaseException as e:   # surfaces in the consumer
            q.put(e)

    th = threading.Thread(target=produce, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


@torch.no_grad()
def evaluate(nets, tr, split, ts, seed=SEED, batch=BATCH, device="cpu", autocast=False, threads=4, log=None):
    """Sums over images of the per-image loss at every level, for every net, on one data split; all nets see the same
    x_t. Returns {name: dict(level_mean [num_t], image_mean [N] (the mean over levels per image))} in float64."""
    N, num_t = len(split["lens"]), len(ts)
    shape = tuple(split["img"].shape[i] for i in (3, 1, 2))
    order = split["order"]
    batches = [order[i:i + batch] for i in range(0, N, batch)]
    sums = {k: torch.zeros(num_t, dtype=torch.float64, device=device) for k in nets}
    per = {k: torch.zeros(N, dtype=torch.float64, device=device) for k in nets}
    ts = ts.to(device)
    pin = torch.device(device).type == "cuda"
    t0 = time.time()
    for b, (idx, noise) in enumerate(zip(batches, _noise_feed(batches, seed, num_t, shape, threads, pin))):
        noise = noise.to(device)   # synchronous: the buffer is free for the producer when this returns
        it = torch.tensor(idx, device=device)
        x = to_input(split["img"][it])
        y, mask = text_batch(split, idx, device)
        for k in range(num_t):
            for name, net in nets.items():
                l = level_loss(net, tr, x, y, mask, ts[k], noise[:, k], autocast).double()
                sums[name][k] += l.sum()
                per[name].index_add_(0, it, l)
        if log and (b + 1) % max(1, len(batches) // 4) == 0:
            log(f"batch {b + 1}/{len(batches)} {time.time() - t0:.0f} s")
    return {k: dict(level_mean=(sums[k] / N).cpu(), image_mean=(per[k] / num_t).cpu()) for k in nets}


# -------------------------------------------------------------------------------------------------------------- model
def run_config(run_dir):
    """Denoiser kwargs and the loss settings from the run's saved configs (all launches must agree)."""
    import yaml
    cfgs = sorted(glob.glob(os.path.join(run_dir, "config-TrainerFn.FITTING-*.yaml")))
    assert cfgs, f"no config-TrainerFn.FITTING-*.yaml in {run_dir}"
    dens, losses = set(), set()
    for c in cfgs:
        with open(c) as f:
            m = yaml.safe_load(f)["model"]
        den, trn = m["denoiser"], m["diffusion_trainer"]
        assert den["class_path"] == "src.models.transformer.JiT_T2I.JiT_T2I", (c, den["class_path"])
        assert trn["class_path"] == "src.diffusion.flow_matching.training_JiT.JiTTrainer", (c, trn["class_path"])
        kw = dict(den["init_args"])
        for k in ("gm_k", "xi_dim", "code_dim", "head_width"):
            assert not kw.get(k), f"{c}: {k} = {kw.get(k)}: not the point model"
        ta = trn["init_args"]
        loss = dict(scheduler=ta["scheduler"]["class_path"], loss_weight_fn=ta["loss_weight_fn"],
                    lognorm_t=ta["lognorm_t"], timeshift=ta["timeshift"], P_mean=ta["P_mean"], P_std=ta["P_std"],
                    t_eps=ta["t_eps"])
        dens.add(json.dumps(kw, sort_keys=True))
        losses.add(json.dumps(loss, sort_keys=True))
    assert len(dens) == 1 and len(losses) == 1, f"{run_dir}: launches differ: {dens} {losses}"
    loss = json.loads(losses.pop())
    assert loss["scheduler"] == "src.diffusion.flow_matching.scheduling.LinearScheduler", loss
    assert loss["loss_weight_fn"] == "src.diffusion.flow_matching.training_JiT.constant", loss
    assert loss["lognorm_t"] is True, loss
    return json.loads(dens.pop()), loss


def make_trainer(loss):
    from src.diffusion.flow_matching.scheduling import LinearScheduler
    from src.diffusion.flow_matching.training_JiT import JiTTrainer, constant
    return JiTTrainer(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=loss["timeshift"],
                      P_mean=loss["P_mean"], P_std=loss["P_std"], t_eps=loss["t_eps"], null_condition_p=0.0)


def build_denoiser(kw, state_dict, prefix, device):
    """JiT_T2I(**kw) with the weights under prefix, loaded strictly (built on the meta device, so no random init)."""
    from src.models.transformer.JiT_T2I import JiT_T2I
    sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    with torch.device("meta"):
        net = JiT_T2I(**kw)
    net.load_state_dict(sub, strict=True, assign=True)
    left = [n for n, t in list(net.named_parameters()) + list(net.named_buffers()) if t.is_meta]
    assert not left, f"tensors without weights: {left[:5]}"
    return net.to(device).float().eval()


# -------------------------------------------------------------------------------------------------------------- tasks
def size_table():
    """size -> (d, nb, nt, training GFLOP per image, source)."""
    with open(os.path.join(REPO, "docs", "ladder_runs.json")) as f:
        t = {r["size"]: (r["d"], r["nb"], r["nt"], r["gflop_per_img"], "docs/ladder_runs.json")
             for r in json.load(f)["runs"]}
    for s, v in FULL_ONLY.items():
        t.setdefault(s, v + ("3 x forward GFLOP of tools/submit_full_ladder.sh (ladder-full)",))
    return t


def spec_of(ckpt):
    ckpt = os.path.abspath(ckpt)
    run_dir = os.path.dirname(ckpt)
    m, r = CKPT_RE.fullmatch(os.path.basename(ckpt)), RUN_RE.fullmatch(os.path.basename(run_dir))
    assert m and r, f"not <...>/exp_ladder_<size>/ms-step=<step>.ckpt: {ckpt}"
    root = os.path.dirname(run_dir)
    size, step = r.group(1), int(m.group(1))
    spec = dict(ckpt=ckpt, run_root=root, ladder="full" if os.path.basename(root) == "full_ladder" else "lite",
                size=size, step=step)
    t = size_table().get(size)
    if t:
        spec.update(d=t[0], nb=t[1], nt=t[2], train_flops=step * GLOBAL_BATCH * t[3] * 1e9, flops_source=t[4])
    return spec


def default_ckpts():
    dirs = (glob.glob(os.path.join(RUNS, "exp_ladder_d*")) + glob.glob(os.path.join(RUNS, "ladder_d*_*c0", "exp_ladder_d*"))
            + glob.glob(os.path.join(RUNS, "full_ladder", "exp_ladder_d*")))
    out = []
    for d in sorted(dirs):
        if RUN_RE.fullmatch(os.path.basename(d)):
            out += [p for p in sorted(glob.glob(os.path.join(d, "ms-step=*.ckpt"))) if CKPT_RE.fullmatch(os.path.basename(p))]
    return out


def done_keys(out):
    keys = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:   # a writer killed inside its write; that entry is redone
                        print(f"[heldout] {out}: skips an unreadable line: {line[:80]!r}", file=sys.stderr, flush=True)
                        continue
                    keys.add((r["ckpt"], r["weights"], r["n_val"], r["n_train"], r["num_t"], r["seed"], r["data"]))
    return keys


def todo_weights(spec, done, a, tag):
    return [w for w, _ in WEIGHTS if (spec["ckpt"], w, a.n_val, a.n_train, NUM_T, SEED, tag) not in done]


# Seconds per eager forward of 128 images at T = 56 (bf16 autocast, both nets alike), measured on a vis2 A40 with `bench`
# (2026-11-06), and the factor assumed for one H100 SXM: 4.8, the ratio of memory bandwidth (3.35 / 0.696 TB/s), below
# the ratio of dense bf16 peaks (6.6). The A40 ran at 22 (d640) to 48 (d1792) TFLOP/s. The smoke job measures the H100.
A40_S_PER_FWD = {640: 0.2071, 768: 0.2743, 896: 0.3564, 1024: 0.4587, 1280: 0.6976, 1536: 1.0084, 1792: 1.3792}
H100_OVER_A40 = 4.8


def est_minutes(d, n_images, batch=BATCH, load_min=0.5):
    """Minutes for one checkpoint on one H100, both weights: 2 nets x NUM_T levels x n_images / batch forwards."""
    s = A40_S_PER_FWD.get(d)
    if s is None:   # by d^2 from the nearest measured width
        near = min(A40_S_PER_FWD, key=lambda w: abs(w - d))
        s = A40_S_PER_FWD[near] * d * d / (near * near)
    return 2 * NUM_T * n_images / batch * s / H100_OVER_A40 / 60 + load_min


def cmd_list(a):
    tag = os.path.basename(a.data)[:-len(".pt")]   # the data file's name keys the done entries, as in work
    done = done_keys(a.out)
    now = time.time()
    tasks, young = [], 0
    for c in (a.ckpt if a.ckpt is not None else default_ckpts()):
        if now - os.path.getmtime(c) < a.min_age:   # may still be written (Lightning writes in place)
            young += 1
            continue
        spec = spec_of(c)
        if todo_weights(spec, done, a, tag):
            spec["est_min"] = round(est_minutes(spec.get("d", 1024), a.n_val + a.n_train), 2)
            tasks.append(spec)
    tasks.sort(key=lambda s: -s["est_min"])   # longest first, so the last tasks are short
    for s in tasks:
        print(json.dumps(s))
    tot = sum(s["est_min"] for s in tasks)
    print(f"[list] {len(tasks)} tasks, {young} checkpoints younger than {a.min_age} s skipped; estimate {tot:.0f} GPU-min "
          f"on H100 ({tot / 8:.0f} min on 8 GPUs)", file=sys.stderr, flush=True)


def claim(counter):
    """The next task index from a shared counter file, under lockf (one worker at a time)."""
    with open(counter, "a+") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        f.seek(0)
        s = f.read().strip()
        i = int(s) if s else 0
        f.seek(0)
        f.truncate()
        f.write(str(i + 1))
        f.flush()
        os.fsync(f.fileno())
        return i


def append_lines(path, recs):
    with open(path, "a") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        f.write("".join(json.dumps(r) + "\n" for r in recs))
        f.flush()
        os.fsync(f.fileno())


def _git_head():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def run_task(spec, weights, data, a, device):
    t0 = time.time()
    log = lambda msg: print(f"[heldout] {spec['size']} step {spec['step']} {spec['ladder']}: {msg}", flush=True)
    kw, loss = run_config(os.path.dirname(spec["ckpt"]))
    assert (kw["hidden_size"], kw["num_blocks"], kw["num_text_blocks"]) == (spec.get("d", kw["hidden_size"]),
            spec.get("nb", kw["num_blocks"]), spec.get("nt", kw["num_text_blocks"])), (kw, spec)
    ck = torch.load(spec["ckpt"], map_location="cpu", mmap=True, weights_only=False)
    assert ck["global_step"] == spec["step"], (ck["global_step"], spec["step"])
    prefixes = dict(WEIGHTS)
    nets = {w: build_denoiser(kw, ck["state_dict"], prefixes[w], device) for w in weights}
    del ck
    tr = make_trainer(loss)
    ts = levels(loss["P_mean"], loss["P_std"], loss["timeshift"])
    load_s = time.time() - t0
    log(f"loaded {','.join(weights)} in {load_s:.0f} s")
    autocast = torch.device(device).type == "cuda"
    res = {s: evaluate(nets, tr, data[s], ts, SEED, a.batch, device, autocast, a.threads, log=lambda m, s=s: log(f"{s} {m}"))
           for s in a.splits}
    elapsed = time.time() - t0
    gpu = torch.cuda.get_device_name(device) if autocast else "cpu"
    recs = []
    for w in weights:
        r = dict(spec)
        r.pop("est_min", None)
        r.update(weights=w, n_val=a.n_val, n_train=a.n_train, num_t=NUM_T, seed=SEED, data=data["tag"],
                 train_shard=os.path.basename(data["train_shard"]), t=ts.tolist(),
                 d=kw["hidden_size"], nb=kw["num_blocks"], nt=kw["num_text_blocks"])
        for s in a.splits:
            lm, im = res[s][w]["level_mean"], res[s][w]["image_mean"]
            r[f"loss_{s}"] = float(lm.mean())
            r[f"loss_{s}_t"] = lm.tolist()
            r[f"loss_{s}_se"] = float(im.std() / math.sqrt(len(im)))   # over images, of the mean over levels
        r.update(elapsed_s=round(elapsed, 1), load_s=round(load_s, 1), weights_in_task=list(weights), gpu=gpu,
                 host=socket.gethostname(), job=os.environ.get("FETCHED_JOB_ID") or os.environ.get("SLURM_JOB_ID", ""),
                 code=_git_head(), tool_sha1=_sha1(os.path.abspath(__file__)),
                 time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        recs.append(r)
    log(" ".join(f"{w}: val {r['loss_val']:.5f} train {r['loss_train']:.5f}" for w, r in zip(weights, recs))
        + f"; {elapsed:.0f} s")
    return recs


def cmd_work(a):
    if a.device == "cuda":
        torch.cuda.set_device(a.gpu)
        device = f"cuda:{a.gpu}"
    else:
        device = "cpu"
    torch.set_num_threads(2)
    data = load_data(a.data, device)
    assert (data["n_val"], data["n_train"]) == (a.n_val, a.n_train), (a.data, a.n_val, a.n_train)
    if a.tasks:
        with open(a.tasks) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        nxt = lambda: claim(a.counter)
    else:
        tasks = [spec_of(c) for c in a.ckpt]
        it = iter(range(len(tasks)))
        nxt = lambda: next(it, len(tasks))
    fails = n = 0
    while True:
        i = nxt()
        if i >= len(tasks):
            break
        spec = tasks[i]
        weights = todo_weights(spec, done_keys(a.out), a, data["tag"])
        if not weights:
            continue
        try:
            append_lines(a.out, run_task(spec, weights, data, a, device))
            n += 1
        except Exception:
            fails += 1
            print(f"[heldout] FAILED {spec['ckpt']}:\n{traceback.format_exc()}", flush=True)
        if device != "cpu":
            torch.cuda.empty_cache()
    print(f"[heldout] worker {a.gpu}: {n} checkpoints done, {fails} failed", flush=True)
    return 1 if fails else 0


def cmd_bench(a):
    """Seconds per forward of a randomly initialized JiT_T2I at batch a.batch and caption length a.T, on one GPU."""
    from src.models.transformer.JiT_T2I import JiT_T2I
    torch.cuda.set_device(a.gpu)
    kw = dict(input_size=IMAGE, in_channels=3, num_groups=a.d // 64, hidden_size=a.d, num_blocks=a.nb,
              num_text_blocks=a.nt, patch_size=16, txt_embed_dim=EMB_DIM, txt_max_length=TEXT_MAX, bottleneck_dim=256)
    net = JiT_T2I(**kw).cuda().eval()
    tr = make_trainer(dict(timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05))
    B = a.batch
    x = torch.randn(B, 3, IMAGE, IMAGE, device="cuda").clamp(-1, 1)
    y = torch.randn(B, a.T, EMB_DIM, device="cuda").bfloat16()
    mask = torch.ones(B, a.T, dtype=torch.bool, device="cuda")
    noise = torch.randn_like(x)
    t = torch.tensor(0.5, device="cuda")
    with torch.no_grad():
        for _ in range(3):
            level_loss(net, tr, x, y, mask, t, noise, True)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.iters):
            level_loss(net, tr, x, y, mask, t, noise, True)
        torch.cuda.synchronize()
    s = (time.time() - t0) / a.iters
    from ladder_flops import fwd
    tf = B * fwd(a.d, a.nb, a.nt, a.T, a.T * a.T) / s / 1e12
    n = a.n_val + a.n_train
    print(json.dumps(dict(d=a.d, nb=a.nb, nt=a.nt, batch=B, T=a.T, s_per_forward=round(s, 4), tflops=round(tf, 1),
                          gpu=torch.cuda.get_device_name(), images=n,
                          compute_min_per_ckpt=round(2 * n * NUM_T / B * s / 60, 2))), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "list", "work", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--n-val", type=int, default=3072)
        p.add_argument("--n-train", type=int, default=3072)
        if name == "prep":
            p.add_argument("--procs", type=int, default=32)
        if name in ("prep", "list", "work"):
            p.add_argument("--data", default=None, help="the prep file (default: OUT_DIR/data_v1_val<N>_train<N>_00787.pt)")
        if name in ("list", "work"):
            p.add_argument("--out", required=True)
            p.add_argument("--ckpt", nargs="*", default=None)
        if name == "list":
            p.add_argument("--min-age", type=float, default=600)
        if name == "work":
            p.add_argument("--tasks", default=None)
            p.add_argument("--counter", default=None)
            p.add_argument("--gpu", type=int, default=0)
            p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
            p.add_argument("--batch", type=int, default=BATCH)
            p.add_argument("--threads", type=int, default=6, help="CPU threads that draw the noise")
            p.add_argument("--splits", default="val,train", help="comma-separated, of " + ",".join(SPLITS))
        if name == "bench":
            for k, v in (("d", 1792), ("nb", 18), ("nt", 4), ("T", 56), ("iters", 10), ("gpu", 0), ("batch", BATCH)):
                p.add_argument(f"--{k}", type=int, default=v)
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        return cmd_prep(a)
    if a.cmd in ("prep", "list", "work"):
        a.data = a.data or data_path(a.n_val, a.n_train)
    if a.cmd == "list":
        return cmd_list(a)
    if a.cmd == "work":
        assert bool(a.tasks) == bool(a.counter) and bool(a.tasks) != bool(a.ckpt), "--tasks with --counter, or --ckpt"
        a.splits = a.splits.split(",")
        assert {"val", "train"} <= set(a.splits) <= set(SPLITS), a.splits
        return cmd_work(a)
    return cmd_bench(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
