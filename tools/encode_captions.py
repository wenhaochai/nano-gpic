"""Precompute the frozen text encoder's outputs for GPIC tar shards, so training reads them instead of running the
encoder inside the synchronous step (docs/text_cache.md).

python tools/encode_captions.py --shards <tar glob> --out <dir> [--rank R --world W] [--batch 32] [--limit N]
Each process takes every W-th shard. A shard's real-token outputs, in tar order, go to <out>/<shard>.emb as bf16 bytes;
<out>/<shard>.idx.npz holds keys, token offsets and lengths. Both appear by rename only when complete, so a rerun skips
finished shards. The encoder, tokenizer, truncation, padding and bf16 autocast are those of the training path.
"""
import argparse
import glob
import json
import os
import tarfile
import time

import numpy as np
import torch

D = 2048


def read_captions(path):
    from webdataset.tariterators import base_plus_ext
    keys, caps = [], []
    with tarfile.open(path) as t:
        for m in t:
            if m.isfile() and m.name.endswith(".json"):
                meta = json.loads(t.extractfile(m).read())
                if "caption" in meta and "caption_type" in meta:
                    keys.append(base_plus_ext(m.name)[0])
                    caps.append(meta["caption"])
    return keys, caps


def encode_shard(te, tok, keys, caps, batch, pad_multiple, max_length, device):
    """Sorted by length and padded per batch as in training; returns real tokens in the input's order, as CPU bf16."""
    ids = tok(caps, truncation=True, max_length=max_length)["input_ids"]
    ids = [i or [tok.eos_token_id] for i in ids]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = [None] * len(ids)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        T = max(len(ids[i]) for i in idx)
        T = min(-(-T // pad_multiple) * pad_multiple, max_length)
        input_ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), T), dtype=torch.bool)
        for r, i in enumerate(idx):
            input_ids[r, :len(ids[i])] = torch.tensor(ids[i], dtype=torch.long)
            mask[r, :len(ids[i])] = True
        meta = {"input_ids": input_ids.to(device), "text_mask": mask.to(device)}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = te._impl_condition(None, meta)
        assert y.dtype == torch.bfloat16 and y.shape[2] == D, (y.dtype, y.shape)
        y = y.cpu()
        for r, i in enumerate(idx):
            out[i] = y[r, :len(ids[i])].clone()
    return out, [len(i) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pad-multiple", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0, help="only the first N shards of this process")
    a = ap.parse_args()
    # small CPU copies only: OpenMP over every visible core spins for minutes in a 7-core cgroup (vis node, 2026-11-02)
    torch.set_num_threads(1)
    from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder
    os.makedirs(a.out, exist_ok=True)
    device = "cuda"
    te = Qwen3TextEncoder("Qwen/Qwen3-1.7B", embed_dim=D, max_length=a.max_length, text_mode="masked",
                          pad_multiple=a.pad_multiple, attn_implementation="sdpa").to(device).eval()
    tok = te.tokenizer
    if a.rank == 0:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            te._impl_uncondition(["x"], {})   # computes te.null_embedding: the null caption of CFG dropout, one eos token
        tmp = os.path.join(a.out, f".null.{os.getpid()}.pt")
        torch.save(te.null_embedding.cpu(), tmp)
        os.replace(tmp, os.path.join(a.out, "null.pt"))
    shards = sorted(glob.glob(a.shards))[a.rank::a.world]
    if a.limit:
        shards = shards[:a.limit]
    t0 = time.time()
    tokens = 0
    for n, path in enumerate(shards):
        base = os.path.basename(path)[:-len(".tar")]
        emb, idx = os.path.join(a.out, base + ".emb"), os.path.join(a.out, base + ".idx.npz")
        if os.path.exists(emb) and os.path.exists(idx):
            continue
        t1 = time.time()
        keys, caps = read_captions(path)
        t2 = time.time()
        out, lens = encode_shard(te, tok, keys, caps, a.batch, a.pad_multiple, a.max_length, device)
        t3 = time.time()
        offs = np.cumsum([0] + lens[:-1], dtype=np.int64)
        # NumPy has no bf16: reinterpret the bytes as int16 (little-endian on this hardware, the same bytes the reader expects)
        torch.cat(out).view(torch.int16).numpy().tofile(emb + f".tmp{os.getpid()}")
        np.savez(idx + f".tmp{os.getpid()}.npz", keys=np.array(keys), offsets=offs, lengths=np.array(lens, dtype=np.int32))
        os.replace(emb + f".tmp{os.getpid()}", emb)
        os.replace(idx + f".tmp{os.getpid()}.npz", idx)
        tokens += sum(lens)
        print(f"[encode] rank {a.rank} {n + 1}/{len(shards)} {base}: {len(keys)} captions, {sum(lens)} tokens, read {t2 - t1:.1f} s, "
              f"encode {t3 - t2:.1f} s, write {time.time() - t3:.1f} s; total {time.time() - t0:.0f} s, {tokens * D * 2 / 2**30:.1f} GiB", flush=True)
    print(f"[encode] rank {a.rank} done {len(shards)} shards in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
