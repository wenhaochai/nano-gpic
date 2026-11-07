#!/usr/bin/env python3
"""MFU lever E2 (src/data/rebalance_loader.py), on CPU: 8 ranks (gloo), each with a fake loader of batches in the packed
format of levers C and E (pil_exact_gpu.pack with random crops and resize tables, packed caption rows, half the batches
long-captioned as the length grouping makes them). Checks, over 6 steps:
1. split_samples then join_samples gives back the batch exactly (every tensor and int of the job, rows, lengths, types);
2. over all ranks each sample of a step comes out once, at its own batch position (its t and noise);
3. each sample's bytes (source crop, both passes' tables, geometry, caption rows) equal those of its source batch;
4. the busiest rank's caption tokens over the mean fall (the deal does its job).
usage: python test_rebalance_loader.py
"""
import os, random, sys, tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))

W, B, D, STEPS = 8, 32, 16, 6


def fake_batch(rank, step):
    from src.data.dataset.pil_exact_gpu import pack, OUT
    rng = np.random.default_rng(1000 * rank + step)
    long = rng.random() < 0.5
    items, lens = [], []
    for i in range(B):
        H, Wd = int(rng.integers(256, 330)), int(rng.integers(256, 330))
        src = rng.integers(0, 256, size=(H, Wd, 3), dtype=np.uint8)
        passes = []
        for ax in (1, 0) if rng.random() < 0.5 else (0, 1):
            K = int(rng.integers(5, 12))
            passes.append((ax, rng.integers(0, 10, size=OUT).astype(np.int32), None,
                           rng.integers(-2 ** 20, 2 ** 22, size=(OUT, K)).astype(np.int32)))
        items.append((src, passes))
        lens.append(int(rng.integers(120, 321)) if long else int(rng.integers(8, 65)))
    job = pack(items)
    emb = torch.from_numpy(rng.standard_normal((sum(lens), D)).astype(np.float32)).to(torch.bfloat16)
    caps = [f"r{rank}s{step}i{i}" for i in range(B)]
    types = [["tag", "short", "medium", "long"][int(rng.integers(0, 4))] for _ in range(B)]
    return job, caps, {"caption_type": types, "text_emb": emb, "text_lens": lens}


class FakeLoader:
    def __init__(self, rank):
        self.rank = rank

    def __iter__(self):
        for s in range(10 ** 9):
            yield fake_batch(self.rank, s)


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def worker(rank, init):
    os.environ["MASTER_PORT"] = "0"
    os.environ["GPIC_MFU_REBALANCE_SLOT_MB"] = "64"
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=W)
    from src.data.rebalance_loader import RebalancingLoader, split_samples, join_samples
    ok = True
    # 1. round trip
    b0 = fake_batch(rank, 0)
    rt = join_samples(split_samples(b0), pin=False)
    ok &= same(rt[0], b0[0]) and same(rt[1], b0[1]) and same(rt[2], b0[2])
    # 2-4. rebalanced steps
    it = iter(RebalancingLoader(FakeLoader(rank)))
    worst_before, worst_after = [], []
    for s in range(STEPS):
        job, caps, meta = next(it)
        got = split_samples((job, caps, meta))
        mine = [(c, i) for i, c in enumerate(caps)]
        allc = [None] * W
        dist.all_gather_object(allc, mine)
        ids = [c for r in allc for c, _ in r]
        ok &= sorted(ids) == sorted(f"r{r}s{s}i{i}" for r in range(W) for i in range(B))
        ok &= all(c.endswith(f"i{i}") for r in allc for c, i in r)
        for smp, c in zip(got, caps):
            r, i = int(c[1:c.index("s")]), int(c[c.index("i") + 1:])
            src = split_samples(fake_batch(r, s))[i]
            K = src["kk"].shape[-1]
            ok &= same(smp["src"], src["src"]) and same(smp["xmin"], src["xmin"]) and same(smp["hw"], src["hw"])
            ok &= same(smp["axis"], src["axis"]) and same(smp["emb"], src["emb"]) and smp["type"] == src["type"]
            k_used = int((src["kk"] != 0).any(dim=(0, 1)).nonzero().max()) + 1 if src["kk"].any() else 0
            ok &= same(smp["kk"][..., :k_used], src["kk"][..., :k_used]) and not smp["kk"][..., k_used:].any()
        loads = [sum(fake_batch(r, s)[2]["text_lens"]) for r in range(W)]
        new = [None] * W
        dist.all_gather_object(new, sum(meta["text_lens"]))
        worst_before.append(max(loads) / np.mean(loads))
        worst_after.append(max(new) / np.mean(new))
    flags = [None] * W
    dist.all_gather_object(flags, bool(ok))
    if rank == 0:
        print(f"busiest rank's caption tokens over the mean, per step: before {[round(x, 3) for x in worst_before]}, "
              f"after {[round(x, 3) for x in worst_after]}", flush=True)
        print("REBALANCE_LOADER_TEST " + ("PASS" if all(flags) and max(worst_after) < 1.1 else f"FAIL {flags}"), flush=True)
    dist.barrier()
    it.slots.close()
    os._exit(0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    init = "file://" + tempfile.mktemp(prefix="rbl_")
    mp.spawn(worker, args=(init,), nprocs=W, join=True)
