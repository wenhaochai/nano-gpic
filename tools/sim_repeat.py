"""Repetition arms (owner 2026-11-07): when does a training run read the same image again? A simulation of the data path of
jobs/train_repeat.sh on CPU, without images, against uniform random sampling of the subset.
  cd <worktree> && PYTHONPATH=baselines/PixelGen CUDA_VISIBLE_DEVICES= python tools/sim_repeat.py <shards> <passes> [--real]
Per rank and worker it takes the worker's entries from EpochShuffledShards (8 ranks x 32 workers, the run's seed), reads
L samples of each entry's shard from its start sample on (EpochShuffledShards.parts, L = the smallest shard), passes the stream through the real
webdataset shuffle(100) and GPICWebDataset._group (window 1024, chunks of 32, text lengths drawn from the GPIC caption
length range), and gives batch b of a rank to worker b mod 32 (the DataLoader's in-order round robin), so sample q of
worker w lands in global step 32 (q // 32) + w. --real takes each shard's sample count from its tar (tools/ in
gpic_dataset.tar_sample_offsets), otherwise 12500 each.
Prints the gap in steps between two reads of one image (passes > 1 includes the pass boundaries), the duplicate images in
one global step of 256 images, and the same for uniform sampling: every step draws 256 images, i.i.d. with replacement
(the gap is then geometric with p = 256 / M) or without replacement inside a step (no duplicate in a step).
"""
import os
import random
import sys
from itertools import islice

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "baselines", "PixelGen"))
from webdataset.filters import _shuffle  # noqa: E402
from src.data.dataset.gpic_dataset import EpochShuffledShards, GPICWebDataset, tar_sample_offsets  # noqa: E402

WORLD, NW, BATCH, SEED = 8, 32, 32, 20261106
TAR = "/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_{:05d}.tar"


class _WI:
    def __init__(self, i, n):
        self.id, self.num_workers = i, n


def as_rank(r, f):
    old = {k: os.environ.get(k) for k in ("RANK", "WORLD_SIZE")}
    os.environ.update(RANK=str(r), WORLD_SIZE=str(WORLD))
    try:
        return f()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def simulate(n_shards, passes, real):
    urls = [TAR.format(i) for i in range(n_shards)]
    sizes = [len(tar_sample_offsets(u)[0]) for u in urls] if real else [12500] * n_shards
    L = min(sizes)
    first = np.cumsum([0] + sizes)                      # global image id = first[shard] + sample
    tlen = np.random.default_rng(7).integers(4, 321, int(first[-1]))   # one caption length per image, as in the data
    sid = {u: i for i, u in enumerate(urls)}
    ds = GPICWebDataset(tar_dir="/nonexistent/", tar_pattern="x", group_text_len=True, group_window=1024, group_batch=BATCH)
    ids, steps, pss = [], [], []
    for r in range(WORLD):
        copies = [as_rank(r, lambda: EpochShuffledShards(urls, SEED, workers=NW)) for _ in range(NW)]
        for c in copies:   # the index the rank reads before the fork (load_index), here with the simulated sizes
            c.index = {u: (np.zeros(sizes[i]), 0) for i, u in enumerate(urls)}
            c.length = min(sizes)
        k = copies[0].k
        for w in range(NW):
            import torch.utils.data as tud
            old = tud.get_worker_info
            tud.get_worker_info = lambda: _WI(w, NW)
            try:
                def stream():
                    for _ in range(passes):
                        for d in as_rank(r, lambda: list(iter(copies[w]))):
                            i = sid[d["url"]]
                            n = sizes[i]
                            s0 = min(int(d["start"] * n), n - 1)
                            for q in range(L):   # EpochShuffledShards.parts: L samples from the start on
                                g = int(first[i] + (s0 + q) % n)
                                yield {"id": g, "text_len": int(tlen[g]), "pass": d["shard_pass"]}
                out = ds._group(_shuffle(stream(), bufsize=100, initial=100, rng=random.Random(r * 77 + w)))
                got = np.array([(s["id"], s["pass"]) for s in out], dtype=np.int64).reshape(-1, 2)
            finally:
                tud.get_worker_info = old
            q = np.arange(len(got))
            ids.append(got[:, 0])
            pss.append(got[:, 1])
            steps.append(BATCH * (q // BATCH) + w)
    ids, steps, pss = np.concatenate(ids), np.concatenate(steps), np.concatenate(pss)
    return ids, steps, pss, int(first[-1]), k


def report(name, ids, steps, M, pss):
    o = np.lexsort((steps, ids))
    i, s, ps = ids[o], steps[o], pss[o]
    same = i[1:] == i[:-1]
    gaps = (s[1:] - s[:-1])[same]
    cross = (ps[1:] != ps[:-1])[same]
    T = int(steps.max()) + 1
    per_step = np.bincount(steps, minlength=T)
    # duplicates in a step: reads of one image in one step beyond the first, summed over the step's images
    key = steps.astype(np.int64) * (M + 1) + ids
    u, c = np.unique(key, return_counts=True)
    dup = np.bincount((u // (M + 1)).astype(np.int64), weights=c - 1, minlength=T)
    full = per_step == WORLD * BATCH
    q = np.percentile(gaps, [0.1, 1, 5, 50, 95, 99])
    print(f"{name}: {len(ids)} reads of {len(np.unique(ids))} of {M} images over {T} steps ({full.mean() * 100:.1f}% of steps hold 256)")
    print(f"  gap (steps): mean {gaps.mean():.0f}, min {gaps.min()}, p0.1 {q[0]:.0f}, p1 {q[1]:.0f}, p5 {q[2]:.0f}, median {q[3]:.0f}, "
          f"p95 {q[4]:.0f}, p99 {q[5]:.0f}; under 10 steps {np.mean(gaps < 10) * 100:.3f}%, under 50 {np.mean(gaps < 50) * 100:.2f}%")
    for nm, m in (("inside a pass", ~cross), ("across a pass boundary", cross)):
        if m.any():
            print(f"  {nm}: {m.sum()} gaps, mean {gaps[m].mean():.0f}, under 10 steps {np.mean(gaps[m] < 10) * 100:.3f}%, "
                  f"gap 0 (one step) {np.sum(gaps[m] == 0)}, share of all gaps under 10: {np.sum(gaps[m] < 10) / max(1, np.sum(gaps < 10)) * 100:.1f}%")
    print(f"  duplicate images per full step: mean {dup[full].mean():.4f}, max {int(dup[full].max())}, steps with any {np.mean(dup[full] > 0) * 100:.2f}%")
    return gaps, dup[full]


def uniform(M, T, seed=0):
    """i.i.d. uniform: per step 256 draws with replacement. Gap: geometric, p = 256 / M; duplicates per step: simulated."""
    p = WORLD * BATCH / M
    g = np.random.default_rng(seed).geometric(p, size=2_000_000)
    dups = []
    rng = np.random.default_rng(seed + 1)
    for _ in range(min(T, 20000)):
        x = rng.integers(0, M, WORLD * BATCH)
        dups.append(WORLD * BATCH - len(np.unique(x)))
    dups = np.array(dups)
    q = np.percentile(g, [0.1, 1, 5, 50, 95, 99])
    print(f"uniform i.i.d. ({M} images, 256 per step): gap mean {g.mean():.0f} (= M / 256 = {M / 256:.0f}), min {g.min()}, p0.1 {q[0]:.0f}, "
          f"p1 {q[1]:.0f}, p5 {q[2]:.0f}, median {q[3]:.0f}, p95 {q[4]:.0f}, p99 {q[5]:.0f}; under 10 steps {np.mean(g < 10) * 100:.3f}%, "
          f"under 50 {np.mean(g < 50) * 100:.2f}%")
    print(f"  duplicate images per step: mean {dups.mean():.4f} (256 x 255 / 2M = {256 * 255 / 2 / M:.4f}), max {dups.max()}, "
          f"steps with any {np.mean(dups > 0) * 100:.2f}%; without replacement inside a step: 0")


def main():
    n, passes = int(sys.argv[1]), int(sys.argv[2])
    real = "--real" in sys.argv
    ids, steps, pss, M, k = simulate(n, passes, real)
    print(f"{n} shards, {k} copies of each per pass, {passes} passes, 8 ranks x 32 workers, seed {SEED}")
    report("this design", ids, steps, M, pss)
    uniform(M, int(steps.max()) + 1)


if __name__ == "__main__":
    main()
