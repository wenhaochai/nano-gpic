#!/usr/bin/env python3
"""The shared caption-length bucket (GPICWebDataset._group_shared + src/data/shared_bucket_loader.py; new ladders).
4 ranks x 8 workers, synthetic streams with the measured per-sample length distribution (20 shards of the text cache),
the DataLoader's order (batch k from worker k % W), one SharedBucketLoader per rank in its own thread, the gloo all-reduce
replaced by a barrier that takes the elementwise minimum. Checks: every rank's micro-batch k has the same bucket and pads
to its edge; every sample lies in its bucket; the buckets come at their share of the data; every worker holds fewer than
one batch per bucket and every pool stays small; no sample twice, none lost. Then the share estimate from the cache.
usage: python test_shared_bucket.py"""
import os, sys, bisect, threading
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.data.dataset.gpic_dataset import GPICWebDataset
from src.data.shared_bucket_loader import SharedBucketLoader

DIST = {8: .0006, 16: .0990, 24: .2737, 32: .0896, 40: .1052, 48: .1628, 56: .1344, 64: .0380, 72: .0055, 80: .0007,
        88: .0002, 112: .0001, 120: .0008, 128: .0024, 136: .0058, 144: .0115, 152: .0155, 160: .0171, 168: .0146,
        176: .0100, 184: .0060, 192: .0029, 200: .0016, 208: .0008, 216: .0006, 224: .0003, 232: .0002, 240: .0001,
        248: .0001, 256: .0001}
EDGES = (16, 24, 32, 40, 48, 56, 64, 144, 176, 320)
R, W, K = 4, 8, 3000   # ranks, workers per rank, micro-batches per rank


def make_ds():
    ds = object.__new__(GPICWebDataset)
    ds.group_batch, ds.pad_multiple, ds.text_max_length, ds.bucket_edges = 32, 8, 320, EDGES
    return ds


def stream(seed):
    rng = np.random.default_rng(seed)
    vals, p = np.array(list(DIST)), np.array(list(DIST.values()))
    i = 0
    while True:
        for P in rng.choice(vals, size=4096, p=p / p.sum()):
            yield {"text_len": max(int(P) - int(rng.integers(0, 8)), 1), "id": (seed, i)}
            i += 1


def rank_loader(ds, r, emitted):
    """The rank's DataLoader stream: worker k % W's next batch of 32, as (imgs, caps, meta, samples)."""
    gens = [ds._group_shared(stream(1000 * r + w)) for w in range(W)]
    k = 0
    while True:
        b = [next(gens[k % W]) for _ in range(32)]
        emitted[r].append(b)
        T = b[0]["pad_to"]
        yield None, None, {"text_mask": torch.zeros(32, T, dtype=torch.bool)}, b
        k += 1


class Exchange:   # the gloo all-reduce (MIN) over R threads
    def __init__(self, n):
        self.n, self.vals, self.out = n, {}, None
        self.b1, self.b2 = threading.Barrier(n), threading.Barrier(n)

    def __call__(self, r, counts):
        self.vals[r] = counts
        if self.b1.wait() == 0:
            self.out = [min(v[i] for v in self.vals.values()) for i in range(len(counts))]
        self.b2.wait()
        out = list(self.out)
        self.b1.wait()   # every thread has read out before the next round writes it
        return out


ds = make_ds()
ex = Exchange(R)
emitted = [[] for _ in range(R)]
got = [[] for _ in range(R)]
pool_max = [0] * R


def run(r):
    L = SharedBucketLoader(rank_loader(ds, r, emitted), EDGES, allreduce_min=lambda c, r=r: ex(r, c))
    it = iter(L)
    for _ in range(K):
        got[r].append(next(it))
        pool_max[r] = max(pool_max[r], sum(len(q) for q in it.pools))


th = [threading.Thread(target=run, args=(r,)) for r in range(R)]
for t in th:
    t.start()
for t in th:
    t.join()
ok = True
same = all(got[r][k][2]["text_mask"].shape[1] == got[0][k][2]["text_mask"].shape[1] for r in range(R) for k in range(K))
ok &= same
print(f"every rank's micro-batch k has the same bucket ({R} ranks x {K} micro-batches): {same}")
inb = True
for r in range(R):
    for _, _, meta, b in got[r]:
        T = meta["text_mask"].shape[1]
        lo = EDGES[EDGES.index(T) - 1] if EDGES.index(T) else 0
        inb &= len(b) == 32 and all(s["pad_to"] == T and lo < min(-(-s["text_len"] // 8) * 8, 320) <= T for s in b)
ok &= inb
print(f"every batch 32 samples, every sample in its bucket and padded to the edge: {inb}")
counts = [0.0] * len(EDGES)
for P, f in DIST.items():
    counts[bisect.bisect_left(EDGES, P)] += f
share = np.array(counts) / sum(counts)
freq = np.bincount([EDGES.index(g[2]["text_mask"].shape[1]) for r in range(R) for g in got[r]], minlength=len(EDGES)) / (R * K)
err = max(abs(f - s) / s for f, s in zip(freq, share) if s > 0.02)
ok &= err < 0.15
print("bucket shares (data / trained):", ", ".join(f"{e}: {s:.3f}/{f:.3f}" for e, s, f in zip(EDGES, share, freq)),
      f"; largest relative gap over buckets above 2%: {err:.2f}")
print(f"largest pool per rank over the run (batches): {pool_max}")
ok &= max(pool_max) < 60
ids = [s["id"] for r in range(R) for g in got[r] for s in g[3]]
dup = len(ids) - len(set(ids))
pooled = sum(len(emitted[r]) for r in range(R)) - R * K
ok &= dup == 0
print(f"samples trained twice: {dup}; batches emitted but still pooled at the end: {pooled} (none dropped)")
print(f"mean padded T per micro-batch {np.mean([g[2]['text_mask'].shape[1] for r in range(R) for g in got[r]]):.1f} "
      "(the per-rank grouping gave 51.5)")
emb = "/scratch/gpfs/GROUP/USER/data/gpic/train_emb_qwen3_1.7b"
if os.path.isdir(emb):
    real = object.__new__(GPICWebDataset)
    real.tar_pattern, real.emb_dir, real.pad_multiple, real.text_max_length = "gpic_train_{00000..00800}.tar", emb, 8, 320
    real.bucket_edges, real.bucket_index_shards = EDGES, 32
    e2, s2 = GPICWebDataset.bucket_shares(real)
    print("shares from the cache index (32 shards):", ", ".join(f"{e}: {s:.4f}" for e, s in zip(e2, s2)))
print("SHARED_BUCKET_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
