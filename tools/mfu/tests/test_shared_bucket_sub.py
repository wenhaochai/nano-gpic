#!/usr/bin/env python3
"""GPIC_SHARED_BUCKET_SUB (gpic-28 2026-11-09): the start transient of the shared bucket, with and without sub-batches.
The setup of the ladders: R ranks x 32 workers, micro-batch 128, the measured per-sample length distribution (DIST of
test_shared_bucket.py), the DataLoader's order (item k from worker k % W), one SharedBucketLoader per rank in its own
thread, the gloo all-reduce replaced by a barrier with the elementwise minimum. For sub = 1 (the loader of before) and
sub = 8 (sub-batches of 16): the first micro-batch of each long bucket after the start, each bucket's share in the first
1696 micro-batches (the FP8 resume test's resumed cooldown) and after them, and the invariants: every rank's micro-batch k
has one bucket, 128 samples, its edge as padding; no sample twice.
usage: python test_shared_bucket_sub.py"""
import os, sys, threading
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.data.dataset.gpic_dataset import GPICWebDataset
import src.data.shared_bucket_loader as sbl
# DIST, EDGES and Exchange as in test_shared_bucket.py (a script: importing it would run it)
DIST = {8: .0006, 16: .0990, 24: .2737, 32: .0896, 40: .1052, 48: .1628, 56: .1344, 64: .0380, 72: .0055, 80: .0007,
        88: .0002, 112: .0001, 120: .0008, 128: .0024, 136: .0058, 144: .0115, 152: .0155, 160: .0171, 168: .0146,
        176: .0100, 184: .0060, 192: .0029, 200: .0016, 208: .0008, 216: .0006, 224: .0003, 232: .0002, 240: .0001,
        248: .0001, 256: .0001}
EDGES = (16, 24, 32, 40, 48, 56, 64, 144, 176, 320)

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



R, W, B, K = 4, 32, 128, 2600


def make_ds(g):
    ds = object.__new__(GPICWebDataset)
    ds.group_batch, ds.pad_multiple, ds.text_max_length, ds.bucket_edges = g, 8, 320, EDGES
    return ds


def stream(seed):
    rng = np.random.default_rng(seed)
    vals, p = np.array(list(DIST)), np.array(list(DIST.values()))
    i = 0
    while True:
        for P in rng.choice(vals, size=4096, p=p / p.sum()):
            yield {"text_len": max(int(P) - int(rng.integers(0, 8)), 1), "id": seed * 10**8 + i}
            i += 1


def rank_loader(r, g):
    gens = [make_ds(g)._group_shared(stream(1000 * r + w)) for w in range(W)]
    k = 0
    while True:
        b = [next(gens[k % W]) for _ in range(g)]
        T = b[0]["pad_to"]
        yield (torch.tensor([s["id"] for s in b]), [s["text_len"] for s in b], {"text_mask": torch.zeros(g, T, dtype=torch.bool)})
        k += 1


def trial(sub):
    os.environ["GPIC_SHARED_BUCKET_SUB"] = str(sub)
    g = B // sub
    ex = Exchange(R)
    got = [[] for _ in range(R)]

    def run(r):
        it = iter(sbl.SharedBucketLoader(rank_loader(r, g), EDGES, allreduce_min=lambda c, r=r: ex(r, c)))
        for _ in range(K):
            got[r].append(next(it))
    th = [threading.Thread(target=run, args=(r,)) for r in range(R)]
    [t.start() for t in th]; [t.join() for t in th]
    T = [[b[2]["text_mask"].shape[1] for b in got[r]] for r in range(R)]
    assert all(T[r] == T[0] for r in range(R)), "ranks disagree on a bucket"
    for r in range(R):
        for x, lens, m in got[r]:
            assert len(x) == B and len(lens) == B and m["text_mask"].shape[0] == B, "a micro-batch is not 128 samples"
            e = m["text_mask"].shape[1]; lo = EDGES[EDGES.index(e) - 1] if EDGES.index(e) else 0
            assert all(lo < -(-L // 8) * 8 <= e for L in lens), "a sample outside its bucket"
        ids = torch.cat([x for x, _, _ in got[r]])
        assert len(ids) == len(set(ids.tolist())), "a sample twice"
    t0 = np.array(T[0])
    first = {e: int(np.argmax(t0 == e)) if (t0 == e).any() else None for e in (64, 144, 176, 320)}
    sh = lambda a: [round(100 * float(np.mean(a == e)), 2) for e in EDGES]
    print(f"sub={sub} (sub-batch {g}): first micro-batch with bucket 64/144/176/320: {first}")
    print(f"  share %, micro-batches 0-1695   : {dict(zip(EDGES, sh(t0[:1696])))}")
    print(f"  share %, micro-batches 1696-{K - 1}: {dict(zip(EDGES, sh(t0[1696:])))}")
    return first


f1, f8 = trial(1), trial(8)
assert f8[320] is not None and (f1[320] is None or f8[320] < f1[320] / 4), (f1, f8)
print("ok: sub-batches bring the long buckets in", f1[320], "->", f8[320])
