#!/usr/bin/env python3
"""The shared bucket's pool cap (src/data/shared_bucket_loader.py, GPIC_SHARED_BUCKET_POOL_CAP; genmodel 2026-11-08: w1682's
host OOM, pinned pooled batches creeping over long segments). Eight simulated ranks (threads, the real loader, a barrier in
place of the gloo all-reduce), each with its own endless stream of batches whose caption buckets follow a skewed
distribution: with the default cap (40) the pools stay at the cap, without a cap (the old loader) they creep; every
micro-batch that is not a free step takes the same bucket on every rank, no batch is lost or taken twice, and until the
first free step both loaders take the very same batches. CPU. usage: python test_shared_bucket_cap.py"""
import os, random, sys, threading, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.data.shared_bucket_loader import SharedBucketLoader  # noqa: E402

EDGES = [16, 24, 32, 40, 48, 56, 64, 144, 176, 320]
P = [0.05, 0.12, 0.14, 0.13, 0.11, 0.09, 0.08, 0.17, 0.06, 0.05]
W, N = 8, 20000


class Stream:   # rank r's batches in a fixed order: (rank, serial, bucket) and a text mask of the bucket's length
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        rng, k = random.Random(1000 + self.r), 0
        while True:
            T = rng.choices(EDGES, P)[0]
            yield ((self.r, k, T), None, {"text_mask": types.SimpleNamespace(shape=(1, T))})
            k += 1


class AllReduceMin:
    def __init__(self):
        self.bar, self.slots = threading.Barrier(W), [None] * W

    def __call__(self, r, counts):
        self.slots[r] = counts
        self.bar.wait()
        out = [min(c[i] for c in self.slots) for i in range(len(counts))]
        self.bar.wait()
        return out


def run(cap):
    ar, taken, pools = AllReduceMin(), [[] for _ in range(W)], [[] for _ in range(W)]

    def rank(r):
        ld = SharedBucketLoader(Stream(r), EDGES, allreduce_min=lambda c, r=r: ar(r, c), pool_cap=cap)
        it = iter(ld)
        for _ in range(N):
            f0 = it.free_steps
            taken[r].append(next(it)[0] + (it.free_steps > f0,))
            pools[r].append(sum(len(q) for q in it.pools))
    th = [threading.Thread(target=rank, args=(r,)) for r in range(W)]
    [t.start() for t in th]
    [t.join() for t in th]
    return taken, pools


new_t, new_p = run(None)
old_t, old_p = run(10 ** 9)
cap = 40
free = [k for k in range(N) if new_t[0][k][3]]
same_bucket = all(len({new_t[r][k][2] for r in range(W)}) == 1 for k in range(N) if not new_t[0][k][3]) and \
    all(len({old_t[r][k][2] for r in range(W)}) == 1 for k in range(N)) and \
    all(len({new_t[r][k][3] for r in range(W)}) == 1 for k in range(N))
unique = all(len({x[:2] for x in t[r]}) == N for t in (new_t, old_t) for r in range(W))
first = free[0] if free else N
prefix_same = all([x[:3] for x in new_t[r][:first]] == [x[:3] for x in old_t[r][:first]] for r in range(W))
new_max, old_max = max(max(p) for p in new_p), max(max(p) for p in old_p)
old_end = max(p[-1] for p in old_p)
ok = same_bucket and unique and prefix_same and new_max <= cap + 8 and old_max > 2 * cap
print(f"{W} ranks x {N} micro-batches: largest pool with the cap ({cap}) {new_max}, without it {old_max} (at the end "
      f"{old_end}); free steps {len(free)} ({100 * len(free) / N:.1f}%); same bucket on every rank outside free steps "
      f"{same_bucket}; no batch lost or repeated {unique}; identical batches until the first free step (micro-batch "
      f"{first}) {prefix_same}")
print("SHARED_BUCKET_CAP_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
