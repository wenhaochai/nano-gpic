"""The shared caption-length bucket of the new ladders (owner's popup 2026-11-07 01:3x; GPICWebDataset shared_bucket).

Why: every rank pads its micro-batch to its own longest caption and the step waits for the slowest rank; with 8 ranks
about 57% of the steps hold one rank with T >= 72 (w1482: own lengths cost d640 +28%, d1280 +15%, d1792 +7% against one
length for all ranks; more with more GPUs).
How: the data workers (GPICWebDataset._group_shared) emit full batches of one bucket each (a bucket is a range of
padded caption lengths; its batch pads to the bucket's edge), greedily, so a worker never holds more than one batch per
bucket. On every rank this loader keeps the batches in one pool per bucket and, before each micro-batch, takes one
batch from the bucket that every rank holds most of: one all-reduce (MIN) of the pool sizes over a gloo group, the same
argmax on every rank (ties to the shorter bucket). So every rank trains micro-batch k on the same bucket, the pools
stay bounded (the bucket with the largest supply everywhere goes first), the buckets come at their share of the data,
and nothing is dropped. The decision depends only on the pools, which the DataLoader fills in its fixed order, so a run
is deterministic. A resume starts with empty pools on every rank.
Data passes (the hang of 2026-11-07, w1718 w1722 w1724): one data pass is one epoch of the wrapped DataLoader (its
persistent workers call iter() once per epoch), and the ranks end their passes at different micro-batches. This loader
therefore never ends: when the wrapped loader ends a pass on a rank, the rank starts the next pass inside the same
iterator, and its pools carry over. Ending the iterator would end Lightning's epoch on that rank alone, and the new
iterator would create its gloo group while the other ranks wait in the old group's all-reduce (creating a group is a
collective of every rank): the s8 and s80 arms hung at their first pass end that way. The gloo group is also made once per
loader, not once per iterator. Lightning stops at max_steps (max_epochs is null), so the whole run is one epoch.
Every 1000 micro-batches rank 0 prints the pools' largest size, the decision rounds per micro-batch and the time spent
waiting for the DataLoader and in the all-reduce (the d640 data wait at 128 x 1, docs/mfu.md; the pools' bound in the
repetition arms).
"""
import os
import time
from collections import deque

import torch
import torch.distributed as dist


def _cat(ts):
    """torch.cat of CPU tensors into pinned memory when the parts are pinned (the DataLoader pinned each sub-batch)."""
    out = torch.empty((sum(t.shape[0] for t in ts), *ts[0].shape[1:]), dtype=ts[0].dtype, pin_memory=ts[0].is_pinned())
    return torch.cat(ts, out=out)


def merge(parts):
    """One micro-batch from GPIC_SHARED_BUCKET_SUB sub-batches of one bucket (gpic-28 2026-11-09): the images stay one packed
    GPU-resize job per sub-batch (a list; training_step resizes each and concatenates, sample by sample the same pixels),
    or are concatenated when they are tensors; captions and per-sample lists are joined, tensors concatenated."""
    if len(parts) == 1:
        return parts[0]
    xs, ys, ms = zip(*parts)
    x = list(xs) if isinstance(xs[0], dict) else _cat(list(xs))
    y = [c for yy in ys for c in yy]
    m = {}
    for k in ms[0]:
        v = [mm[k] for mm in ms]
        m[k] = _cat(v) if torch.is_tensor(v[0]) else [e for vv in v for e in vv] if isinstance(v[0], list) else v[0]
    return x, y, m


def choose(min_counts):
    """The bucket every rank holds most of (the first among ties), or None if no bucket is held by every rank."""
    best = max(range(len(min_counts)), key=lambda i: (min_counts[i], -i))
    return best if min_counts[best] > 0 else None


class SharedBucketLoader:
    """Wraps the training DataLoader of GPICWebDataset(shared_bucket=True). pool_min: batches each rank keeps pooled
    before a decision (default two per bucket).
    pool_cap (GPIC_SHARED_BUCKET_POOL_CAP, default 40): the ranks draw their buckets independently, so each bucket's count
    differs between ranks like a random walk, and a rank's surplus waits until every other rank holds that bucket too: the
    pools grow like the square root of the micro-batches (49 batches within the first hour at d1536; on long segments the
    pinned pooled batches reached 41-51 GB per rank and tipped the host over at the final save, genmodel 2026-11-08, w1682).
    A rank that holds pool_cap batches takes no new batch and makes the micro-batch free: every rank takes the oldest batch
    of its own fullest bucket (each batch keeps its samples and its own padded length; that step waits for the longest
    rank, as without the shared bucket), so the full rank's pools shrink by one. Until a rank first reaches the cap the
    loader takes the very same batches as before."""

    def __init__(self, loader, edges, pool_min=None, allreduce_min=None, pool_cap=None):
        self.loader, self.edges = loader, list(edges)
        self.pool_min = pool_min or 2 * len(self.edges)
        self.pool_cap = pool_cap or int(os.environ.get("GPIC_SHARED_BUCKET_POOL_CAP", "40"))
        # GPIC_SHARED_BUCKET_SUB = k (gpic-28 2026-11-09; default 1, the loader of before): the workers emit sub-batches of
        # batch / k samples (GPICWebDataset.build_dataloader) and a micro-batch joins k sub-batches of one bucket. Every new
        # process starts with empty worker queues, and a worker emits a bucket only when its queue holds a whole batch: with
        # 32 workers bucket 320 (1.3-1.8% of the micro-batches) first came at step 2047 of a run, and a resume inside the
        # cooldown trained its last 1696 steps without it (FP8 resume test: FD +5.3%). Pools, pool_min and pool_cap count
        # micro-batches (k sub-batches each).
        self.sub = int(os.environ.get("GPIC_SHARED_BUCKET_SUB", "1"))
        self._allreduce_min = allreduce_min   # tests: a stand-in for the gloo all-reduce
        self.taken = [0] * len(self.edges)
        self.group = None   # the gloo group, made at the first decision of the run, on every rank at once
        self.passes = 0     # data passes this rank finished (restarts of the wrapped loader)

    def __iter__(self):
        return _Iter(self)

    def __getattr__(self, name):   # everything else (e.g. the dataset) from the wrapped loader
        return getattr(self.__dict__["loader"], name)


class _Iter:
    def __init__(self, parent):
        self.p, self.it = parent, None   # started at the first next(): Lightning's iter() check
        self.pools = [deque() for _ in parent.edges]          # drops an iterator without calling next()
        self.st = dict(n=0, rounds=0, pulls=0, pull_s=0.0, ar_s=0.0, pool_max=0, free=0)
        self.over, self.free_steps = False, 0   # free_steps: all of this iterator's free micro-batches (st's are per report)

    def _size(self):   # pooled micro-batches (k sub-batches each)
        return sum(len(q) for q in self.pools) / self.p.sub

    def _bucket(self, batch):
        T = batch[2]["text_mask"].shape[1]
        return self.p.edges.index(T)

    def _pull(self):
        t = time.perf_counter()
        try:
            b = next(self.it)
        except StopIteration:   # this rank's data pass ended: the next pass starts here, in the same epoch (module docstring)
            self.p.passes += 1
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            print(f"[shared_bucket] rank {rank}: data pass {self.p.passes} ended after micro-batch {self.st['n']}; "
                  f"the next pass starts with {self._size():g} batches pooled", flush=True)
            self.it = iter(self.p.loader)
            try:
                b = next(self.it)
            except StopIteration:   # a StopIteration out of __next__ would end the epoch on this rank alone: fail instead
                raise RuntimeError("the training loader gave no batch in a whole data pass") from None
        self.st["pull_s"] += time.perf_counter() - t
        self.st["pulls"] += 1
        self.pools[self._bucket(b)].append(b)

    def _min_counts(self):
        """The pools' sizes, MIN over the ranks, and one more entry: -1 if any rank holds pool_cap batches (a free step)."""
        counts = [len(q) // self.p.sub for q in self.pools] + [-1 if self.over else 0]
        if self.p._allreduce_min is not None:
            return self.p._allreduce_min(counts)
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return counts
        if self.p.group is None:
            self.p.group = dist.new_group(backend="gloo")
        t = torch.tensor(counts, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.p.group)
        return t.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        # this rank's flag for the micro-batch, from its pools before the pull: at the cap no batch comes in and the
        # micro-batch is free (_min_counts), so the pools shrink by one
        self.over = self._size() >= self.p.pool_cap
        if self.it is None:
            self.it = iter(self.p.loader)
        elif not self.over and self._size() >= self.p.pool_min:
            self._pull()   # one batch in for each one out: the decision then rarely fails (one all-reduce, not two)
        while self._size() < self.p.pool_min:
            self._pull()
        st = self.st
        st["pool_max"] = max(st["pool_max"], self._size())
        while True:
            t = time.perf_counter()
            mins = self._min_counts()
            st["ar_s"] += time.perf_counter() - t
            st["rounds"] += 1
            if mins[-1] < 0:   # a free step: this rank's fullest bucket (the first among ties)
                b = max(range(len(self.pools)), key=lambda i: (len(self.pools[i]), -i))
                while len(self.pools[b]) < self.p.sub:   # local pulls only: every rank makes the same all-reduces per micro-batch
                    self._pull()
                    b = max(range(len(self.pools)), key=lambda i: (len(self.pools[i]), -i))
                st["free"] += 1
                self.free_steps += 1
            else:
                b = choose(mins[:-1])
            if b is not None:
                self.p.taken[b] += 1
                st["n"] += 1
                if st["n"] % 1000 == 0:
                    self._report()
                return merge([self.pools[b].popleft() for _ in range(self.p.sub)])
            self._pull()   # no bucket on every rank yet: one more batch, then decide again

    def _report(self):
        st = self.st
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            n = 1000
            print(f"[shared_bucket] rank 0, micro-batches {st['n'] - n + 1}-{st['n']}: largest pool {st['pool_max']} batches "
                  f"(cap {self.p.pool_cap}, {st['free']} free steps), "
                  f"{st['rounds'] / n:.2f} decisions and {st['pulls'] / n:.2f} pulls per micro-batch, waits per micro-batch: "
                  f"DataLoader {1000 * st['pull_s'] / n:.1f} ms, all-reduce {1000 * st['ar_s'] / n:.1f} ms", flush=True)
        st.update(rounds=0, pulls=0, pull_s=0.0, ar_s=0.0, pool_max=0, free=0)
