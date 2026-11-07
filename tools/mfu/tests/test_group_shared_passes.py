#!/usr/bin/env python3
"""GPICWebDataset._group_shared across data passes (one pass = one epoch of a persistent worker, so the stage runs once
per pass on the same dataset object): the unfinished batches of a pass must reach the next pass. Three finite passes of
synthetic samples with the measured length mix: every sample of the three passes leaves in a full batch of its bucket,
except the ones still queued after the last pass (fewer than one batch per bucket); without the carry-over every pass
end would drop its queues.
usage: python test_group_shared_passes.py"""
import os, sys, bisect
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.data.dataset.gpic_dataset import GPICWebDataset

EDGES = (16, 24, 32, 40, 48, 56, 64, 144, 176, 320)
P = np.array([.0996, .2737, .0896, .1052, .1628, .1344, .0380, .0270, .0570, .0127])
ds = object.__new__(GPICWebDataset)
ds.group_batch, ds.pad_multiple, ds.text_max_length, ds.bucket_edges = 128, 8, 320, EDGES
rng = np.random.default_rng(0)
seen, out, ok = [], [], True
for p in range(3):
    lens = [int(EDGES[b]) - int(rng.integers(0, 8)) for b in rng.choice(len(EDGES), size=12600, p=P / P.sum())]
    src = [{"text_len": L, "id": (p, i)} for i, L in enumerate(lens)]
    seen += [s["id"] for s in src]
    got = list(ds._group_shared(iter(src)))
    ok &= len(got) % 128 == 0
    for k in range(0, len(got), 128):
        b = got[k:k + 128]
        T = b[0]["pad_to"]
        lo = EDGES[EDGES.index(T) - 1] if EDGES.index(T) else 0
        ok &= all(x["pad_to"] == T and lo < min(-(-x["text_len"] // 8) * 8, 320) <= T for x in b)
    out += [s["id"] for s in got]
qs = ds.__dict__.get("_shared_queues", [])   # the code before the carry-over keeps no queues on the dataset
queued = sum(len(q) for q in qs)
ids_out = set(out)
lost = len(set(seen) - ids_out - {s["id"] for q in qs for s in q})
ok &= len(ids_out) == len(out) and lost == 0 and all(len(q) < 128 for q in qs)
print(f"3 passes of 12600 samples: {len(out)} trained in full batches, {queued} still queued (< 128 per bucket), "
      f"lost at the pass ends: {lost}; none twice: {len(ids_out) == len(out)}")
print("GROUP_SHARED_PASSES_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
