#!/usr/bin/env python3
# Cost of the periodic steps of each arm: excess over the median of the 10 steps around it. python period_cost.py <bench dir> <arm>...
import json, sys
import numpy as np
print(f"{'arm':18s} {'all':>6s} {'median':>6s} | {'%200 n':>6s} {'ms':>6s} | {'%100 n':>6s} {'ms':>6s} | {'%50 n':>6s} {'ms':>6s} | {'other>1.6med n':>14s} {'ms/step':>7s} | {'periodic ms/step':>16s} {'mean without':>12s}")
for arm in sys.argv[2:]:
    r = [x for x in json.load(open(f"{sys.argv[1]}/{arm}/steps_rank0.json"))["rows"] if "t_start" in x and x["step"] >= 100]
    g = np.array([b["t_start"] - a["t_start"] for a, b in zip(r, r[1:])]) * 1e3
    s = np.array([a["step"] for a in r[:-1]])
    keep = g < 1000            # the checkpoint saves are not steps
    g, s = g[keep], s[keep]
    med = np.median(g)
    ex = np.zeros(len(g))
    for i in range(len(g)):
        lo, hi = max(0, i - 5), min(len(g), i + 6)
        nb = np.r_[g[lo:i], g[i + 1:hi]]
        ex[i] = g[i] - np.median(nb)
    n = len(g)
    m200 = (s + 1) % 200 == 0
    m100 = ((s + 1) % 100 == 0) & ~m200
    m50 = ((s + 1) % 50 == 0) & ~m200 & ~m100
    oth = ~(m200 | m100 | m50)
    slow = oth & (g > med * 1.6)
    per = (ex[m200].sum() + ex[m100].sum() + ex[m50].sum()) / n
    print(f"{arm:18s} {g.mean():6.1f} {med:6.1f} | {m200.sum():6d} {ex[m200].mean():6.1f} | {m100.sum():6d} {ex[m100].mean():6.1f} | {m50.sum():6d} {ex[m50].mean():6.1f} | {slow.sum():14d} {(g[slow]-med).sum()/n:7.2f} | {per:16.2f} {g.mean()-per:12.2f}")
