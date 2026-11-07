#!/usr/bin/env python3
# Means over seeds of toy_design.py. python table_design.py <res dir>
import collections, glob, json, sys
import numpy as np
rows = collections.defaultdict(list)
kl = collections.defaultdict(list)
extra = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] + "/*.json"):
    try:
        data = json.load(open(f))
    except ValueError:
        continue
    for r in data:
        if r.get("kl") is not None:
            kl[(r["dataset"], r["arm"], r["step"])].append(r["kl"])
        for g in r["gen"]:
            key = (r["dataset"], r["arm"], g["mode"], g["grid"], g["calls"], r["step"])
            rows[key].append(g["ed"])
            for k in ("spread", "coherent", "dist", "coverage_min"):
                if k in g:
                    extra[key + (k,)].append(g[k])
arms = sorted({k[1:3] for k in rows}, key=lambda a: (a[0].split("_")[0] != "point", a))
DS = ("coupled", "manifold", "embedded")
def get(d, a, m, grid, calls, step):
    v = rows.get((d, a, m, grid, calls, step))
    return np.mean(v) if v else float("nan")
for title, grid, calls, step in (("50 calls uniform, step 10000", "uniform", 50, 10000), ("50 calls logit grid, step 10000", "logit", 50, 10000), ("8 calls uniform, step 10000", "uniform", 8, 10000), ("8 calls logit grid, step 10000", "logit", 8, 10000), ("2 calls, step 10000", "uniform", 2, 10000), ("1 call, step 10000", "uniform", 1, 10000), ("50 calls uniform, step 500", "uniform", 50, 500), ("50 calls uniform, step 1000", "uniform", 50, 1000), ("50 calls uniform, step 2000", "uniform", 50, 2000), ("50 calls uniform, step 5000", "uniform", 50, 5000)):
    print("===", title)
    print(f"{'arm':18s} {'mode':7s} " + " ".join(f"{d:>9s}" for d in DS))
    for a, m in arms:
        print(f"{a:18s} {m:7s} " + " ".join(f"{get(d, a, m, grid, calls, step):9.4f}" for d in DS))
print("=== KL nats at step 10000")
for a in sorted({k[1] for k in kl}):
    print(f"{a:18s} " + " ".join(f"{np.mean(kl.get((d, a, 10000), [float('nan')])):9.2f}" for d in DS))
print("=== coupled spread and coherent, step 10000, 50 calls uniform")
for a, m in arms:
    k = ("coupled", a, m, "uniform", 50, 10000)
    print(f"{a:18s} {m:7s} spread {np.mean(extra.get(k + ('spread',), [float('nan')])):6.3f} coherent {np.mean(extra.get(k + ('coherent',), [float('nan')])):6.3f}")
print("=== coupled spread, step 1000, 50 calls uniform")
for a, m in arms:
    k = ("coupled", a, m, "uniform", 50, 1000)
    print(f"{a:18s} {m:7s} spread {np.mean(extra.get(k + ('spread',), [float('nan')])):6.3f} coherent {np.mean(extra.get(k + ('coherent',), [float('nan')])):6.3f}")
