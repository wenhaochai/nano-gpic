#!/usr/bin/env python3
# Means over seeds of toy_code.py, with the point arm of res4 as reference. python table_code.py
import collections, glob, json
import numpy as np
rows, kl, lam = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
extra = collections.defaultdict(list)
files = glob.glob("res6/*.json") + [f for f in glob.glob("res4/*_point_s*.json")] + glob.glob("res4/*_gauss_r0d_s*.json")
for f in files:
    for r in json.load(open(f)):
        kl[(r["dataset"], r["arm"], r["step"])].append(r.get("kl") or 0)
        lam[(r["dataset"], r["arm"], r["step"])].append(r.get("lam") or 0)
        for g in r["gen"]:
            k = (r["dataset"], r["arm"], g["mode"], g["grid"], g["calls"], r["step"])
            rows[k].append(g["ed"])
            for e in ("spread", "coherent", "dist"):
                if e in g:
                    extra[k + (e,)].append(g[e])
arms = sorted({k[1:3] for k in rows}, key=lambda a: (not a[0].startswith("point"), not a[0].startswith("gauss"), a))
DS = ("coupled", "manifold", "embedded")
def get(d, a, m, grid, calls, step):
    v = rows.get((d, a, m, grid, calls, step))
    return np.mean(v) if v else float("nan")
for title, grid, calls, step in (("50 calls logit, step 500", "logit", 50, 500), ("50 calls logit, step 1000", "logit", 50, 1000), ("50 calls logit, step 2000", "logit", 50, 2000), ("50 calls logit, step 5000", "logit", 50, 5000), ("50 calls uniform, step 10000", "uniform", 50, 10000), ("50 calls logit, step 10000", "logit", 50, 10000), ("8 calls uniform, step 10000", "uniform", 8, 10000), ("1 call, step 10000", "uniform", 1, 10000), ("50 calls uniform, step 500", "uniform", 50, 500), ("50 calls uniform, step 1000", "uniform", 50, 1000), ("50 calls uniform, step 2000", "uniform", 50, 2000), ("50 calls uniform, step 5000", "uniform", 50, 5000)):
    print("===", title)
    print(f"{'arm':24s} {'mode':10s} " + " ".join(f"{d:>9s}" for d in DS))
    for a, m in arms:
        print(f"{a:24s} {m:10s} " + " ".join(f"{get(d, a, m, grid, calls, step):9.4f}" for d in DS))
print("=== KL nats and lambda at step 10000")
for a in sorted({k[1] for k in kl if k[1].startswith("code")}):
    print(f"{a:24s} " + " ".join(f"{np.mean(kl.get((d, a, 10000), [float('nan')])):7.2f}/{np.mean(lam.get((d, a, 10000), [float('nan')])):6.3f}" for d in DS))
print("=== coupled spread and coherent, step 10000, 50 calls uniform")
for a, m in arms:
    k = ("coupled", a, m, "uniform", 50, 10000)
    print(f"{a:24s} {m:10s} spread {np.mean(extra.get(k + ('spread',), [float('nan')])):6.3f} coherent {np.mean(extra.get(k + ('coherent',), [float('nan')])):6.3f}")
