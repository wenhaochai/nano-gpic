import collections, glob, json
import numpy as np
rows, extra = collections.defaultdict(list), collections.defaultdict(list)
files = glob.glob("res6/*.json") + glob.glob("res4/*_point_s*.json")
for f in files:
    for r in json.load(open(f)):
        for g in r["gen"]:
            k = (r["dataset"], r["arm"], g["mode"], g["grid"], g["calls"], r["step"])
            rows[k].append(g["ed"])
            for e in ("spread", "coherent"):
                if e in g:
                    extra[k + (e,)].append(g[e])
DS = ("coupled", "manifold", "embedded")
arms = [("point", "mean")] + [(f"code_chan_late_r{r}", m) for r in (1, 2, 4, 8, 16) for m in ("prior", "gauss")] + [("code_chan_in_r4", "prior")] + [("code_x_late_r4", "prior"), ("code_x_late_r8", "prior"), ("code_x_late_l0.01", "prior"), ("code_x_late_l0.1", "prior"), ("code_x_late_l1", "prior")]
def cell(d, a, m, grid, calls, step):
    v = rows.get((d, a, m, grid, calls, step))
    return f"{np.mean(v):7.4f}+-{np.std(v):6.4f}" if v else " " * 15
for title, grid, calls, step in (("50 uniform, step 500", "uniform", 50, 500), ("50 uniform, step 1000", "uniform", 50, 1000), ("50 uniform, step 2000", "uniform", 50, 2000), ("50 uniform, step 5000", "uniform", 50, 5000), ("50 uniform, step 10000", "uniform", 50, 10000), ("50 logit, step 500", "logit", 50, 500), ("50 logit, step 1000", "logit", 50, 1000), ("50 logit, step 2000", "logit", 50, 2000), ("50 logit, step 5000", "logit", 50, 5000), ("50 logit, step 10000", "logit", 50, 10000), ("8 uniform, step 10000", "uniform", 8, 10000), ("1 call, step 10000", "uniform", 1, 10000)):
    print("===", title)
    for a, m in arms:
        print(f"{a:22s} {m:6s} " + " ".join(cell(d, a, m, grid, calls, step) for d in DS))
print("=== coupled spread and coherent, 50 uniform, step 10000")
for a, m in arms:
    k = ("coupled", a, m, "uniform", 50, 10000)
    print(f"{a:22s} {m:6s} spread {np.mean(extra.get(k + ('spread',), [float('nan')])):6.3f} coherent {np.mean(extra.get(k + ('coherent',), [float('nan')])):6.3f}")
