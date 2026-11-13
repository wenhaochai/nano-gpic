#!/usr/bin/env python3
# Tables of the toy runs, mean over seeds (seed count printed). usage: table.py <res dir>
import collections, glob, json, sys
rows = collections.defaultdict(dict)
for f in glob.glob(sys.argv[1] + "/*_s*.json"):
    for x in json.load(open(f)):
        rows[(x["dataset"], x["arm"], x["seed"])][x["step"]] = x
ARMS = ["point", "wiener", "tangent", "tangent_mix", "mix_t", "mix_global", "mix_global_multi", "mix_multi", "mix_nll"]
def avg(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")
for d in sorted({k[0] for k in rows}):
    seeds = {a: sorted(k[2] for k in rows if k[0] == d and k[1] == a) for a in ARMS}
    steps = sorted({s for k, v in rows.items() if k[0] == d for s in v})
    print(f"\n== {d}; seeds per arm {[len(seeds[a]) for a in ARMS]}\nenergy distance, 50 backbone calls, by training steps\n{'arm':10s} " + " ".join(f"{s:>7d}" for s in steps))
    def get(a, s, c, u):
        return [next((g for g in rows[(d, a, sd)].get(s, {"gen": []})["gen"] if g["calls"] == c and g["sub"] == u), None) for sd in seeds[a]]
    for a in ARMS:
        print(f"{a:16s} " + " ".join(f"{avg([g['ed'] if g else None for g in get(a, s, 50, 1)]):7.4f}" for s in steps))
    last = steps[-1]
    print(f"at step {last}, by calls/substeps: energy distance and other measures")
    keys = [k for k in next(g for g in get(ARMS[0], last, 50, 1) if g) if k not in ("calls", "sub", "ed")]
    print("  " + ", ".join(keys))
    for a in ARMS:
        for c, u in ((50, 1), (8, 1), (8, 16), (1, 16)):
            g = [x for x in get(a, last, c, u) if x]
            if g:
                print(f"{a:16s} {c:3d}/{u:<3d} ed {avg([x['ed'] for x in g]):.4f}  " + " ".join(f"{avg([x[k] for x in g]):.3f}" for k in keys))
    print("posterior mean squared error at t = 0.1, 0.3, 0.5, 0.7, 0.9 (higher t: less noise)")
    for a in ARMS:
        v = [rows[(d, a, sd)][last]["val"] for sd in seeds[a] if last in rows[(d, a, sd)]]
        if v:
            print(f"{a:10s} " + " ".join(f"{avg([x[k] for x in v]):.5f}" for k in v[0]))
