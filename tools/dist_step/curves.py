#!/usr/bin/env python3
# One measure at 50 backbone calls by training steps, mean over seeds. usage: curves.py <res dir> <dataset> <key>
import collections, glob, json, sys
d, key = sys.argv[2], sys.argv[3]
out = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(f"{sys.argv[1]}/{d}_*_s*.json"):
    for x in json.load(open(f)):
        g = next((g for g in x["gen"] if g["calls"] == 50 and g["sub"] == 1), None)
        if g:
            out[x["arm"]][x["step"]].append(g[key])
steps = sorted({s for a in out.values() for s in a})
print(d, key, "\narm        " + " ".join(f"{s:>7d}" for s in steps))
for a in ("point", "wiener", "tangent", "tangent_mix", "mix_t", "mix_multi", "mix_global", "mix_nll"):
    if a in out:
        print(f"{a:10s} " + " ".join(f"{sum(out[a][s]) / max(1, len(out[a][s])):7.3f}" for s in steps))
