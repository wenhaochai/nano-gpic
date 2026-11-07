#!/usr/bin/env python3
# Is the cost of a periodic step repaid by the steps after it? Mean step time by offset from the periodic step, against the median of all steps away from it.
# python period_profile.py <bench dir> <period> <skip multiples of> <arm>...
import json, sys
import numpy as np
period, skip = int(sys.argv[2]), int(sys.argv[3])
for arm in sys.argv[4:]:
    r = [x for x in json.load(open(f"{sys.argv[1]}/{arm}/steps_rank0.json"))["rows"] if "t_start" in x and x["step"] >= 100]
    g = {a["step"]: (b["t_start"] - a["t_start"]) * 1e3 for a, b in zip(r, r[1:]) if b["step"] == a["step"] + 1}
    pts = [s for s in g if (s + 1) % period == 0 and (skip == 0 or (s + 1) % skip != 0)]
    far = [v for s, v in g.items() if v < 1000 and min((s + 1) % 50, 50 - (s + 1) % 50) >= 12]
    base, base_med = np.mean(far), np.median(far)
    prof = []
    for off in range(-3, 11):
        v = [g[s + off] for s in pts if s + off in g and g[s + off] < 1000]
        prof.append(np.mean(v) if v else np.nan)
    prof = np.array(prof)
    print(f"{arm:18s} n={len(pts):3d} far mean {base:6.2f} far median {base_med:6.2f}")
    print("   offset " + " ".join(f"{o:6d}" for o in range(-3, 11)))
    print("   ms     " + " ".join(f"{p:6.1f}" for p in prof))
    print("   excess " + " ".join(f"{p-base:6.1f}" for p in prof) + f"   sum(0..+10) {np.nansum(prof[3:]-base):7.1f} ms")
