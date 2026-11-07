#!/usr/bin/env python3
# Where the late rank lost its time: for each long step, the late rank's own largest start-to-start gap in the 8 steps before it and its parts.
# python pin_gap.py <bench arm dir> [ms]
import glob, json, re, sys
import numpy as np
arm, ms = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 300
R = {}
for f in sorted(glob.glob(f"{arm}/steps_rank*.json")):
    r = int(re.search(r"rank(\d+)", f).group(1))
    R[r] = {x["step"]: x for x in json.load(open(f))["rows"] if "t_start" in x}
steps = sorted(R[0])
base = {r: np.median([R[r][s]["t_start"] - R[0][s]["t_start"] for s in steps[100:]]) for r in R}
rows = []
for a, b in zip(steps, steps[1:]):
    g = (R[0][b]["t_start"] - R[0][a]["t_start"]) * 1e3
    if g <= ms or g >= 1500 or a < 12:
        continue
    lag = {r: (R[r][a]["t_start"] - R[0][a]["t_start"] - base[r]) * 1e3 for r in R}
    l = max(lag, key=lag.get)
    gaps = [(s, (R[l][s + 1]["t_start"] - R[l][s]["t_start"]) * 1e3, R[l][s + 1]["data_wait_s"] * 1e3, R[l][s]["host_s"] * 1e3) for s in range(a - 8, a + 1) if s + 1 in R[l]]
    big = max(gaps, key=lambda x: x[1])
    rows.append((a, g, l, lag[l], big))
    print(f"long step {a:5d} {g:6.0f} ms: rank {l} lags others by {lag[l]:6.0f} ms; its largest gap: step {big[0]} ({big[0]-a:+d}) {big[1]:6.0f} ms, data wait {big[2]:6.1f}, host {big[3]:6.1f}")
print("late rank counts", {r: sum(1 for x in rows if x[2] == r) for r in R})
print("sum of long steps s", round(sum(x[1] for x in rows) / 1e3, 2), "in data wait of the late rank s", round(sum(x[4][2] for x in rows) / 1e3, 2), "in its host time s", round(sum(x[4][3] for x in rows) / 1e3, 2))
