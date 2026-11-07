#!/usr/bin/env python3
# The loader's buffers per arm and at each stall of the late rank. python pin_queue.py <bench dir> <arm>...
import glob, json, re, sys
import numpy as np
for arm in sys.argv[2:]:
    R = {}
    for f in sorted(glob.glob(f"{sys.argv[1]}/{arm}/steps_rank*.json")):
        r = int(re.search(r"rank(\d+)", f).group(1))
        R[r] = [x for x in json.load(open(f))["rows"] if "t_start" in x and x["step"] >= 100]
    all_rows = [x for r in R.values() for x in r]
    print(f"== {arm}: q_ready mean {np.mean([x['q_ready'] for x in all_rows]):.1f} q_raw mean {np.mean([x['q_raw'] for x in all_rows]):.1f} q_ahead mean {np.mean([x['q_ahead'] for x in all_rows]):.1f}; share of steps with q_ready = 0 {np.mean([x['q_ready'] == 0 for x in all_rows]):.3f}; data wait mean ms {1e3 * np.mean([x['data_wait_s'] for x in all_rows]):.2f}")
    rows = []
    for r, xs in R.items():
        for a, b in zip(xs, xs[1:]):
            if b["data_wait_s"] > 0.15 and b["step"] % 1000 != 0:
                rows.append((b["data_wait_s"] * 1e3, r, b["step"], b["q_ready"], b["q_raw"], b["q_ahead"], (b["delay_main_ns"] - a["delay_main_ns"]) / 1e6, (b["delay_pin_ns"] - a["delay_pin_ns"]) / 1e6, (b["q_rcvd"] - 1) % 32))
    rows.sort(key=lambda x: x[2])
    print(f"   data waits over 150 ms: {len(rows)}; q_ahead median {np.median([x[5] for x in rows]) if rows else 0}; q_ready median {np.median([x[3] for x in rows]) if rows else 0}; q_raw median {np.median([x[4] for x in rows]) if rows else 0}; main delay ms median {np.median([x[6] for x in rows]) if rows else 0:.1f}; pin delay ms median {np.median([x[7] for x in rows]) if rows else 0:.1f}")
    for x in rows[:10]:
        print("   wait %6.0f ms rank %d step %5d ready %3d raw %3d ahead %3d main delay %6.1f pin delay %6.1f worker %2d" % x)
    w = {}
    for x in rows:
        w.setdefault((x[1], x[8]), []).append(x[0])
    print("   worker repeats:", sorted(((k, len(v)) for k, v in w.items()), key=lambda kv: -kv[1])[:6])
