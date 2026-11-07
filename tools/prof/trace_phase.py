#!/usr/bin/env python3
# GPU time and kernels per rank-step by launching op: which part is the optimizer, EMA and gradient clip. python trace_phase.py <store.db> [skip]
import collections, re, sqlite3, sys
con = sqlite3.connect(sys.argv[1])
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 1
steps = con.execute("select rank, idx, ts_ns, dur_ns from steps order by rank, idx").fetchall()
idxs = sorted({r[1] for r in steps})[skip:]
kept = [s for s in steps if s[1] in idxs]
GPU = "('kernel','comm','memcpy','memset')"
rows = collections.defaultdict(lambda: [0, 0.0])
for rank, idx, ts, dur in kept:
    for name, d, corr in con.execute(f"select name, dur_ns, corr from events where rank=? and cat in {GPU} and ts_ns>=? and ts_ns<?", (rank, ts, ts + dur)):
        op = con.execute("select op from kernel_op where rank=? and corr=?", (rank, corr)).fetchone()
        rows[(op[0] if op else "?")[:50]][0] += 1
        rows[(op[0] if op else "?")[:50]][1] += d / 1e6
n = len(kept)
tot = sum(v[1] for v in rows.values())
print(f"{n} rank-steps, step wall mean {sum(s[3] for s in kept)/n/1e6:.1f} ms, kernel sum {tot/n:.2f} ms, kernels {sum(v[0] for v in rows.values())/n:.0f}")
for k, v in sorted(rows.items(), key=lambda x: -x[1][1])[:16]:
    print(f"  {k:50s} {v[0]/n:7.1f} {v[1]/n:7.2f} ms  {100*v[1]/tot:5.1f}%  {1e3*v[1]/max(v[0],1):7.1f} us each")
