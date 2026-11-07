#!/usr/bin/env python3
"""Where an 8-GPU step's time goes, from every rank's torch.profiler trace (src/callbacks/bench.py BENCH_PROFILE writes
trace/rank<r>.json; the ranks of one node share the host clock).

Per profiled step (ProfilerStep#N windows, merged over streams) and rank: the span, the compute busy time (union of the
non-NCCL kernels), the NCCL kernel time, the exposed NCCL time (NCCL kernel time with no compute kernel running beside
it), and the idle time (no kernel at all). Across ranks: the compute imbalance (slowest rank's busy time minus the mean),
which a rank spends waiting in its collectives. Also the first and last kernel of the step per rank, to show who starts
late (host or data) and who ends last.
usage: python rank_anatomy.py <trace dir>
"""
import glob, json, os, re, statistics as st, sys


def union(iv):
    iv = sorted(iv)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def clip(iv, w0, w1):
    return [(max(s, w0), min(e, w1)) for s, e in iv if e > w0 and s < w1]


def minus(a, b):
    """Total length of the intervals a not covered by the intervals b."""
    return union(a + b) - union(b)


def load(f):
    ev = json.load(open(f))["traceEvents"]
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    steps = {}
    for e in ev:
        if e.get("cat") == "gpu_user_annotation" and e.get("name", "").startswith("ProfilerStep#") and "dur" in e:
            s0, e0 = steps.get(e["name"], (e["ts"], e["ts"] + e["dur"]))
            steps[e["name"]] = (min(s0, e["ts"]), max(e0, e["ts"] + e["dur"]))
    if not steps:   # no GPU-side step annotations (Lightning's profiler): a step's kernels are those launched inside its
        # CPU-side ProfilerStep window (runtime call -> kernel by correlation id); its GPU window runs from its first kernel
        # to the next step's first kernel
        cpu = sorted([e for e in ev if e.get("cat") == "user_annotation" and e.get("name", "").startswith("ProfilerStep#")
                      and "dur" in e], key=lambda e: e["ts"])
        kc = {e["args"]["correlation"]: e for e in k if "correlation" in e.get("args", {})}
        first = {}
        for e in ev:
            if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
                ke = kc.get(e["args"]["correlation"])
                if ke is None:
                    continue
                for c in cpu:
                    if c["ts"] <= e["ts"] < c["ts"] + c["dur"]:
                        first[c["name"]] = min(first.get(c["name"], ke["ts"]), ke["ts"])
                        break
        names = [c["name"] for c in cpu if c["name"] in first]
        end = max(e["ts"] + e["dur"] for e in k)
        for a, b in zip(names, names[1:] + [None]):
            steps[a] = (first[a], first[b] if b else end)
        if names:   # the last window ends at the last kernel, which may include the profiler's own flush: drop it
            steps.pop(names[-1])
    nccl = [(e["ts"], e["ts"] + e["dur"]) for e in k if re.search("nccl", e["name"], re.I)]
    comp = [(e["ts"], e["ts"] + e["dur"]) for e in k if not re.search("nccl", e["name"], re.I)]
    return steps, comp, nccl


def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, "rank*.json")), key=lambda f: int(re.findall(r"rank(\d+)", f)[-1]))
    R = {int(re.findall(r"rank(\d+)", f)[-1]): load(f) for f in files}
    names = sorted(set.intersection(*[set(v[0]) for v in R.values()]), key=lambda s: int(s.split("#")[1]))
    print(f"{d}: {len(R)} ranks, steps {names}")
    rows = {r: [] for r in R}
    imb, spans = [], []
    for n in names:
        busy = {}
        for r, (steps, comp, nccl) in R.items():
            w0, w1 = steps[n]
            c, x = clip(comp, w0, w1), clip(nccl, w0, w1)
            b = union(c)
            busy[r] = b
            first = min((s for s, _ in c), default=w0) - w0
            rows[r].append({"span": w1 - w0, "busy": b, "nccl": union(x), "exposed": minus(x, c),
                            "idle": (w1 - w0) - union(c + x), "first": first})
        mx, mean = max(busy.values()), st.mean(busy.values())
        imb.append(mx - mean)
        spans.append(st.mean(rows[r][-1]["span"] for r in R))
    print(f"{'rank':>4s} {'span':>8s} {'busy':>8s} {'nccl':>8s} {'exposed':>8s} {'idle':>8s} {'1st kern':>8s}  (ms, mean over steps)")
    for r in sorted(R):
        m = {k: st.mean(x[k] for x in rows[r]) / 1e3 for k in rows[r][0]}
        print(f"{r:4d} {m['span']:8.2f} {m['busy']:8.2f} {m['nccl']:8.2f} {m['exposed']:8.2f} {m['idle']:8.2f} {m['first']:8.2f}")
    print(f"compute imbalance per step (slowest rank's busy minus the mean): {[round(v / 1e3, 2) for v in imb]} ms, "
          f"mean {st.mean(imb) / 1e3:.2f} ms of a {st.mean(spans) / 1e3:.2f} ms span")


if __name__ == "__main__":
    main()
