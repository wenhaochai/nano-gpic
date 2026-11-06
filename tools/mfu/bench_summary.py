#!/usr/bin/env python3
"""Step time and MFU of benchmark arms (jobs/mfu/arm.sh), from the step timer's steps_rank<r>.json (src/callbacks/bench.py).

Step time: start to start of consecutive steps (CUDA events), averaged over ranks per step, over the steps after --skip
and outside a profiled window. The mean is the number that counts (a sync spike is repaid by the next step); median,
p90 and the count of steps over 2x the median show the shape.
MFU = training GFLOP/img (3 x forward, mean padded text length 51.5, docs/mfu.md) x 256 / 8 GPUs / step / 989.4 TFLOP/s.
usage: python bench_summary.py <arm dir or bench dir>... [--skip 20] [--json out.json]
"""
import argparse, glob, json, os, re, statistics

GF = {640: 104.1, 768: 163.8, 896: 242.1, 1024: 346.9, 1280: 622.4, 1536: 1024.9, 1792: 1554.8}
PEAK = 989.4e12


def width_of(path):
    for f in [os.path.join(path, "..", "train.log"), os.path.join(path, "train.log")]:
        if os.path.exists(f):
            with open(f, errors="replace") as fh:
                for line in fh:
                    m = re.search(r"hidden_size[\"']?[:=]\s*(\d+)", line)
                    if m:
                        return int(m.group(1))
    m = re.search(r"d(\d{3,4})", path)
    return int(m.group(1)) if m else None


def summarize(path, skip, prof):
    bench = path if glob.glob(os.path.join(path, "steps_rank*.json")) else os.path.join(path, "bench")
    files = sorted(glob.glob(os.path.join(bench, "steps_rank*.json")))
    if not files:
        return None
    per_step = {}
    waits = []
    peak = 0
    for f in files:
        rows = json.load(open(f))["rows"]
        for r in rows:
            if "step_ms" in r and r["step"] >= skip and not (prof and prof[0] - 1 <= r["step"] < prof[1] + 1):
                per_step.setdefault(r["step"], []).append(r["step_ms"])
                waits.append(r["data_wait_s"] * 1e3)
            peak = max(peak, r.get("max_reserved_gb", 0))
    steps = [statistics.mean(v) for k, v in sorted(per_step.items()) if len(v) == len(files)]
    if not steps:
        return None
    med = statistics.median(steps)
    s = sorted(steps)
    d = width_of(bench)
    mean = statistics.mean(steps)
    out = {"arm": os.path.relpath(path), "ranks": len(files), "n": len(steps), "mean_ms": mean, "median_ms": med,
           "p90_ms": s[int(0.9 * (len(s) - 1))], "slow": sum(x > 2 * med for x in steps),
           "data_wait_ms": statistics.mean(waits), "peak_reserved_gb": peak, "d": d}
    if d in GF:
        out["mfu_mean"] = GF[d] * 1e9 * 32 / (mean / 1e3) / PEAK
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--skip", type=int, default=20)
    ap.add_argument("--prof", default=None, help="a:b, the profiled steps to leave out")
    ap.add_argument("--json")
    a = ap.parse_args()
    prof = tuple(int(x) for x in a.prof.split(":")) if a.prof else None
    res = [r for r in (summarize(p, a.skip, prof) for p in a.paths) if r]
    print(f"{'arm':44s} {'d':>5s} {'n':>4s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'slow':>4s} {'wait':>6s} {'GiB':>5s} {'MFU':>6s}")
    for r in res:
        print(f"{r['arm'][-44:]:44s} {r['d'] or 0:5d} {r['n']:4d} {r['mean_ms']:7.2f} {r['median_ms']:7.2f} {r['p90_ms']:7.2f} "
              f"{r['slow']:4d} {r['data_wait_ms']:6.2f} {r['peak_reserved_gb']:5.1f} {100 * r.get('mfu_mean', float('nan')):5.1f}%")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
