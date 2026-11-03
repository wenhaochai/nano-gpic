"""Table of the arms of jobs/prof_ladder.sh: step time and each stage's GPU span and host time, from BENCH_STAGES.

python tools/prof/ladder_stages.py <run dir>
Over the last 200 steps of each arm, or the last half if it has fewer: median and mean of start-to-start step time;
median stage spans on rank 0, GPU ms / host ms; the slowest stage median over all ranks; recompilations after the warm-up.
"""
import glob
import json
import os
import statistics as st
import sys


def main():
    out = sys.argv[1]
    for arm in sorted(os.listdir(os.path.join(out, "bench"))):
        d = os.path.join(out, "bench", arm)
        try:
            rows = [r for r in json.load(open(os.path.join(d, "steps_rank0.json")))["rows"] if "step_ms" in r]
        except (OSError, ValueError) as e:
            print(f"{arm}: no timing ({e!r})")
            continue
        w = rows[-min(200, len(rows) // 2):]
        steps = [r["step_ms"] for r in w]
        log = os.path.join(out, f"{arm}.log")
        late = open(log, errors="replace").read().split("[warmup] done")[-1].count("Recompiling function") if os.path.exists(log) else "?"
        print(f"\n{arm}: steps {len(rows)}; step ms median {st.median(steps):.1f} mean {st.fmean(steps):.1f}; "
              f"peak GB {w[-1]['max_alloc_gb']:.1f}; recompilations after the warm-up {late}")
        files = sorted(glob.glob(os.path.join(d, "stages_rank*.json")))
        if not files:
            continue
        per_rank = [json.load(open(f))[-len(w):] for f in files]
        keys = [k for k in per_rank[0][-1] if all(k in r for r in per_rank[0][-20:])]
        print(f"  {'stage':22s} {'gpu':>7s} {'host':>7s} {'gpu, slowest rank':>18s}")
        total = 0.0
        for k in keys:
            gpu = [st.median(r[k][0] for r in rows if k in r) for rows in per_rank]
            host = st.median(r[k][1] for r in per_rank[0] if k in r)
            total += gpu[0]
            print(f"  {k:22s} {gpu[0]:7.2f} {host:7.2f} {max(gpu):18.2f}")
        print(f"  {'sum':22s} {total:7.2f}")


if __name__ == "__main__":
    main()
