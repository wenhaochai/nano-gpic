"""Checks of jobs/smoke_repeat.sh (repetition arms, owner 2026-11-07). Run in baselines/PixelGen with PYTHONPATH=$PWD.
usage: check_repeat_smoke.py <smoke dir>
1. workers: in launch a every rank started 32 entries of pass 0, one per data worker (each worker has one entry per pass
   on the 8-shard subset with 4 ranks x 32 workers), exactly the rank's entries of the pass-0 schedule, each once
2. step time: the median start-to-start step time of the arm (launch a, steps 20-399 without the save steps) is within 10%
   of the Full path's (steps 20-299) on the same GPUs; the mean data wait per step of both
3. resume: launch b cut the log at the dump time in its step-300 checkpoint and kept launch a's lines up to it; every
   pass-0 entry was started before the checkpoint, so launch b resumed in pass 1 and skipped all of them: it started no
   pass-0 entry, and started each of the rank's 32 pass-1 entries once; every rank printed its resume point
Exit 1 when a check fails.
"""
import glob
import json
import os
import re
import statistics
import sys

import braceexpand

from src.data.dataset.gpic_dataset import EpochShuffledShards, make_shard_spec

TAR_DIR = "/scratch/gpfs/GROUP/USER/data/gpic/train/"
PATTERN = "gpic_train_{00000..00007}.tar"
WORLD, NW = 4, 32
RESUME_STEP = 300
SAVES = (200, 300, 400)   # launch a: ms at 200 and 400, last.ckpt at 300 (C0 = 800 steps, last every 300)


def lines(p):
    return [x.rstrip("\n") for x in open(p)] if os.path.exists(p) else []


def schedule(urls, seed, r, p):
    os.environ.update(RANK=str(r), WORLD_SIZE=str(WORLD))
    c = EpochShuffledShards(urls, seed, workers=NW)
    c.load_index()
    return [e[0] for e in c._schedule(p)[r::WORLD]], c.k


def median_step(d, lo, hi):
    fs = sorted(glob.glob(os.path.join(d, "steps_rank*.json")))
    med, wait = [], []
    for p in fs:
        rows = [x for x in json.load(open(p))["rows"] if x.get("step_ms") is not None]
        base = [x for x in rows if lo <= x["step"] < hi and not any(abs(x["step"] - s) <= 1 for s in SAVES)]
        med.append(statistics.median(x["step_ms"] for x in base))
        wait.append(statistics.mean(x["data_wait_s"] for x in base) * 1e3)
    return len(fs), med, wait


def main():
    smk = sys.argv[1]
    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"{name}: {detail}  {'ok' if good else 'FAIL'}", flush=True)

    run = os.path.join(smk, "arm", "exp_ladder_d640")
    snap = os.path.join(smk, "resume_snap")
    seed = int(open(os.path.join(run, "shard_seed.txt")).read())
    urls = list(braceexpand.braceexpand(make_shard_spec(TAR_DIR, PATTERN)))

    # 1. every worker started its entry
    w_ok, k = True, None
    sched = {}
    for r in range(WORLD):
        for p in (0, 1):
            sched[(r, p)], k = schedule(urls, seed, r, p)
        rows = [x.split() for x in lines(os.path.join(snap, f"shards_started_rank{r}.txt"))]
        ents = [u for _, p, u in rows if p == "0"]
        w_ok &= len(rows) == NW and sorted(ents) == sorted(sched[(r, 0)]) and len(set(ents)) == NW
    check("1. every worker reads", w_ok and k == 16, f"seed {seed}, k = {k} copies per shard; each of {WORLD} ranks started "
          f"{NW} distinct pass-0 entries, its whole pass-0 schedule, one per worker")

    # 2. step time against the Full path
    nf, mf, wf = median_step(os.path.join(smk, "bench", "full"), 20, 300)
    na, ma, wa = median_step(os.path.join(smk, "bench", "arm"), 20, 400)
    if nf == WORLD and na == WORLD:
        f, a = statistics.median(mf), statistics.median(ma)
        check("2. step time", a <= 1.10 * f, f"arm {a:.1f} ms (ranks {min(ma):.1f}-{max(ma):.1f}, data wait {statistics.mean(wa):.2f} ms) "
              f"vs Full path {f:.1f} ms (ranks {min(mf):.1f}-{max(mf):.1f}, data wait {statistics.mean(wf):.2f} ms): {100 * (a / f - 1):+.1f}%")
    else:
        check("2. step time", False, f"step files: Full {nf}, arm {na}")

    # 3. the resume
    info = [re.search(r"shard seed (\d+), shard log kept up to ([\d.]+) \(checkpoint step (\S+)\): (\d+) lines", x)
            for x in lines(os.path.join(run, "launches.txt"))]
    info = [m for m in info if m]
    good = len(info) == 2 and len({m.group(1) for m in info}) == 1 and info[1].group(3) == str(RESUME_STEP)
    check("3a. one seed, clock from the checkpoint", good, f"{len(info)} launches; launch 2 cut at the step-"
          f"{info[1].group(3) if len(info) == 2 else '?'} checkpoint's dump time")
    if not good:
        sys.exit(1)
    keep_t = float(info[1].group(2))
    log_b = "\n".join(lines(os.path.join(smk, "resume_b.log")))
    printed = {int(r): (int(p), int(n)) for r, p, n in re.findall(r"\[shards\] rank (\d+): resume in pass (\d+), (\d+) of its entries", log_b)}
    prefix_ok = skip_ok = next_ok = True
    n_b = 0
    for r in range(WORLD):
        a = lines(os.path.join(snap, f"shards_started_rank{r}.txt"))
        f = lines(os.path.join(run, f"shards_started_rank{r}.txt"))
        kept = [x for x in a if float(x.split()[0]) <= keep_t]
        prefix_ok &= f[:len(kept)] == kept and len(kept) == NW   # all 32 started before the checkpoint
        rows_b = [x.split() for x in f[len(kept):]]
        n_b += len(rows_b)
        skip_ok &= not any(p == "0" for _, p, _ in rows_b)
        b1 = [u for _, p, u in rows_b if p == "1"]
        next_ok &= sorted(b1) == sorted(sched[(r, 1)])
    check("3b. kept lines", prefix_ok, f"launch 2 kept the {NW} pass-0 lines of each rank, all started before the checkpoint")
    check("3c. read entries skipped", skip_ok and printed == {r: (1, 0) for r in range(WORLD)},
          f"launch 2 started no pass-0 entry; ranks printed (pass, skipped) {printed}")
    check("3d. pass 1 whole", next_ok, f"launch 2 started {n_b} entries: each rank's {NW} pass-1 entries, each once")

    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
