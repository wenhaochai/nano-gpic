"""Checks of jobs/smoke_full.sh (docs/ladder_plan.md section 12). Run in baselines/PixelGen with PYTHONPATH=$PWD.
usage: check_shard_resume.py <smoke dir>
1. stop and resume (resume/exp_ladder_d640): both launches used one shard seed; launch 2 cut the log at the dump time stored
   in its checkpoint (step 600), kept exactly the lines of launch 1 up to it and dropped at least one per rank after it, with
   the cut between the last kept and the first dropped line; launch 2 started no shard that the log kept and started again
   every dropped one; over the two launches each rank started each of its pass-0 shards once; every pass-1 line is a shard
   of that rank in the pass-1 permutation, each once; every rank with kept shards printed its resume with that number
2. d1536 and d1792 (bench/<size>/steps_rank*.json): median start-to-start step time after step 20 without the save steps,
   the extra time of a save step, the peak allocated and reserved memory over the 8 ranks (fails above 76 GiB reserved)
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
PATTERN = "gpic_train_{00000..00063}.tar"
WORLD = 8
RESUME_STEP = 600   # jobs/smoke_full.sh: launch 1 saves last.ckpt at step 600 and stops at 1000


def lines(p):
    return [x.rstrip("\n") for x in open(p)] if os.path.exists(p) else []


def main():
    smk = sys.argv[1]
    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"{name}: {detail}  {'ok' if good else 'FAIL'}", flush=True)

    run = os.path.join(smk, "resume", "exp_ladder_d640")
    snap = os.path.join(smk, "resume_snap")
    info = [re.search(r"shard seed (\d+), shard log kept up to ([\d.]+) \(checkpoint step (\S+)\): (\d+) lines", x)
            for x in lines(os.path.join(run, "launches.txt"))]
    info = [m for m in info if m]
    seeds = {m.group(1) for m in info}
    check("1a. one shard seed", len(info) == 2 and len(seeds) == 1, f"{len(info)} launches, seeds {sorted(seeds)}")
    if len(info) != 2 or len(seeds) != 1:
        sys.exit(1)
    seed, keep_t, ck_step = int(info[0].group(1)), float(info[1].group(2)), info[1].group(3)
    check("1g. clock from the checkpoint", ck_step == str(RESUME_STEP),
          f"launch 2 cut the log at the dump time stored in its checkpoint (step {ck_step}), not at the file's mtime")
    urls = list(braceexpand.braceexpand(make_shard_spec(TAR_DIR, PATTERN)))
    perm = {p: EpochShuffledShards(urls, seed)._perm(p) for p in (0, 1, 2)}
    log_b = "\n".join(lines(os.path.join(smk, "resume_b.log")))
    printed = {int(r): int(n) for r, n in re.findall(r"\[shards\] rank (\d+): resume in pass 0, (\d+) of its shards", log_b)}
    prefix_ok = skip_ok = cover_ok = next_ok = print_ok = reread_ok = True
    n_kept = n_dropped = n_next = 0
    t_kept_max, t_drop_min = 0.0, float("inf")
    for r in range(WORLD):
        a = lines(os.path.join(snap, f"shards_started_rank{r}.txt"))
        f = lines(os.path.join(run, f"shards_started_rank{r}.txt"))
        kept = [x for x in a if float(x.split()[0]) <= keep_t]
        dropped = [x for x in a if float(x.split()[0]) > keep_t]
        n_kept += len(kept)
        n_dropped += len(dropped)
        t_kept_max = max([t_kept_max] + [float(x.split()[0]) for x in kept])
        t_drop_min = min([t_drop_min] + [float(x.split()[0]) for x in dropped])
        prefix_ok &= f[:len(kept)] == kept
        # a shard started after the checkpoint was read only for steps that launch 2 trains again: launch 2 reads it again
        reread_ok &= {x.split()[2] for x in dropped} <= {x.split()[2] for x in f[len(kept):]}
        rows_k = [x.split() for x in kept]
        rows_b = [x.split() for x in f[len(kept):]]
        k0 = [u for _, p, u in rows_k if p == "0"]
        b0 = [u for _, p, u in rows_b if p == "0"]
        b1 = [u for _, p, u in rows_b if p == "1"]
        n_next += len(b1)
        skip_ok &= len(k0) == len(rows_k) and not set(k0) & set(b0)
        cover_ok &= sorted(k0 + b0) == sorted(perm[0][r::WORLD])
        next_ok &= set(b1) <= set(perm[1][r::WORLD]) and len(set(b1)) == len(b1) and all(p in ("0", "1") for _, p, _ in rows_b)
        print_ok &= printed.get(r, 0) == len(k0)
    check("1b. kept and dropped lines", prefix_ok and n_kept >= WORLD and n_dropped >= WORLD and t_kept_max <= keep_t < t_drop_min,
          f"launch 2 kept {n_kept} lines up to {keep_t:.0f} and dropped {n_dropped} later ones (the last kept line "
          f"{keep_t - t_kept_max:.0f} s before the cut, the first dropped one {t_drop_min - keep_t:.0f} s after); kept lines are the log's head")
    check("1h. dropped shards read again", reread_ok, "every shard started after the checkpoint is started again by launch 2")
    check("1c. no shard read twice", skip_ok, "launch 2 read none of the kept shards")
    check("1d. pass 0 whole", cover_ok, "kept + launch 2 = each rank's 8 pass-0 shards, each once")
    check("1e. pass 1", next_ok and n_next >= 1, f"{n_next} pass-1 shards started, all in the pass-1 permutation of their rank")
    check("1f. resume print", print_ok, f"ranks printed {printed}")

    for size in ("d1536", "d1792"):
        if not os.path.isdir(os.path.join(smk, "bench")):
            print(f"2. {size}: skipped (a resume-only smoke)", flush=True)
            continue
        fs = sorted(glob.glob(os.path.join(smk, "bench", size, "steps_rank*.json")))
        if len(fs) != WORLD:
            check(f"2. {size}", False, f"{len(fs)} step files")
            continue
        med, save_extra, alloc, resv = [], [], 0.0, 0.0
        for p in fs:
            rows = [x for x in json.load(open(p))["rows"] if x.get("step_ms") is not None]
            near_save = lambda s: any(abs(s - k) <= 1 for k in (100, 200, 300))
            base = [x["step_ms"] for x in rows if x["step"] >= 20 and not near_save(x["step"])]
            m = statistics.median(base)
            med.append(m)
            save_extra.append(max(x["step_ms"] for x in rows if near_save(x["step"])) - m)
            alloc = max(alloc, max(x["max_alloc_gb"] for x in rows))
            resv = max(resv, max(x["max_reserved_gb"] for x in rows))
        check(f"2. {size}", resv <= 76,
              f"median step {statistics.median(med):.1f} ms (ranks {min(med):.1f}-{max(med):.1f}), a save step adds up to "
              f"{max(save_extra) / 1000:.1f} s, peak allocated {alloc:.1f} GiB, reserved {resv:.1f} GiB")

    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
