"""Checks of an 8-GPU ladder smoke (jobs/smoke_ladder.sh) and the step-time table that sets each run's --time.

python tools/check_ladder_smoke.py <smoke_dir> <docs/ladder_runs.json>
Exit 1 when a check fails. Checks:
1. every size: train log ends with 'main.py exit 0'; last.ckpt at global step 320 with optimizer, LR-scheduler and
   callback state; ms-step=000080/160/240/320 with the denoiser and the EMA denoiser and no optimizer state;
2. resume: d640's second launch says 'resuming from', and its first timed step is step 160;
3. epoch ends: the 2400-step run finished; for every epoch pass, all 8 ranks logged the same leading shards, and the
   passes differ from each other;
4. eval: metrics.json with an FD.
Table: median start-to-start step time over the last 100 timed steps (rank 0), peak allocated memory (max over
ranks), compiled-graph count, and the production hours max_steps x step time.
"""
import glob
import json
import os
import re
import statistics
import sys

import torch

SIZES = ["d640", "d768", "d896", "d1024", "d1280"]


def ckpt(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def main():
    smk, runs_json = sys.argv[1], sys.argv[2]
    runs = {r["size"]: r for r in json.load(open(runs_json))["runs"]}
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("ok   " if cond else "FAIL ") + msg, flush=True)

    rows = []
    for size in SIZES:
        tags = ["d640_a", "d640_b"] if size == "d640" else [size]
        for t in tags:
            log = open(os.path.join(smk, f"{t}.log")).read()
            check("main.py exit 0" in log, f"{t}: main.py exit 0")
        run = os.path.join(smk, f"exp_ladder_{size}")
        last = ckpt(os.path.join(run, "last.ckpt"))
        check(last["global_step"] == 320 and last.get("optimizer_states") and last.get("lr_schedulers")
              and last.get("callbacks"), f"{size}: last.ckpt step {last['global_step']} with optimizer/scheduler/callback state")
        del last
        for s in (80, 160, 240, 320):
            f = glob.glob(os.path.join(run, f"ms-*{s:06d}*.ckpt"))
            if not f:
                check(False, f"{size}: ms checkpoint for step {s}")
                continue
            c = ckpt(f[0])
            keys = c["state_dict"].keys()
            has = any(k.startswith("denoiser.") for k in keys) and any(k.startswith("ema_denoiser.") for k in keys)
            check(has and not c.get("optimizer_states") and c["global_step"] == s,
                  f"{size}: {os.path.basename(f[0])} model+EMA, no optimizer state, step {c['global_step']}")
            del c
        bench = json.load(open(os.path.join(smk, "bench", tags[-1], "steps_rank0.json")))
        steps = [r["step_ms"] for r in bench["rows"] if "step_ms" in r]
        med = statistics.median(steps[-100:])
        peak = max(json.load(open(p))["rows"][-1]["max_alloc_gb"]
                   for p in glob.glob(os.path.join(smk, "bench", tags[-1], "steps_rank*.json")))
        graphs = bench["compile"].get("counters", {}).get("stats", {}).get("unique_graphs")
        log = open(os.path.join(smk, f"{tags[-1]}.log")).read()
        warm = re.findall(r"\[warmup\] T=\d+: ([\d.]+) s", log)
        late = log.split("[warmup] done")[-1].count("Recompiling function")
        print(f"     {size}: warm-up {warm} s; recompilations after the warm-up (all ranks): {late}", flush=True)
        hours = runs[size]["max_steps"] * med / 3.6e6
        rows.append((size, med, 256 / med * 1e3, peak, graphs, runs[size]["max_steps"], hours))
        if size == "d640":
            b = open(os.path.join(smk, "d640_b.log")).read()
            first = bench["rows"][0]["step"]
            check("resuming from" in b and first == 160, f"d640: second launch resumed (first timed step {first})")

    # epoch ends
    elog = open(os.path.join(smk, "epoch.log")).read()
    check("main.py exit 0" in elog, "epoch run: main.py exit 0")
    passes = {}
    for rank, p, shards in re.findall(r"\[shards\] rank (\d+) pass (\d+): (.+)", elog):
        passes.setdefault(int(p), {})[int(rank)] = shards.strip()
    full = {p: v for p, v in passes.items() if len(v) == 8}
    same = all(len(set(v.values())) == 1 for v in full.values())
    distinct = len({next(iter(v.values())) for v in full.values()}) == len(full)
    check(len(full) >= 3 and same and distinct,
          f"epoch run: {len(full)} passes logged by all 8 ranks; same shards on every rank {same}; passes differ {distinct}")

    # eval
    m = os.path.join(smk, "exp_ladder_d640", "eval_000320", "metrics.json")
    check(os.path.exists(m) and "fd" in open(m).read().lower(), f"eval: {m}")

    print("\nsize   step_ms  img/s   peak_GB  graphs  max_steps  hours(8 GPUs)")
    for size, med, ips, peak, graphs, steps, hours in rows:
        print(f"{size:6s} {med:7.1f} {ips:7.0f} {peak:8.1f}  {graphs!s:6s}  {steps:9d}  {hours:6.2f}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
