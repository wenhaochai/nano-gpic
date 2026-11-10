#!/usr/bin/env python3
"""Checker of jobs/smoke_launch.sh: the launch levers of jobs/train_ladder.sh (gate beside the start, vector-set memo,
thread count from the start) reach the first step from the same state as the launch without them, sooner, and train
no slower afterwards.

The manifest has one line per launch (arm):
  arm name=<arm> group=<group> lever=off|on memo=off|miss|hit step=<start step> log=<launch log> run=<run directory>
      n=<number of this launch in the run directory> bench=<BENCH_OUT> rc=<exit status> wall=<seconds>
      [ranks=0|1] [role=base|test] [gain=<seconds>] [ema_copy=keep|skip]
lever: the first three levers (gate beside the start, memo, thread count); ranks: lever B (the script starts every rank);
ema_copy: lever G1 (skip: configure_model leaves out the second copy of the EMA slot).
A group is one setting (width, fresh or resumed) with at least two base arms and one test arm; without role, an arm
with lever=off is a base arm and one with lever=on a test arm. Read from each launch:
the "[launch-fp]" lines of src/callbacks/launch_fingerprint.py (8 ranks), the "[gate]" and "[warmup]" lines,
the step timer's files, the gate's report and the "launch levers" line of launches.txt.

Checks, FAIL on any:
  every arm      exit 0, one start line and two batch lines per rank, the start step, a healthy gate
  hard fields    step, weights and buffers of the denoiser and of the EMA slot, optimizer and scheduler state, vector
                 sets: equal per rank between the base arms (else the launcher itself does not repeat and nothing can be
                 shown) and between each on arm and the base arms
  soft fields    the four random streams, the first two batches and the first two losses: where the base arms agree
                 on every rank, each on arm must agree too; where they differ the field is listed as not testable.
                 A loss agrees within 0.1% (two launches from one state may sum on the GPU in another order); a loss
                 that is not equal to the last digit is a note
  threads        with the thread lever every rank has the launcher's count and OMP_NUM_THREADS before fit and at the
                 first step. Without it Lightning sets both at fit, but only in a rank whose strategy has a launcher:
                 under FSDP2 (ModelParallelStrategy) ranks 1-7 have none and train with one thread per CPU (w2486), so
                 the lever also changes their thread count in training there: a note, with the counts
  levers         the launches.txt line; on arms print the gate's verdict line once; before fit (when the fingerprint's
                 module is imported) every rank of an on arm has the launcher's thread count, and no rank of an off
                 arm has OMP_NUM_THREADS; memo counts (off: none; miss: no error, something stored; hit: no error,
                 nothing new stored); with ranks=1 the 8 rank processes started within 3 s of each other, without it
                 ranks 1-7 started later than that after rank 0; with ema_copy=skip no rank made the EMA slot's
                 second copy, without it every rank did
  gate           an on arm's gate ran beside main.py's start: its slowest GPU is within 7% of its median, and its
                 median within 3% of the base arms' medians
  speed          a hit arm reaches the first step at least 5 s (or the manifest's gain) before every base arm of its
                 group, a miss arm not after the slowest base arm; an on arm's median step time (steps from the 17th) is at most 2% over the
                 slowest off arm's
The report also gives the seconds and the minor page faults of the EMA slot's second copy (no lever yet).
usage: check_launch.py <manifest>   (exit 0 = PASS)"""
import glob
import json
import os
import re
import statistics
import sys

HARD = ("step", "den", "den_buf", "ema", "ema_buf", "opt", "sched", "isa")
SOFT = ("rng_cpu", "rng_cuda", "rng_np", "rng_py", "batch0", "batch1", "loss0", "loss1")
RANKS = range(8)
SKIP = 16          # steps left out of the step time: compile leftovers and the allocator's first growth
GAIN_MIN_S = 5.0   # a hit arm must reach the first step this much sooner than every off arm
STEP_TOL = 1.02
GATE_MIN = 0.93
GATE_MED = 0.97
LOSS_TOL = 1e-3    # relative; other random draws move a rank's loss by percents, another order of a GPU sum by 1e-6
TOGETHER_S = 3.0   # lever B: the script starts the 8 ranks in one loop; without it rank 0 imports and builds first (6 s and more)


def is_base(a):
    return a.get("role", "base" if a["lever"] == "off" else "test") == "base"


def read_arm(a):
    text = open(a["log"], errors="replace").read()
    a["start"], a["dup"], a["batch"] = {}, 0, {}
    for m in re.finditer(r"\[launch-fp\] rank (\d+) start ([^\n]*)", text):   # anywhere in a line: 8 ranks share the log
        r = int(m.group(1))
        a["dup"] += r in a["start"]
        a["start"][r] = dict(kv.split("=", 1) for kv in m.group(2).split() if "=" in kv)
    for m in re.finditer(r"\[launch-fp\] rank (\d+) batch (\d+) step=(\d+) hash=(\S+) t=([\d.]+)", text):
        key = (int(m.group(1)), int(m.group(2)))
        a["dup"] += key in a["batch"]
        a["batch"][key] = (int(m.group(3)), m.group(4), float(m.group(5)))
    a["gate_wait"] = [float(x) for x in re.findall(r"\[gate\] healthy; fit waited ([\d.]+) s", text)]
    a["warmup"] = [float(s) for s in re.findall(r"\[warmup\] T=\d+: ([\d.]+) s", text)]
    a["main_rc"] = re.findall(r"main\.py exit (\d+)", text)
    n = int(a["n"])
    gates = sorted(glob.glob(os.path.join(a["run"], "gpu_health_*.txt")))
    gate = open(gates[n - 1]).read() if len(gates) >= n else ""
    a["gate_tf"] = [float(x) for x in re.findall(r"GPU \S+: (\d+) TFLOP/s", gate)]
    a["gate_ok"] = bool(re.search(r"median \d+ TFLOP/s; healthy", gate))
    try:
        lines = [l for l in open(os.path.join(a["run"], "launches.txt")) if " launch levers: " in l]
        a["levers"] = dict(kv.split("=", 1) for kv in lines[n - 1].split(" launch levers: ")[1].split())
    except (OSError, IndexError):
        a["levers"] = {}
    a["step_ms"], a["loss"], a["wait_ms"], a["rows"] = {}, {}, {}, {}
    for r in RANKS:
        try:
            rows = json.load(open(os.path.join(a["bench"], f"steps_rank{r}.json")))["rows"]
        except (OSError, ValueError, KeyError):
            continue
        ms = [x["step_ms"] for x in rows[SKIP:] if "step_ms" in x]
        a["rows"][r] = len(rows)
        a["step_ms"][r] = statistics.median(ms) if ms else None
        a["wait_ms"][r] = 1e3 * statistics.mean(x["data_wait_s"] for x in rows[SKIP:]) if len(rows) > SKIP else None
        a["loss"][r] = [repr(x["loss"]) if "loss" in x else None for x in rows[:2]]


def field(a, f, r):
    if f in ("batch0", "batch1"):
        b = a["batch"].get((r, int(f[-1])))
        return None if b is None else f"{b[0]}:{b[1]}"
    if f in ("loss0", "loss1"):
        l = a["loss"].get(r) or []
        return l[int(f[-1])] if len(l) > int(f[-1]) else None
    if f == "thr":   # the intra-op thread count; the inter-op count is in the report only
        return (a["start"].get(r, {}).get("threads") or "").split("/")[0] or None
    return a["start"].get(r, {}).get(f)


def same(f, x, y):
    if x is None or y is None:
        return False
    if x != y and f in ("loss0", "loss1"):
        try:
            x, y = float(x), float(y)
            return abs(x - y) <= LOSS_TOL * max(abs(x), abs(y))
        except ValueError:
            return False
    return x == y


def t(a, r, k):
    try:
        return float(a["start"][r]["t_" + k])
    except (KeyError, ValueError):
        return float("nan")


def copy(a, k, scale=1.0):   # the EMA slot's copy: rank 0 / the largest rank
    v = {}
    for r in a["start"]:
        try:
            v[r] = float(a["start"][r][k]) / scale
        except (KeyError, ValueError):
            pass
    return f"{v[0]:.1f}/{max(v.values()):.1f}" if 0 in v else "?"


def first_step(a):   # seconds from the launcher's start to the first training batch, the last rank
    ts = [a["batch"][(r, 0)][2] for r in RANKS if (r, 0) in a["batch"]]
    return max(ts) if len(ts) == len(RANKS) else float("nan")


def step_ms(a):
    v = [x for x in a["step_ms"].values() if x is not None]
    return statistics.median(v) if v else float("nan")


def memo(a, r):
    m = re.fullmatch(r"h(\d+)/m(\d+)/s(\d+)/e(\d+)", a["start"].get(r, {}).get("memo", ""))
    return tuple(int(x) for x in m.groups()) if m else None


def whole(a):   # every rank's lines are there, once
    return sorted(a["start"]) == list(RANKS) and len(a["batch"]) == 2 * len(RANKS) and not a["dup"]


def ranks(rs):
    rs = sorted(rs)
    return "all ranks" if rs == list(RANKS) else "rank " + ",".join(map(str, rs))


def main(manifest):
    arms = []
    for line in open(manifest):
        p = line.split()
        if p and p[0] == "arm":
            a = dict(kv.split("=", 1) for kv in p[1:])
            read_arm(a)
            arms.append(a)
    fails, notes = [], []
    groups = {}
    for a in arms:
        groups.setdefault(a["group"], []).append(a)
    if not arms:
        fails.append("no arm in the manifest")

    for a in arms:   # every arm by itself
        nm, on = a["name"], a["lever"] == "on"
        if a["rc"] != "0" or a["main_rc"] != ["0"]:
            fails.append(f"{nm}: exit status {a['rc']}, main.py {a['main_rc']}")
        if not whole(a):
            fails.append(f"{nm}: {len(a['start'])} start and {len(a['batch'])} batch lines, {a['dup']} repeated (need 8 and 16)")
        bad = [r for r in a["start"] if a["start"][r].get("step") != a["step"]]
        if bad:
            fails.append(f"{nm}: starts at step {a['start'][bad[0]].get('step')}, not {a['step']} ({ranks(bad)})")
        bad = [k for k, b in a["batch"].items() if b[0] != int(a["step"]) + k[1]]
        if bad:
            fails.append(f"{nm}: the first two batches are not steps {a['step']} and {int(a['step']) + 1}")
        if not a["gate_ok"] or len(a["gate_tf"]) != len(RANKS):
            fails.append(f"{nm}: no healthy gate report of 8 GPUs")
        if len(a["step_ms"]) != len(RANKS) or any(v is None for v in a["step_ms"].values()):
            fails.append(f"{nm}: the step timer's files are missing or too short")
        want = {"gate_bg": "1"} if on else {"gate_bg": "0", "vec_isa_memo": "off", "omp": "off"}
        lv = a["levers"]
        if any(lv.get(k) != v for k, v in want.items()) or (on and (lv.get("vec_isa_memo") in (None, "off") or
                                                                     not lv.get("omp", "").startswith("OMP_NUM_THREADS="))):
            fails.append(f"{nm}: launches.txt has the levers {lv}")
        if lv.get("ranks", "0") != a.get("ranks", "0"):
            fails.append(f"{nm}: launches.txt has ranks={lv.get('ranks', '0')}, the manifest {a.get('ranks', '0')}")
        tp = [t(a, r, "proc") for r in a["start"]]
        if len(tp) == len(RANKS) and (max(tp) - min(tp) <= TOGETHER_S) != (a.get("ranks", "0") == "1"):
            fails.append(f"{nm}: the 8 rank processes started {max(tp) - min(tp):.1f} s apart with ranks={a.get('ranks', '0')}")
        skip = a.get("ema_copy", "keep") == "skip"
        if lv.get("skip_ema_copy", "0") != ("1" if skip else "0"):
            fails.append(f"{nm}: launches.txt has skip_ema_copy={lv.get('skip_ema_copy', '0')}, the manifest ema_copy={a.get('ema_copy', 'keep')}")
        bad = [r for r, s in a["start"].items() if (s.get("copy_s") == "none") != skip]
        if bad:
            fails.append(f"{nm}: the EMA slot's second copy on {ranks(bad)}: copy_s={a['start'][bad[0]].get('copy_s')} with ema_copy={a.get('ema_copy', 'keep')}")
        if len(a["gate_wait"]) != (1 if on else 0):
            fails.append(f"{nm}: {len(a['gate_wait'])} gate verdict lines for lever {a['lever']}")
        n_thr = lv.get("omp", "").rpartition("=")[2]   # the launcher's count; without one the line above has failed
        if on and n_thr.isdigit():
            bad = [r for r, s in a["start"].items() if s.get("thr0") != n_thr or s.get("omp0") != n_thr]
            if bad:
                fails.append(f"{nm}: not the launcher's {n_thr} threads before fit on {ranks(bad)}: "
                             f"thr0={a['start'][bad[0]].get('thr0')} omp0={a['start'][bad[0]].get('omp0')}")
        if on and n_thr.isdigit():
            bad = [r for r, s in a["start"].items() if field(a, "thr", r) != n_thr or s.get("omp") != n_thr]
            if bad:
                fails.append(f"{nm}: not the launcher's {n_thr} threads at the first step on {ranks(bad)}: "
                             f"threads={a['start'][bad[0]].get('threads')} omp={a['start'][bad[0]].get('omp')}")
        if not on:
            bad = [r for r, s in a["start"].items() if s.get("omp0") != "unset"]
            if bad:
                fails.append(f"{nm}: OMP_NUM_THREADS={a['start'][bad[0]].get('omp0')} before fit without the lever ({ranks(bad)})")
        ms = {r: memo(a, r) for r in a["start"]}
        if None in ms.values():
            fails.append(f"{nm}: no memo counts")
        elif a["memo"] == "off" and any(m != (0, 0, 0, 0) for m in ms.values()):
            fails.append(f"{nm}: memo counts without the memo: {sorted(set(ms.values()))}")
        elif a["memo"] == "miss" and (any(m[3] or m[2] > m[1] or m[0] + m[1] < 1 for m in ms.values())
                                      or sum(m[2] for m in ms.values()) < 1):
            fails.append(f"{nm}: memo counts of the first launch with the memo: {sorted(set(ms.values()))}")
        elif a["memo"] == "hit" and any(m[2] or m[3] or m[0] < 1 for m in ms.values()):   # a new store = a key that moves
            fails.append(f"{nm}: memo counts of a launch after the first: {sorted(set(ms.values()))}")
        elif a["memo"] == "hit" and any(m[1] for m in ms.values()):   # a False answer is never stored
            notes.append(f"{nm}: {max(m[1] for m in ms.values())} vector-set checks without a stored answer are asked at every launch")
        if a["gate_tf"] and on and min(a["gate_tf"]) < GATE_MIN * statistics.median(a["gate_tf"]):
            fails.append(f"{nm}: the gate's slowest GPU is {min(a['gate_tf']):.0f} of a median {statistics.median(a['gate_tf']):.0f} TFLOP/s")

    for g, ga in groups.items():   # on against off
        offs, ons = [a for a in ga if is_base(a)], [a for a in ga if not is_base(a)]
        if len(offs) < 2 or not ons:
            fails.append(f"{g}: needs two base arms and one test arm")
            continue
        offs, ons = [a for a in offs if whole(a)], [a for a in ons if whole(a)]   # the others have failed above
        if len(offs) < 2:
            continue
        for f in HARD + SOFT:
            vals = {r: [field(a, f, r) for a in offs] for r in RANKS}
            differ = [r for r in RANKS if not all(same(f, v, vals[r][0]) for v in vals[r])]
            if differ:
                if f in HARD:
                    fails.append(f"{g}: the base arms differ in {f} ({ranks(differ)}): the launcher itself does not repeat")
                else:
                    notes.append(f"{g}: {f} differs between the base arms ({ranks(differ)}): not testable")
                continue
            for a in ons:
                bad = [r for r in RANKS if not all(same(f, field(a, f, r), v) for v in vals[r])]
                if bad:
                    fails.append(f"{a['name']}: {f} differs from the base arms ({ranks(bad)}): "
                                 f"{field(a, f, bad[0])} against {vals[bad[0]][0]}")
                    continue
                near = [r for r in RANKS if any(field(a, f, r) != v for v in vals[r])]
                if near:
                    notes.append(f"{a['name']}: {f} agrees with the base arms within {LOSS_TOL:g} but not to the last digit ({ranks(near)}): "
                                 f"{field(a, f, near[0])} against {vals[near[0]][0]}")
        n_on = {x["levers"].get("omp", "").rpartition("=")[2] for x in ga if x["lever"] == "on"}
        if len(n_on) == 1 and next(iter(n_on)).isdigit():   # what the thread lever changes in training (see "threads" above)
            n_thr = next(iter(n_on))
            for a in ga:
                bad = [r for r, s in a["start"].items() if a["lever"] == "off" and (field(a, "thr", r) != n_thr or s.get("omp") != n_thr)]
                if bad:
                    s = a["start"][bad[0]]
                    notes.append(f"{g}: without the thread lever {ranks(bad)} train with {field(a, 'thr', bad[0])} threads and OMP_NUM_THREADS "
                                 f"{s.get('omp')} ({a['name']}); with it every rank has {n_thr}")
                    break
        t_off, s_off = [first_step(a) for a in offs], [step_ms(a) for a in offs]
        g_off = [statistics.median(a["gate_tf"]) for a in offs if a["gate_tf"]]
        for a in ons:
            tf, sm = first_step(a), step_ms(a)
            if a["memo"] == "hit" and not tf <= min(t_off) - float(a.get("gain", GAIN_MIN_S)):
                fails.append(f"{a['name']}: first step after {tf:.1f} s, the base arms after {min(t_off):.1f}-{max(t_off):.1f} s")
            if a["memo"] != "hit" and not tf <= max(t_off):
                fails.append(f"{a['name']}: first step after {tf:.1f} s, later than the base arms ({max(t_off):.1f} s)")
            if not sm <= STEP_TOL * max(s_off):
                fails.append(f"{a['name']}: step {sm:.2f} ms, the base arms {min(s_off):.2f}-{max(s_off):.2f} ms")
            if a["gate_tf"] and g_off and statistics.median(a["gate_tf"]) < GATE_MED * statistics.median(g_off):
                fails.append(f"{a['name']}: gate median {statistics.median(a['gate_tf']):.0f} TFLOP/s, the base arms {statistics.median(g_off):.0f}")

    # the report: seconds from the start of train_ladder.sh, rank 0 unless said
    print(f"{'arm':<13}{'lever':<6}{'memo':<5}{'proc':>6}{'init':>7}{'setup':>7}{'fit':>7}{'train':>7}{'first':>7}{'last rank':>10}"
          f"{'warm-up':>8}{'r1-7 build':>11}{'gate wait':>10}{'step ms':>9}{'data ms':>8}{'gate TFLOP/s':>14}{'copy s':>10}{'copy kflt':>12}"
          f"  threads before fit -> at the first step (rank 0; ranks 1-7), CPUs")
    for a in arms:
        build = [t(a, r, "init") - t(a, r, "proc") for r in RANKS if r and r in a["start"]]
        b0 = a["batch"].get((0, 0), (0, "", float("nan")))[2]
        thr = [f"{a['start'].get(r, {}).get('thr0', '?')} -> {a['start'].get(r, {}).get('threads', '?')}" for r in RANKS]
        cpus = sorted({a["start"].get(r, {}).get("cpus", "?") for r in RANKS})
        gt = f"{min(a['gate_tf']):.0f}/{statistics.median(a['gate_tf']):.0f}" if a["gate_tf"] else "?"
        wait = [x for x in a["wait_ms"].values() if x is not None]
        print(f"{a['name']:<13}{a['lever']:<6}{a['memo']:<5}{t(a, 0, 'proc'):>6.1f}{t(a, 0, 'init'):>7.1f}{t(a, 0, 'setup'):>7.1f}"
              f"{t(a, 0, 'fit'):>7.1f}{t(a, 0, 'train'):>7.1f}{b0:>7.1f}{first_step(a):>10.1f}{sum(a['warmup']):>8.1f}"
              f"{(statistics.mean(build) if build else float('nan')):>11.1f}"
              f"{(a['gate_wait'][0] if a['gate_wait'] else float('nan')):>10.1f}{step_ms(a):>9.2f}"
              f"{(statistics.mean(wait) if wait else float('nan')):>8.2f}{gt:>14}{copy(a, 'copy_s'):>10}{copy(a, 'copy_flt', 1e3):>12}"
              f"  {thr[0]}; {' | '.join(sorted(set(thr[1:])))}, {'|'.join(cpus)}")
    for g, ga in groups.items():
        offs, ons = [a for a in ga if is_base(a)], [a for a in ga if not is_base(a)]
        if not offs:
            continue
        base, sb = statistics.mean(first_step(a) for a in offs), statistics.mean(step_ms(a) for a in offs)
        for a in ons:
            print(f"gain {a['name']} ({a['memo']}): first step {base - first_step(a):+.1f} s sooner ({base:.1f} -> {first_step(a):.1f} s), "
                  f"step time {100 * (step_ms(a) / sb - 1):+.2f}% ({sb:.2f} -> {step_ms(a):.2f} ms)")
        equal = [f for f in HARD + SOFT if all(len({field(a, f, r) for a in ga}) == 1 and field(ga[0], f, r) is not None for r in RANKS)]
        print(f"equal in all {len(ga)} arms of {g}, every rank: {' '.join(equal) or 'nothing'}")
    for n in notes:
        print("note:", n)
    for f in fails:
        print("FAIL:", f)
    print(f"CHECK_LAUNCH {'FAIL' if fails else 'PASS'}: {len(arms)} arms in {len(groups)} groups, {len(fails)} failures, {len(notes)} notes")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
