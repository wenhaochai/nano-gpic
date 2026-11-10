"""Test, without a GPU, of launch lever B of jobs/train_ladder.sh (GPIC_LAUNCH_RANKS=1: the script starts every rank at
once, gpic-28 2026-11-09, logs/mfu_infra_scan_1109b.yaml item 6): the real launcher with a made-up worktree, a stub
tools/gpu_health.py and a stub main.py that every rank runs (it records its environment, runs the real before_fit of
baselines/PixelGen/main.py and then does what the case asks of its rank).
Checked: each rank gets its LOCAL_RANK and the variables Lightning's own launcher gives the ranks it starts; all ranks
start together with one command line; the launch's status (all 0; the first rank that fails, and KILL for the others;
a rank that fails or hangs after rank 0 ended with 0 does not change it); the script waits for every rank and leaves no
process; the gate beside the start (every rank waits for the verdict, one line; a failed gate: 98, no fit); a cut (TERM
to the process group) with ranks that save and stop, and with ranks that die of it; the switch off; the device count.
usage: python tools/mfu/tests/test_launch_ranks.py <new scratch directory>     (a vis node; about 2 min)
Result: TEST_LAUNCH_RANKS PASS: N cases."""
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
REAL_MAIN = f"{WT}/baselines/PixelGen/main.py"
LAUNCHER = os.environ.get("TEST_LAUNCHER") or f"{WT}/jobs/train_ladder.sh"   # TEST_LAUNCHER: a changed copy, to see this test fail
OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT)   # a new directory: nothing of an earlier run is read
results = []


def check(name, problems):
    results.append((name, problems))
    print(("ok   " if not problems else "FAIL ") + name + "".join(f"\n       {p}" for p in problems), flush=True)


STUB_GATE = '''# stub of tools/gpu_health.py (tools/mfu/tests/test_launch_ranks.py): STUB_GATE = pass | fail [:seconds]
import os, sys, time
mode, secs = (os.environ.get("STUB_GATE", "pass:1") + ":1").split(":")[:2]
time.sleep(float(secs))
print(f"stubhost: median 100 TFLOP/s; {'UNHEALTHY GPUs 0' if mode == 'fail' else 'healthy'}", flush=True)
with open(os.environ["STUB_GATE_LOG"], "a") as f:
    f.write(f"end {time.time():.3f}\\n")
sys.exit(1 if mode == "fail" else 0)
'''
STUB_MAIN = '''# stub of baselines/PixelGen/main.py (tools/mfu/tests/test_launch_ranks.py). STUB_RANKS = <rank or *>=<what>,...
#   fit:<s>          before_fit, then <s> seconds of training, exit 0 (TERM ends it: 143)
#   die:<status>:<s> exit <status> after <s> seconds, before fit
#   term0:<s>:<x>    as fit:<s>, but a TERM is caught: exit 0 after <x> more seconds (a rank that saves and stops)
#   hang:<s>         as fit:<s>, then it ignores TERM and never ends (a rank that hangs at its exit)
import ast, json, os, signal, sys, textwrap, time
def real_before_fit(path):   # the method's own source, from the real main.py
    src = open(path).read()
    fn = next(n for c in ast.parse(src).body if isinstance(c, ast.ClassDef) and c.name == "ReWriteRootDirCli"
              for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "before_fit")
    ns = {"os": os, "sys": sys, "time": time}
    exec(textwrap.dedent(ast.get_source_segment(src, fn, padded=True)), ns)
    return ns["before_fit"]
r = os.environ.get("LOCAL_RANK", "x")
rec = {"t_start": time.time(), "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv[1:],
       "env": {k: os.environ.get(k) for k in ("LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "NODE_RANK", "WORLD_SIZE",
                                              "OMP_NUM_THREADS", "GPIC_GATE_" + "RC")}}
def dump():
    p = os.environ["STUB_OUT"] + "." + r
    with open(p + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(p + ".tmp", p)
plan = dict(x.split("=", 1) for x in os.environ.get("STUB_RANKS", "*=fit:0.5").split(","))
mode = plan.get(r, plan.get("*", "fit:0.5")).split(":")
got = []
if mode[0] == "term0":
    signal.signal(signal.SIGTERM, lambda *a: got.append(1))
dump()
if mode[0] == "die":
    time.sleep(float(mode[2]))
    sys.exit(int(mode[1]))
real_before_fit(os.environ["STUB_REAL_MAIN"])(None)
rec["t_fit"] = time.time()
dump()
t = time.time()
while not got and time.time() - t < float(mode[1]):
    time.sleep(0.05)
if got:
    time.sleep(float(mode[2]))
if mode[0] == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    rec["t_hang"] = time.time()
    dump()
    time.sleep(600)
'''
assert "GPIC_GATE_RC" not in STUB_MAIN   # the launcher looks for this name in main.py: the marker line below adds it
MEMO = "/scratch/gpfs/GROUP/USER/cache/inductor/vec_isa_memo"   # the launcher's default
FW, FW_OLD = f"{OUT}/worktree", f"{OUT}/worktree_old"   # FW_OLD: a model file that does not know lever G1's switch
for fw in (FW, FW_OLD):
    os.makedirs(f"{fw}/baselines/PixelGen/src")
    os.makedirs(f"{fw}/tools")
    with open(f"{fw}/tools/gpu_health.py", "w") as f:
        f.write(STUB_GATE)
    with open(f"{fw}/baselines/PixelGen/main.py", "w") as f:
        f.write(STUB_MAIN + "# this main.py knows GPIC_GATE_RC\n")
    with open(f"{fw}/baselines/PixelGen/src/lightning_model_gpic.py", "w") as f:
        f.write("# stub\n" + ("# knows GPIC_LAUNCH_SKIP_EMA_COPY\n" if fw == FW else ""))
count = [0]


def session(sid, wait=3.0):   # processes of the launcher's session, after up to <wait> s
    t = time.time()
    while True:
        pids = subprocess.run(["pgrep", "-s", str(sid)], capture_output=True, text=True).stdout.split()
        if not pids or time.time() - t > wait:
            return pids
        time.sleep(0.2)


def launch(name, ranks="*=fit:1", gate="pass:0.3", env=None, args=(), cut=None, partial_ok=True, wt=None):
    count[0] += 1
    d, tok = f"{OUT}/{name}", f"rktok{os.getpid()}n{count[0]}"
    os.makedirs(f"{d}/tmp")
    os.makedirs(f"{d}/bin")
    with open(f"{d}/bin/nproc", "w") as f:   # the CPU count the launcher sees
        f.write("#!/bin/sh\necho 96\n")
    os.chmod(f"{d}/bin/nproc", 0o755)
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("GPIC_", "LADDER_", "SLURM_", "STUB_", "OMP_", "PROBE_", "MASTER_"))
         and k not in ("FETCHED_JOB_ID", "BENCH_OUT", "LOCAL_RANK", "NODE_RANK", "WORLD_SIZE")}
    e.update(PATH=f"{d}/bin:{os.environ['PATH']}", TMPDIR=f"{d}/tmp", LADDER_WT=wt or FW, LADDER_RUN_ROOT=f"{d}/root",
             STUB_GATE=gate, STUB_RANKS=ranks, STUB_OUT=f"{d}/main.json", STUB_GATE_LOG=f"{d}/gate.log", STUB_REAL_MAIN=REAL_MAIN,
             GPIC_LAUNCH_RANKS="1")
    if partial_ok:
        e["LADDER_PARTIAL_OK"] = "1"
    e.update(env or {})
    t0 = time.time()
    p = subprocess.Popen(["bash", LAUNCHER, "d640", "640", "9", "2", "4096", "--tok", tok, *args], env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    if cut:   # the pilot's cut: TERM to the whole process group, once 8 ranks started ("started") or reached fit ("fit")
        def started():
            n = 0
            for r in range(8):
                try:
                    with open(f"{d}/main.json.{r}") as f:
                        n += cut == "started" or "t_fit" in json.load(f)
                except (FileNotFoundError, ValueError):
                    pass
            return n
        while started() < 8 and p.poll() is None and time.time() - t0 < 60:
            time.sleep(0.05)
        time.sleep(0.5)
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:   # the launcher ended before the cut: the case's own checks say so
            pass
    try:
        out = p.communicate(timeout=120)[0]
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out = p.communicate()[0] + "\n[test] the launcher did not end in 120 s"
    r = {"rc": p.returncode, "out": out, "wall": time.time() - t0, "ranks": {}, "levers": None, "gate_end": None}
    for f in os.listdir(d):
        m = re.fullmatch(r"main\.json\.(\w+)", f)
        if m:
            with open(f"{d}/{f}") as g:
                r["ranks"][m.group(1)] = json.load(g)
    lt = f"{d}/root/exp_ladder_d640/launches.txt"
    if os.path.exists(lt):
        with open(lt) as f:
            r["levers"] = next((l.split("launch levers: ")[1].strip() for l in f if "launch levers: " in l), None)
    if os.path.exists(f"{d}/gate.log"):
        with open(f"{d}/gate.log") as f:
            r["gate_end"] = next((float(l.split()[1]) for l in f if l.startswith("end")), None)
    r["left"] = session(p.pid)
    if r["left"]:   # what they are, then the test cleans up after the failed case
        r["left"] = subprocess.run(["ps", "-o", "pid=,ppid=,etimes=,args=", "-p", ",".join(r["left"])], capture_output=True, text=True).stdout.strip().splitlines() or r["left"]
    for q in [x.split()[0] for x in r["left"]]:
        try:
            os.kill(int(q), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    r["tmp"] = sorted(os.listdir(f"{d}/tmp"))
    return r


def expect(r, rc, n=8, levers=None, text=(), no_text=(), wall=None, omp="12", fit=None):
    """n: the ranks that started (None: the one process of the switch-off launch); fit: how many reached fit."""
    pr = []
    if r["rc"] != rc:
        pr.append(f"exit status {r['rc']}, expected {rc}")
    if levers is not None and r["levers"] != levers:
        pr.append(f"launches.txt: levers {r['levers']!r}, expected {levers!r}")
    pr += [f"no /{t}/ in the output" for t in text if not re.search(t, r["out"])]
    pr += [f"/{t}/ in the output" for t in no_text if re.search(t, r["out"])]
    R = r["ranks"]
    if n is None:
        if sorted(R) != ["x"] or any(R["x"]["env"][k] is not None for k in ("LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "NODE_RANK", "WORLD_SIZE")):
            pr.append(f"the switch-off launch: processes {sorted(R)}, environment {R.get('x', {}).get('env')}")
    else:
        if sorted(R) != [str(i) for i in range(n)]:
            pr.append(f"ranks that started: {sorted(R)}, expected 0..{n - 1}")
        else:
            for i in range(n):
                e = R[str(i)]["env"]
                want = {"LOCAL_RANK": str(i), "MASTER_ADDR": "127.0.0.1", "NODE_RANK": "0", "WORLD_SIZE": str(n), "OMP_NUM_THREADS": omp}
                bad = {k: e[k] for k in want if e[k] != want[k]}
                if bad:
                    pr.append(f"rank {i} saw {bad}, expected {want}")
            ports = {R[k]["env"]["MASTER_PORT"] for k in R}
            if len(ports) != 1 or not (next(iter(ports)) or "").isdigit() or not 10000 <= int(next(iter(ports))) < 30000:
                pr.append(f"MASTER_PORT of the ranks: {ports}")
            if len({json.dumps(R[k]["argv"]) for k in R}) != 1 or len({R[k]["ppid"] for k in R}) != 1:
                pr.append("the ranks do not share one command line and one parent")
            if max(R[k]["t_start"] for k in R) - min(R[k]["t_start"] for k in R) > 3:
                pr.append(f"the ranks started {max(R[k]['t_start'] for k in R) - min(R[k]['t_start'] for k in R):.1f} s apart")
            if R["0"]["pid"] != min(R[k]["pid"] for k in R):
                pr.append("rank 0 is not the oldest process")
    if fit is not None and sum("t_fit" in v for v in R.values()) != fit:
        pr.append(f"{sum('t_fit' in v for v in R.values())} ranks reached fit, expected {fit}")
    if wall and not wall[0] <= r["wall"] <= wall[1]:
        pr.append(f"the launcher took {r['wall']:.1f} s, expected {wall[0]}-{wall[1]} s")
    if any(f.startswith("gpu_gate_") for f in r["tmp"]):
        pr.append(f"left in TMPDIR: {r['tmp']}")
    if r["left"]:
        pr.append(f"processes left in the launcher's session: {r['left']}")
    return pr + (["output: " + r["out"][-900:].replace("\n", " | ")] if pr else [])


ON = f"gate_bg=1 vec_isa_memo={MEMO} omp=OMP_NUM_THREADS=12 ranks=1 skip_ema_copy=0"
r = launch("all_ok")
check("8 ranks, all end with 0", expect(r, 0, levers=ON, text=[r"main\.py exit 0", r"\[gate\] healthy"], no_text=[r"KILL", r"unhealthy"], fit=8, wall=(1, 25)))
r = launch("off", env={"GPIC_LAUNCH_RANKS": "0"})
check("the switch off: one process, none of the rank variables", expect(r, 0, n=None, levers=ON.replace("ranks=1", "ranks=0"), text=[r"main\.py exit 0"]))
for name, env, wt, want in (("g1_on", {"GPIC_LAUNCH_SKIP_EMA_COPY": "1", "GPIC_MFU_NO_EMA": "1"}, FW, "1"), ("g1_with_ema", {"GPIC_LAUNCH_SKIP_EMA_COPY": "1"}, FW, "0"),
                            ("g1_not_asked", {"GPIC_MFU_NO_EMA": "1"}, FW, "0"), ("g1_old_model", {"GPIC_LAUNCH_SKIP_EMA_COPY": "1", "GPIC_MFU_NO_EMA": "1"}, FW_OLD, "0")):
    r = launch(name, env=env, wt=wt)
    check(f"lever G1, {name}: launches.txt says skip_ema_copy={want}", expect(r, 0, levers=ON.replace("skip_ema_copy=0", f"skip_ema_copy={want}"), fit=8))
r = launch("default", env={"GPIC_LAUNCH_RANKS": ""})
check("the switch not set: off", expect(r, 0, n=None, levers=ON.replace("ranks=1", "ranks=0"), text=[r"main\.py exit 0"]))
r = launch("gate_beside", gate="pass:4")
late = [k for k, v in r["ranks"].items() if r["gate_end"] is None or v.get("t_fit", 0) < r["gate_end"] or v["t_start"] > r["gate_end"]]
check("the gate beside the start: every rank starts before the verdict and waits for it; one line",
      expect(r, 0, levers=ON, text=[r"\[gate\] healthy; fit waited \d+\.\d s"], fit=8)
      + ([f"ranks not beside the gate: {late}"] if late else []) + ([] if r["out"].count("[gate] healthy") == 1 else [f"{r['out'].count('[gate] healthy')} verdict lines"]))
r = launch("gate_fail", gate="fail:2", ranks="*=fit:30")
check("the gate beside the start, unhealthy: 98, no rank reaches fit",
      expect(r, 98, levers=ON, text=[r"\[gate\] exit status 1 after", r"unhealthy GPU on \S+; exit 98"], no_text=[r"main\.py exit"], fit=0, wall=(2, 25)))
r = launch("gate_off", gate="pass:2", env={"GPIC_LAUNCH_GATE_BG": "0"})
check("the gate in front: the ranks start after it",
      expect(r, 0, levers=ON.replace("gate_bg=1", "gate_bg=0"), no_text=[r"\[gate\]"], fit=8)
      + ([] if r["gate_end"] and all(v["t_start"] >= r["gate_end"] for v in r["ranks"].values()) else ["a rank started before the gate's end"]))
r = launch("rank_dies", ranks="3=die:5:1.5,*=fit:60")
check("rank 3 ends with 5: KILL for the others, status 5", expect(r, 5, levers=ON, text=[r"rank 3 ended with 5; KILL for the other 7", r"main\.py exit 5"], wall=(1, 25)))
r = launch("rank0_dies", ranks="0=die:7:1.5,*=fit:60")
check("rank 0 ends with 7: KILL for the others, status 7", expect(r, 7, levers=ON, text=[r"rank 0 ended with 7; KILL for the other 7", r"main\.py exit 7"], wall=(1, 25)))
r = launch("two_die", ranks="2=die:4:1,6=die:9:1.6,*=fit:60")
check("two ranks fail: the first one's status", expect(r, 4, levers=ON, text=[r"rank 2 ended with 4", r"main\.py exit 4"], wall=(1, 25)))
r = launch("rank0_first", ranks="0=fit:1,*=fit:5")
check("rank 0 ends first: the script waits for the others", expect(r, 0, levers=ON, text=[r"main\.py exit 0"], no_text=[r"KILL", r"after rank 0"], fit=8, wall=(5, 30)))
r = launch("late_fail", ranks="0=fit:1,5=die:9:4,*=fit:2")
check("a rank fails after rank 0 ended with 0: status 0", expect(r, 0, levers=ON, text=[r"a rank ended with 9 after rank 0 ended with 0", r"main\.py exit 0"], no_text=[r"KILL"], wall=(4, 30)))
r = launch("late_hang", ranks="0=fit:1,6=hang:1,*=fit:1", env={"GPIC_LAUNCH_RANKS_EXIT_S": "4"})
check("a rank hangs at its exit after rank 0 ended with 0: KILL after the limit, status 0",
      expect(r, 0, levers=ON, text=[r"1 ranks did not end in 4 s; KILL", r"a rank ended with 137 after rank 0 ended with 0", r"main\.py exit 0"], wall=(4, 30)))
r = launch("cut_save", ranks="*=term0:60:1.5", cut="fit")
check("a cut, ranks that save and stop: the script waits for them, status 0",
      expect(r, 0, levers=ON, text=[r"train_ladder\.sh: TERM", r"main\.py exit 0"], no_text=[r"KILL"], fit=8, wall=(1.5, 40)))
r = launch("cut_save_run", ranks="*=term0:60:1.5", cut="fit", partial_ok=False)
check("the same cut in a run that is not done: 99", expect(r, 99, levers=ON, text=[r"train_ladder\.sh: TERM", r"main\.py exit 0 before max_steps", r"main\.py exit 99"], fit=8))
r = launch("cut_die", ranks="*=fit:60", cut="fit")
check("a cut, ranks that die of the TERM: 143", expect(r, 143, levers=ON, text=[r"train_ladder\.sh: TERM", r"main\.py exit 143"], wall=(0, 40)))
r = launch("cut_early", ranks="*=fit:60", gate="pass:25", cut="started")
check("a cut while the ranks wait for the gate: 143, nothing left", expect(r, 143, levers=ON, text=[r"train_ladder\.sh: TERM", r"main\.py exit 143"], no_text=[r"unhealthy"], fit=0, wall=(0, 25)))
r = launch("devices_4", args=("--trainer.devices", "4"))
check("--trainer.devices 4: 4 ranks, 24 threads", expect(r, 0, n=4, levers=ON.replace("=12 ", "=24 "), omp="24", fit=4))
r = launch("devices_1", args=("--trainer.devices=1",))
check("--trainer.devices=1: one process as before", expect(r, 0, n=None, levers=ON.replace("=12 ranks=1", "=96 ranks=0")))
r = launch("devices_auto", args=("--trainer.devices", "auto"))
check("--trainer.devices auto: one process as before", expect(r, 0, n=None, levers=f"gate_bg=1 vec_isa_memo={MEMO} omp=off ranks=0 skip_ema_copy=0"))
r = launch("caller_omp", env={"OMP_NUM_THREADS": "5"})
check("a thread count set by the caller stays in every rank", expect(r, 0, levers=f"gate_bg=1 vec_isa_memo={MEMO} omp=off ranks=1 skip_ema_copy=0", omp="5", fit=8))
pr = []   # rank 0 and the others end in the same moment: the helper that limits the others' exit must not outlive the script
for i in range(12):   # (a first version left its sleep behind in 1 of 8 launches)
    r = launch(f"together{i}", ranks="*=fit:0.2", gate="pass:0.1")
    pr += [f"launch {i}: {x}" for x in expect(r, 0, levers=ON, fit=8)]
check("12 launches whose ranks all end together: status 0, no process left", pr)
pr = []
for i in range(12):
    r = launch(f"cut_together{i}", ranks="*=term0:60:0.3", cut="fit")
    pr += [f"launch {i}: {x}" for x in expect(r, 0, levers=ON, text=[r"train_ladder\.sh: TERM"], fit=8)]
check("12 cuts whose ranks all save and stop together: status 0, no process left", pr)
bad = [n for n, p in results if p]
print(f"TEST_LAUNCH_RANKS {'PASS' if not bad else 'FAIL'}: {len(results)} cases" + (f", {len(bad)} failed" if bad else ""), flush=True)
sys.exit(1 if bad else 0)
