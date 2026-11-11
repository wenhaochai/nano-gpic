"""Test, without a GPU, of the launch guard of jobs/train_ladder.sh and of GPIC_FAIL_FAST in baselines/PixelGen/main.py
(2026-11-10, w2503: wandb.init failed with an HTTP 500 before the first step; rank 0 printed its traceback and then hung in
the interpreter's exit, ranks 1-7 spun in the barrier, GPU 0 idle, until a hand cancel 10 min later).
The real launcher, a made-up worktree with a stub tools/gpu_health.py and a stub main.py that every launch (and, with lever
B, every rank) runs. The stub runs the real before_fit (the gate), writes the first-step marker through the real
src/callbacks/first_step.py, and on a failure leaves through the real leave_after_failure of main.py, with the two things
that can keep w2503's rank 0 alive in place: Lightning's NCCL destroy at exit (here blocked for ever, as a finalize that waits
for ranks stuck in a collective) and a thread that is not a daemon (8 s). In Lightning's start (lever B off) the failing process has
two children that ignore TERM (ranks 1-7 in the barrier, data workers).
STUB_PLAN = one mode per launch attempt (the last one repeats):
  ok:<s>        the gate, the first step (marker), <s> s of training, exit 0; a TERM after the first step: exit 0 (saved)
  raise:<s>     the gate, <s> s, then an exception before the first step (the w2503 failure)
  late:<s>      the gate, the first step, <s> s, then an exception
  hang[:ign]    the gate, then no first step for ever (ign: TERM ignored)
  slowsave:<s>  the gate, no first step; a TERM: exit 0 after <s> s
usage: python tools/mfu/tests/test_launch_guard.py <new scratch directory>     (a vis node; about 6 min)
Result: TEST_LAUNCH_GUARD PASS: N cases."""
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
REAL_PIXGEN = f"{WT}/baselines/PixelGen"
LAUNCHER = os.environ.get("TEST_LAUNCHER") or f"{WT}/jobs/train_ladder.sh"
OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT)
results = []


def check(name, problems):
    results.append((name, problems))
    print(("ok   " if not problems else "FAIL ") + name + "".join(f"\n       {p}" for p in problems), flush=True)


STUB_GATE = '''# stub of tools/gpu_health.py: STUB_GATE = pass | fail [:seconds]
import os, sys, time
mode, secs = (os.environ.get("STUB_GATE", "pass:0.3") + ":0.3").split(":")[:2]
time.sleep(float(secs))
print(f"stubhost: {'UNHEALTHY GPUs 0' if mode == 'fail' else 'healthy'}", flush=True)
sys.exit(1 if mode == "fail" else 0)
'''
STUB_MAIN = '''# stub of baselines/PixelGen/main.py (tools/mfu/tests/test_launch_guard.py)
import ast, atexit, json, os, signal, subprocess, sys, textwrap, threading, time
REAL = os.environ["STUB_REAL_PIXGEN"]
def real_fn(name, cls=None):   # a function of the real main.py, from its source
    src = open(REAL + "/main.py").read()
    body = ast.parse(src).body
    if cls:
        body = next(c for c in body if isinstance(c, ast.ClassDef) and c.name == cls).body
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"os": os, "sys": sys, "time": time}
    exec(textwrap.dedent(ast.get_source_segment(src, fn, padded=True)), ns)
    return ns[name]
attempt = int(os.environ.get("GPIC_LAUNCH_ATTEMPT") or 0)
r = os.environ.get("LOCAL_RANK", "x")
first = r in ("x", "0")
plan = os.environ["STUB_PLAN"].split(",")
mode = plan[min(attempt, len(plan) - 1)].split(":")
rec = {"attempt": attempt, "rank": r, "pid": os.getpid(), "t_start": time.time(), "mode": mode,
       "env": {k: os.environ.get(k) for k in ("GPIC_LAUNCH_ID", "GPIC_FIRST_STEP_FILE", "OMP_NUM_THREADS", "LD_PRELOAD", "TMPDIR",
                                              "GPIC_GATE_" + "RC", "MASTER_PORT", "STUB_MARK")}}
def dump():
    p = f"{os.environ['STUB_OUT']}.{attempt}.{r}"
    with open(p + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(p + ".tmp", p)
dump()
real_fn("before_fit", "ReWriteRootDirCli")(None)
rec["t_fit"] = time.time()
dump()
saved = []
def first_step():
    sys.path.insert(0, REAL)
    from src.callbacks.first_step import FirstStepMarker
    class T:
        is_global_zero = first
        global_step = 16521   # a resumed launch: the step count does not start at 0
    FirstStepMarker().on_train_batch_end(T(), None, None, None, 0)
    signal.signal(signal.SIGTERM, lambda *a: saved.append(time.time()))   # save_on_term from the first step on
    rec["t_first"] = time.time()
    dump()
def train(s):
    t = time.time()
    while not saved and time.time() - t < s:
        time.sleep(0.05)
    if saved:
        time.sleep(0.3)   # the save
        rec["t_saved"] = time.time()
        dump()
def fail(msg):
    leave = real_fn("leave_after_failure")
    try:
        raise RuntimeError(msg)
    except Exception as e:
        rec["t_raise"] = time.time()
        dump()
        if os.environ.get("GPIC_FAIL_FAST", "1") != "1":
            raise
        leave(e)
def stuck_at_exit():   # what kept w2503's rank 0 alive after its traceback
    import torch.distributed
    import lightning.fabric.utilities.distributed as LD
    def blocked(*a, **k):
        rec["t_destroy"] = time.time()
        dump()
        while True:
            time.sleep(1)
    LD._distributed_is_initialized = lambda: True
    torch.distributed.destroy_process_group = blocked
    atexit.register(LD._destroy_dist_connection)   # as distributed.py:287 does for NCCL
    threading.Thread(target=lambda: time.sleep(8), daemon=False).start()   # the old exit waits 8 s for it, then the destroy
    if r == "x":   # Lightning's start: ranks 1-7 and the data workers, which ignore TERM
        for _ in range(2):
            subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3600)"])
m = mode[0]
if m == "ok":
    first_step()
    train(float(mode[1]))
elif m in ("raise", "late"):
    if m == "late":
        first_step()
    if first:
        stuck_at_exit()
        time.sleep(float(mode[1]))
        fail("made-up CommError: Failed to get resume status for run (HTTP 500)" if m == "raise" else "made-up failure after the first step")
    else:   # ranks 1-7: in a collective with rank 0 until they get KILL
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(3600)
elif m == "hang":
    if len(mode) > 1 and mode[1] == "ign":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(3600)
elif m == "slowsave":
    got = []
    signal.signal(signal.SIGTERM, lambda *a: got.append(1))
    while not got:
        time.sleep(0.05)
    time.sleep(float(mode[1]))
    rec["t_saved"] = time.time()
    dump()
'''
assert "GPIC_GATE_RC" not in STUB_MAIN
FW, FW_OLD = f"{OUT}/worktree", f"{OUT}/worktree_old"   # FW_OLD: a worktree without src/callbacks/first_step.py
for fw in (FW, FW_OLD):
    os.makedirs(f"{fw}/baselines/PixelGen/src/callbacks")
    os.makedirs(f"{fw}/tools")
    with open(f"{fw}/tools/gpu_health.py", "w") as f:
        f.write(STUB_GATE)
    with open(f"{fw}/baselines/PixelGen/main.py", "w") as f:
        f.write(STUB_MAIN + "# this main.py knows GPIC_GATE_RC\n")
    with open(f"{fw}/baselines/PixelGen/src/lightning_model_gpic.py", "w") as f:
        f.write("# stub\n")
    if fw == FW:   # the launcher looks for the class name; the stub uses the real file
        with open(f"{fw}/baselines/PixelGen/src/callbacks/first_step.py", "w") as f:
            f.write("# stub: class FirstStepMarker lives in the real worktree\n")
count = [0]


def session(sid, wait=3.0):
    t = time.time()
    while True:
        pids = subprocess.run(["pgrep", "-s", str(sid)], capture_output=True, text=True).stdout.split()
        if not pids or time.time() - t > wait:
            return pids
        time.sleep(0.2)


def recs(d):
    out = {}
    for f in os.listdir(d):
        m = re.fullmatch(r"main\.json\.(\d+)\.(\w+)", f)
        if m:
            with open(f"{d}/{f}") as g:
                out[(int(m.group(1)), m.group(2))] = json.load(g)
    return out


def launch(name, plan, ranks=False, env=None, cut=None, partial_ok=True, wt=None, gate="pass:0.3", timeout=200):
    count[0] += 1
    d = f"{OUT}/{name}"
    os.makedirs(f"{d}/tmp")
    os.makedirs(f"{d}/bin")
    with open(f"{d}/bin/nproc", "w") as f:
        f.write("#!/bin/sh\necho 96\n")
    os.chmod(f"{d}/bin/nproc", 0o755)
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("GPIC_", "LADDER_", "SLURM_", "STUB_", "OMP_", "PROBE_", "MASTER_"))
         and k not in ("FETCHED_JOB_ID", "BENCH_OUT", "LOCAL_RANK", "NODE_RANK", "WORLD_SIZE", "LD_PRELOAD")}
    e.update(PATH=f"{d}/bin:{os.environ['PATH']}", TMPDIR=f"{d}/tmp", LADDER_WT=wt or FW, LADDER_RUN_ROOT=f"{d}/root",
             STUB_GATE=gate, STUB_PLAN=plan, STUB_OUT=f"{d}/main.json", STUB_REAL_PIXGEN=REAL_PIXGEN, STUB_MARK=f"m{count[0]}",
             GPIC_LAUNCH_RANKS="1" if ranks else "0", GPIC_LAUNCH_RETRY_WAIT_S="2", GPIC_LAUNCH_START_TIMEOUT_S="20",
             GPIC_LAUNCH_WATCHDOG_KILL_S="3", CUDA_VISIBLE_DEVICES="")
    if partial_ok:
        e["LADDER_PARTIAL_OK"] = "1"
    e.update(env or {})
    e = {k: v for k, v in e.items() if v is not None}
    t0 = time.time()
    p = subprocess.Popen(["bash", LAUNCHER, "d640", "640", "9", "2", "4096", "--trainer.devices", "4" if ranks else "auto"], env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t_cut = None
    if cut:   # the pilot's cut: TERM to the process group once cut(d) holds
        while not cut(d) and p.poll() is None and time.time() - t0 < timeout:
            time.sleep(0.05)
        time.sleep(0.3)
        t_cut = time.time()
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        out = p.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out = p.communicate()[0] + f"\n[test] the launcher did not end in {timeout} s"
    r = {"rc": p.returncode, "out": out, "wall": time.time() - t0, "t0": t0, "t_cut": t_cut, "recs": recs(d), "launches": ""}
    lt = f"{d}/root/exp_ladder_d640/launches.txt"
    if os.path.exists(lt):
        with open(lt) as f:
            r["launches"] = f.read()
    r["left"] = session(p.pid)
    if r["left"]:
        r["left"] = subprocess.run(["ps", "-o", "pid=,ppid=,etimes=,args=", "-p", ",".join(r["left"])], capture_output=True, text=True).stdout.strip().splitlines() or r["left"]
    for q in [x.split()[0] for x in r["left"]]:
        try:
            os.kill(int(q), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    r["tmp"] = sorted(os.listdir(f"{d}/tmp"))
    return r


def expect(r, rc, attempts, text=(), no_text=(), lines=(), no_lines=(), wall=None, marker=True):
    """attempts: the launch attempts that started (main.py ran); lines / no_lines: patterns in launches.txt."""
    pr = []
    if r["rc"] != rc:
        pr.append(f"exit status {r['rc']}, expected {rc}")
    got = sorted({a for a, _ in r["recs"]})
    if got != list(range(attempts)):
        pr.append(f"attempts that started main.py: {got}, expected 0..{attempts - 1}")
    pr += [f"no /{t}/ in the output" for t in text if not re.search(t, r["out"])]
    pr += [f"/{t}/ in the output" for t in no_text if re.search(t, r["out"])]
    pr += [f"no /{t}/ in launches.txt" for t in lines if not re.search(t, r["launches"])]
    pr += [f"/{t}/ in launches.txt" for t in no_lines if re.search(t, r["launches"])]
    guard = re.findall(r"launch guard: attempt=(\d+)", r["launches"])
    if guard != [str(i) for i in range(attempts)]:
        pr.append(f"launch guard lines for attempts {guard}, expected 0..{attempts - 1}")
    if wall and not wall[0] <= r["wall"] <= wall[1]:
        pr.append(f"the launcher took {r['wall']:.1f} s, expected {wall[0]}-{wall[1]} s")
    ids = {a: v["env"]["GPIC_LAUNCH_ID"] for (a, k), v in r["recs"].items()}
    if len(set(ids.values())) != len(ids):
        pr.append(f"launch ids repeat across attempts: {ids}")
    for (a, k), v in r["recs"].items():   # each attempt gets the first environment back, plus its own launch names
        e = v["env"]
        if marker and not (e["GPIC_FIRST_STEP_FILE"] or "").endswith("first_step_" + (e["GPIC_LAUNCH_ID"] or "?")):
            pr.append(f"attempt {a} rank {k}: marker file {e['GPIC_FIRST_STEP_FILE']} is not named after its launch {e['GPIC_LAUNCH_ID']}")
        if (e["LD_PRELOAD"] or "").count("libtcmalloc") > 1:
            pr.append(f"attempt {a}: LD_PRELOAD {e['LD_PRELOAD']}")
        if e["STUB_MARK"] is None or not e["TMPDIR"]:
            pr.append(f"attempt {a}: lost the caller's environment: {e}")
    if any(f.startswith(("gpu_gate_", "first_step_")) for f in r["tmp"]):
        pr.append(f"left in TMPDIR: {r['tmp']}")
    if r["left"]:
        pr.append(f"processes left in the launcher's session: {r['left']}")
    return pr + (["output: " + r["out"][-1200:].replace("\n", " | ")] if pr else [])


def fast_exit(r, attempt=0):   # the failing rank left within seconds of its exception (GPIC_FAIL_FAST)
    m = re.findall(r"\[main\] fit failed \(RuntimeError\); leaving with status 1 (\d+\.\d) s after the traceback(.*)", r["out"])
    if not m:
        return ["no [main] fit failed line"]
    return [f"leave_after_failure took {s} s{note}" for s, note in m if float(s) > 3 or note.strip()]


def cut_when(pattern, key=None):
    def f(d):
        if key:
            return any(key in v for v in recs(d).values())
        try:
            with open(f"{d}/root/exp_ladder_d640/launches.txt") as g:
                return re.search(pattern, g.read()) is not None
        except FileNotFoundError:
            return False
    return f


RETRY1 = r"launch retry 1: status 1, reason failed before the first step"
for b in (False, True):
    tag = "B on" if b else "B off"
    r = launch(f"retry_ok_{int(b)}", "raise:1,ok:1", ranks=b)
    check(f"{tag}: a failure before the first step, then a good launch: one retry, exit 0",
          expect(r, 0, 2, text=[r"launch retry 1 of 2: status 1", r"main\.py exit 0$"], lines=[RETRY1], no_lines=[r"launch retry 2", r"watchdog"], wall=(3, 90))
          + fast_exit(r))
    r = launch(f"retry_run_{int(b)}", "raise:1,ok:1", ranks=b, partial_ok=False)
    check(f"{tag}: the same in a run that is not done: 99, as the real path ends", expect(r, 99, 2, text=[r"main\.py exit 0 before max_steps", r"main\.py exit 99"], lines=[RETRY1]))
    r = launch(f"always_raise_{int(b)}", "raise:1", ranks=b)
    check(f"{tag}: a failure before the first step every time: 2 retries, then exit 1, no process left",
          expect(r, 1, 3, text=[r"main\.py exit 1$"], lines=[RETRY1, r"launch retry 2: status 1, reason failed before the first step", r"launch retry: none left \(2 used\); status 1"],
                 no_lines=[r"launch retry 3"], wall=(4, 120)) + fast_exit(r))
    r = launch(f"hang_ok_{int(b)}", "hang,ok:1", ranks=b, env={"GPIC_LAUNCH_START_TIMEOUT_S": "15"})
    check(f"{tag}: a launch that hangs before the first step: the watchdog ends it at 15 s, one retry, exit 0",
          expect(r, 0, 2, text=[r"launch watchdog: no first step and no exit \d+ s after the start \(limit 15 s\)", r"main\.py exit 0$"],
                 lines=[r"launch watchdog: no first step", r"launch retry 1: status 125, reason watchdog: no first step in 15 s"], no_lines=[r"launch retry 2"], wall=(17, 120)))
    r = launch(f"hang_always_{int(b)}", "hang:ign", ranks=b, env={"GPIC_LAUNCH_START_TIMEOUT_S": "4"})
    check(f"{tag}: a launch that hangs every time and ignores TERM: KILL after 3 s, 2 retries, then exit 125, no process left",
          expect(r, 125, 3, text=[r"main\.py exit 125$"], lines=[r"launch retry 2: status 125", r"launch retry: none left \(2 used\); status 125"], wall=(20, 150)))
    r = launch(f"late_{int(b)}", "late:1,ok:1", ranks=b)
    check(f"{tag}: a failure after the first step: no retry, exit 1, no process left",
          expect(r, 1, 1, text=[r"main\.py exit 1$"], no_text=[r"launch retry"], no_lines=[r"launch retry", r"watchdog"]) + fast_exit(r))
    # the stub's first step comes some 5 s after its start (it imports lightning for the real callback): a 12-s limit
    r = launch(f"long_{int(b)}", "ok:25", ranks=b, env={"GPIC_LAUNCH_START_TIMEOUT_S": "12"})
    check(f"{tag}: a run longer than the start limit, first step in time: no watchdog, exit 0",
          expect(r, 0, 1, text=[r"main\.py exit 0$"], no_lines=[r"watchdog", r"launch retry"], wall=(25, 80)))
    r = launch(f"cut_wait_{int(b)}", "raise:1,ok:1", ranks=b, env={"GPIC_LAUNCH_RETRY_WAIT_S": "30"}, cut=cut_when(r"launch retry 1:"), partial_ok=False)
    check(f"{tag}: a cut during the 30 s wait before a retry: exit 99 at once, no further launch",
          expect(r, 99, 1, text=[r"train_ladder\.sh: TERM", r"main\.py exit 99 \(a cut during the wait"], lines=[RETRY1, r"launch retry 1: cut during the wait; exit 99"])
          + ([] if r["t_cut"] and r["t0"] + r["wall"] - r["t_cut"] < 5 else [f"the launcher ended {r['t0'] + r['wall'] - (r['t_cut'] or r['t0']):.1f} s after the cut"]))
    r = launch(f"cut_run_{int(b)}", "ok:60", ranks=b, cut=cut_when(None, "t_first"), partial_ok=False, env={"GPIC_LAUNCH_START_TIMEOUT_S": "12"})
    check(f"{tag}: a cut in a running launch (after its first step): save, exit 99, no retry, no watchdog",
          expect(r, 99, 1, text=[r"train_ladder\.sh: TERM", r"main\.py exit 0 before max_steps", r"main\.py exit 99"], no_lines=[r"launch retry", r"watchdog"], wall=(0, 30))
          + ([] if all("t_saved" in v for v in r["recs"].values()) else ["a rank did not save"]))
    r = launch(f"cut_start_{int(b)}", "slowsave:6", ranks=b, cut=cut_when(None, "t_fit"), partial_ok=False, env={"GPIC_LAUNCH_START_TIMEOUT_S": "3"})
    check(f"{tag}: a cut before the first step, the save outlasts the start limit: no watchdog, no retry, exit 99",
          expect(r, 99, 1, text=[r"train_ladder\.sh: TERM", r"main\.py exit 99"], no_lines=[r"launch retry", r"watchdog"], wall=(6, 40))
          + ([] if all("t_saved" in v for v in r["recs"].values()) else ["a rank did not finish its save"]))
    r = launch(f"cut_dies_{int(b)}", "hang", ranks=b, cut=cut_when(None, "t_fit"))
    check(f"{tag}: a cut before the first step, the launch dies of the TERM: exit 143, no retry",
          expect(r, 143, 1, text=[r"main\.py exit 143$"], no_lines=[r"launch retry", r"watchdog"], wall=(0, 30)))

r = launch("failfast_off", "raise:1,ok:1", env={"GPIC_FAIL_FAST": "0", "GPIC_LAUNCH_START_TIMEOUT_S": "18"})
hung = [v for (a, k), v in r["recs"].items() if a == 0 and "t_destroy" in v]
check("GPIC_FAIL_FAST=0 (the old exit): rank 0 hangs in Lightning's NCCL destroy after its traceback (w2503 reproduced); the watchdog ends it, retry, exit 0",
      expect(r, 0, 2, text=[r"made-up CommError", r"launch watchdog: no first step"], no_text=[r"\[main\] fit failed"], lines=[r"launch retry 1: status 125, reason watchdog"])
      + ([] if hung else ["rank 0 did not reach the blocked destroy"]))
r = launch("retries_off", "raise:1,ok:1", env={"GPIC_LAUNCH_RETRIES": "0"})
check("GPIC_LAUNCH_RETRIES=0: no retry, exit 1", expect(r, 1, 1, text=[r"main\.py exit 1$"], lines=[r"launch guard: attempt=0 retries=0"], no_lines=[r"launch retry"]) + fast_exit(r))
r = launch("watchdog_off", "ok:1", env={"GPIC_LAUNCH_START_TIMEOUT_S": "0"})
check("GPIC_LAUNCH_START_TIMEOUT_S=0: main.py in the foreground as before, exit 0", expect(r, 0, 1, lines=[r"launch guard: attempt=0 retries=2 retry_wait=2 start_timeout=0"]))
r = launch("defaults", "ok:1", env={"GPIC_LAUNCH_RETRY_WAIT_S": None, "GPIC_LAUNCH_START_TIMEOUT_S": None})
check("the defaults: 2 retries, 60 s wait, 1500 s start limit", expect(r, 0, 1, lines=[r"launch guard: attempt=0 retries=2 retry_wait=60 start_timeout=1500"]))
r = launch("old_worktree", "raise:1,ok:1", wt=FW_OLD)
check("a worktree without the marker callback: the guard stays off, exit 1",
      expect(r, 1, 1, lines=[r"launch guard: attempt=0 retries=0 retry_wait=2 start_timeout=0"], no_lines=[r"launch retry"], marker=False))
r = launch("gate_fail", "ok:1", gate="fail:0.5")
check("the gate fails (98): no retry", expect(r, 98, 1, text=[r"unhealthy GPU on \S+; exit 98"], no_lines=[r"launch retry"]))
bad = [n for n, p in results if p]
print(f"TEST_LAUNCH_GUARD {'PASS' if not bad else 'FAIL'}: {len(results)} cases" + (f", {len(bad)} failed" if bad else ""), flush=True)
sys.exit(1 if bad else 0)
