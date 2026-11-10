"""Test, without a GPU, of the launch levers of jobs/train_ladder.sh (gpic-28 2026-11-09, logs/mfu_infra_scan_1109b.yaml
item 6). Three parts:
  a  the vector-set memo of src/utils/patch_bugs.py against the compiler's own check (torch._inductor.cpu_vec_isa): the
     same list of vector sets with and without the memo, hits only for answers that were True, no file for a False
     answer, the real check after any error, several processes at once
  b  before_fit of baselines/PixelGen/main.py, taken from the file itself: the verdict file, a gate that ended without a
     verdict, the wait limit
  c  the real launcher with a made-up worktree, a stub tools/gpu_health.py and a stub main.py that runs the real
     before_fit: the gate beside the start and in front of it, a failed gate, a main.py that ends first, a cut (TERM to
     the process group), the thread count, the switches
usage: python tools/mfu/tests/test_launch_levers.py <new scratch directory> [a] [b] [c]
Run on a vis node (part a compiles the check's test library once). Result: TEST_LAUNCH_LEVERS PASS: N cases.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
PIXGEN = f"{WT}/baselines/PixelGen"
REAL_MAIN = f"{PIXGEN}/main.py"
LAUNCHER = f"{WT}/jobs/train_ladder.sh"
OUT = os.path.abspath(sys.argv[1])
PARTS = sys.argv[2:] or ["a", "b", "c"]
os.makedirs(OUT)   # a new directory: nothing of an earlier run is read
results = []


def check(name, problems):
    results.append((name, problems))
    print(("ok   " if not problems else "FAIL ") + name + "".join(f"\n       {p}" for p in problems), flush=True)


# ---------------------------------------------------------------------------------------------------------------- part a
MEMO_CHILD = '''
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
mode = sys.argv[2]
t0 = time.time()
import src.utils.patch_bugs as pb
import platform
from torch._inductor import cpu_vec_isa as cvi
real = {"calls": 0, "true": 0}
def count(inner):
    def counted(self, code):
        real["calls"] += 1
        ok = False if mode == "false" else inner(self, code)
        real["true"] += ok is True
        return ok
    return counted
memo_on = hasattr(pb, "_check_build_real")
if memo_on:
    pb._check_build_real = count(pb._check_build_real)
else:
    cvi.VecISA.check_build = count(cvi.VecISA.check_build)
if mode == "release":
    platform.release = lambda: "another-kernel"
if mode == "cpuerr":
    def boom():
        raise OSError("made-up: no /proc/cpuinfo")
    pb._vec_isa_cpu = boom
isas = [str(i) for i in cvi.valid_vec_isa_list()]
print("RESULT " + json.dumps({"isas": isas, "stats": pb.VEC_ISA_MEMO_STATS, "real": real, "memo_on": memo_on,
                              "s": round(time.time() - t0, 1)}), flush=True)
'''


def part_a():
    d = f"{OUT}/a"
    os.makedirs(d)
    child = f"{d}/memo_child.py"
    with open(child, "w") as f:
        f.write(MEMO_CHILD)
    base = {k: v for k, v in os.environ.items() if not k.startswith(("GPIC_", "TORCHINDUCTOR_"))}
    cache, cache_new = f"{d}/cache", f"{d}/cache_new"

    def start(mode, cache_dir, memo):
        e = {**base, "TORCHINDUCTOR_CACHE_DIR": cache_dir, "OMP_NUM_THREADS": "2"}
        if memo:
            e["GPIC_VEC_ISA_MEMO"] = memo
        return subprocess.Popen([sys.executable, child, PIXGEN, mode], env=e, cwd=d, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

    def end(p):
        out = p.communicate()[0]
        line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
        return json.loads(line[7:]) if line and p.returncode == 0 else {"error": out[-800:].replace("\n", " | ")}

    def files(m):
        return sorted(os.listdir(m)) if os.path.isdir(m) else []

    def memo_holds(m, n):   # n empty files named by a sha256, nothing else (no temporary file left)
        fs = files(m)
        ok = len(fs) == n and all(re.fullmatch(r"[0-9a-f]{64}", f) and os.path.getsize(f"{m}/{f}") == 0 for f in fs)
        return [] if ok else [f"{m} holds {len(fs)} files ({fs[:2]}...), expected {n} empty files"]

    def cmp(r, isas, stats, calls):
        if "error" in r:
            return ["the process failed: " + r["error"]]
        pr = []
        if r["isas"] != isas:
            pr.append(f"vector sets {r['isas']}, expected {isas}")
        if r["stats"] != stats:
            pr.append(f"memo counts {r['stats']}, expected {stats}")
        if r["real"]["calls"] != calls:
            pr.append(f"{r['real']['calls']} real checks, expected {calls}")
        return pr

    n = end(start("plain", cache, None))   # the compiler's own check: builds the test libraries into <cache>
    if "error" in n:
        return check("a: no memo directory: the compiler's own answers", ["the process failed: " + n["error"]])
    isas, calls, true = n["isas"], n["real"]["calls"], n["real"]["true"]
    pr = []
    if n["memo_on"] or any(n["stats"].values()):
        pr.append(f"the memo is on without a directory: {n['stats']}")
    if true < 1:
        pr.append(f"nothing to test on this CPU: {calls} checks, {true} True")
    check(f"a: no memo directory: the compiler's own answers ({isas}; {calls} checks, {true} True; {n['s']} s)", pr)
    if pr:
        return
    m = f"{d}/memo"
    r = end(start("plain", cache, m))
    check(f"a: empty memo: every check asked, the True answers stored ({r.get('s')} s)",
          cmp(r, isas, {"hit": 0, "miss": calls, "stored": true, "error": 0}, calls) + memo_holds(m, true))
    m_rel, m_cpu, m_false, m_file = f"{d}/memo_release", f"{d}/memo_cpuerr", f"{d}/memo_false", f"{d}/memo_is_a_file"
    shutil.copytree(m, m_rel)
    shutil.copytree(m, m_cpu)
    with open(m_file, "w"):
        pass
    ps = {"hit": start("plain", cache_new, m), "file": start("plain", cache, m_file), "release": start("release", cache, m_rel),
          "cpuerr": start("cpuerr", cache, m_cpu), "false": start("false", cache, m_false)}
    rs = {k: end(p) for k, p in ps.items()}
    built = [f for _, _, fs in os.walk(cache_new) for f in fs if f.endswith((".so", ".cpp", ".o"))]
    check(f"a: filled memo: the True answers read, no check, nothing built ({rs['hit'].get('s')} s)",
          cmp(rs["hit"], isas, {"hit": true, "miss": calls - true, "stored": 0, "error": 0}, calls - true) + memo_holds(m, true)
          + ([f"built in the new cache directory: {built[:3]}"] if built and calls == true else []))
    check("a: memo path is a file: the real check, each failed store counted",
          cmp(rs["file"], isas, {"hit": 0, "miss": calls, "stored": 0, "error": true}, calls))
    check("a: another kernel release: no hit, new files beside the old ones",
          cmp(rs["release"], isas, {"hit": 0, "miss": calls, "stored": true, "error": 0}, calls) + memo_holds(m_rel, 2 * true))
    check("a: the CPU cannot be read: the real check every time, nothing stored",
          cmp(rs["cpuerr"], isas, {"hit": 0, "miss": 0, "stored": 0, "error": calls}, calls) + memo_holds(m_cpu, true))
    f = rs["false"]
    pr = ["the process failed: " + f["error"]] if "error" in f else []
    if not pr and (f["isas"] or f["stats"]["stored"] or f["stats"]["hit"] or f["stats"]["error"]
                   or f["stats"]["miss"] != f["real"]["calls"] or f["real"]["calls"] < 1):
        pr.append(f"vector sets {f['isas']}, memo counts {f['stats']}, {f['real']['calls']} real checks")
    check("a: False answers: no vector set, nothing stored", pr + memo_holds(m_false, 0))
    m_par = f"{d}/memo_parallel"
    rs = [end(p) for p in [start("plain", cache, m_par) for _ in range(4)]]
    pr = []
    for r in rs:
        if "error" in r:
            pr.append("a process failed: " + r["error"])
        elif (r["isas"] != isas or r["stats"]["error"] or r["stats"]["hit"] + r["stats"]["miss"] != calls
              or r["stats"]["stored"] > r["stats"]["miss"] or r["real"]["calls"] != r["stats"]["miss"]):
            pr.append(f"vector sets {r['isas']}, memo counts {r['stats']}, {r['real']['calls']} real checks")
    check("a: four processes at once on an empty memo: the same answers, one file per True answer", pr + memo_holds(m_par, true))


# ---------------------------------------------------------------------------------------------------------------- part b
LOAD_FIT = '''
import ast, os, sys, textwrap, time
def real_before_fit(path):   # the method's own source, from the real main.py
    src = open(path).read()
    fn = next(n for c in ast.parse(src).body if isinstance(c, ast.ClassDef) and c.name == "ReWriteRootDirCli"
              for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "before_fit")
    ns = {"os": os, "sys": sys, "time": time}
    exec(textwrap.dedent(ast.get_source_segment(src, fn, padded=True)), ns)
    return ns["before_fit"]
'''
FIT_CHILD = LOAD_FIT + '''
fit = real_before_fit(sys.argv[1])
print("READY", flush=True)
t0 = time.time()
fit(None)
print(f"RETURNED {time.time() - t0:.2f}", flush=True)
'''


def state(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return None


def part_b():
    d = f"{OUT}/b"
    os.makedirs(d)
    child = f"{d}/fit_child.py"
    with open(child, "w") as f:
        f.write(FIT_CHILD)
    base = {k: v for k, v in os.environ.items() if not k.startswith("GPIC_") and k != "LOCAL_RANK"}

    def rc_file(name, text=None):
        p = f"{d}/{name}.rc"
        if text is not None:
            with open(p + ".tmp", "w") as f:
                f.write(text)
            os.replace(p + ".tmp", p)
        return p

    def run(name, env, want_rc, want, lo=0.0, hi=1.0, late=None):
        # want: a pattern the output must hold, None = no [gate] line; lo, hi: the seconds before_fit may take
        p = subprocess.Popen([sys.executable, child, REAL_MAIN], env={**base, **env}, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        first = p.stdout.readline().strip()
        t0 = time.time()
        if late:
            time.sleep(late[1])
            rc_file(late[0], "0\n")
        out = p.stdout.read()
        rc = p.wait()
        dt = time.time() - t0
        pr = []
        if first != "READY":
            pr.append(f"the process did not reach before_fit: {first!r}")
        if rc != want_rc:
            pr.append(f"exit status {rc}, expected {want_rc}")
        if ("RETURNED" in out) != (want_rc == 0):
            pr.append("before_fit returned" if "RETURNED" in out else "before_fit did not return")
        if (want is None and "[gate]" in out) or (want is not None and not re.search(want, out)):
            pr.append(f"expected {'no [gate] line' if want is None else '/' + want + '/'}")
        if not lo <= dt <= hi:
            pr.append(f"{dt:.2f} s, expected {lo} to {hi} s")
        check(f"b: {name}", pr + ([f"output: {out.strip()!r}"] if pr else []))

    gone = subprocess.Popen(["true"])
    gone.wait()                                       # a pid that is no longer there
    zombie = subprocess.Popen(["true"])               # ended and not waited for
    alive = subprocess.Popen(["sleep", "300"])
    t = time.time()
    while state(zombie.pid) != "Z" and time.time() - t < 5:
        time.sleep(0.02)
    pr = [] if (state(gone.pid), state(zombie.pid)) == (None, "Z") and state(alive.pid) in ("S", "R") else [
        f"states {state(gone.pid)}, {state(zombie.pid)}, {state(alive.pid)}"]
    check("b: three made-up gate processes: gone, zombie, alive", pr)
    G, Z, A = str(gone.pid), str(zombie.pid), str(alive.pid)
    run("no gate file named: fit goes on, no line", {}, 0, None)
    run("verdict 0: fit goes on", {"GPIC_GATE_RC": rc_file("ok", "0\n"), "GPIC_GATE_PID": G}, 0,
        r"\[gate\] healthy; fit waited 0\.\d s for the verdict")
    run("verdict 0 on rank 3: fit goes on, no line", {"GPIC_GATE_RC": rc_file("ok"), "GPIC_GATE_PID": G, "LOCAL_RANK": "3"}, 0, None)
    run("verdict 0, gate a zombie: fit goes on", {"GPIC_GATE_RC": rc_file("ok"), "GPIC_GATE_PID": Z}, 0, r"\[gate\] healthy")
    run("verdict 1: exit 98", {"GPIC_GATE_RC": rc_file("bad", "1\n"), "GPIC_GATE_PID": G}, 98,
        r"\[gate\] exit status 1 after 0\.\d s: no fit")
    run("verdict 1 on rank 3: exit 98, no line", {"GPIC_GATE_RC": rc_file("bad"), "GPIC_GATE_PID": G, "LOCAL_RANK": "3"}, 98, None)
    run("verdict file holds no number: exit 98", {"GPIC_GATE_RC": rc_file("junk", "healthy\n"), "GPIC_GATE_PID": A}, 98,
        r"\[gate\] exit status -1 after")
    run("verdict file is empty: exit 98", {"GPIC_GATE_RC": rc_file("empty", ""), "GPIC_GATE_PID": A}, 98,
        r"\[gate\] exit status -1 after")
    run("no file, gate gone: exit 98 at once", {"GPIC_GATE_RC": rc_file("none1"), "GPIC_GATE_PID": G}, 98,
        r"\[gate\] no verdict after 0\.\d s: no fit")
    run("no file, gate a zombie: exit 98 at once", {"GPIC_GATE_RC": rc_file("none2"), "GPIC_GATE_PID": Z}, 98,
        r"\[gate\] no verdict after 0\.\d s: no fit")
    run("no file, gate alive, limit 1 s: exit 98 at the limit",
        {"GPIC_GATE_RC": rc_file("none3"), "GPIC_GATE_PID": A, "GPIC_GATE_WAIT_S": "1"}, 98,
        r"\[gate\] no verdict after 1\.\d s: no fit", lo=0.9, hi=2.5)
    run("no file, no gate pid, limit 1 s: exit 98 at the limit", {"GPIC_GATE_RC": rc_file("none4"), "GPIC_GATE_WAIT_S": "1"}, 98,
        r"\[gate\] no verdict after 1\.\d s: no fit", lo=0.9, hi=2.5)
    run("file comes 0.6 s later, gate alive: fit goes on", {"GPIC_GATE_RC": rc_file("late"), "GPIC_GATE_PID": A}, 0,
        r"\[gate\] healthy; fit waited 0\.[5-9] s", lo=0.5, hi=2.0, late=("late", 0.6))
    alive.kill()
    alive.wait()
    zombie.wait()


# ---------------------------------------------------------------------------------------------------------------- part c
STUB_GATE = '''# stub of tools/gpu_health.py (tools/mfu/tests/test_launch_levers.py): STUB_GATE = pass | fail | crash | hang [:seconds]
import os, subprocess, sys, time
mode, secs = (os.environ.get("STUB_GATE", "pass:1") + ":1").split(":")[:2]
log = os.environ["STUB_GATE_LOG"]
with open(log, "a") as f:
    f.write(f"start {time.time():.3f} omp={os.environ.get('OMP_NUM_THREADS')}\\n")
if mode == "crash":
    print("made-up crash of the gate", flush=True)
    sys.exit(3)
secs = "600" if mode == "hang" else secs
kids = [subprocess.Popen([sys.executable, "-c", "import sys, time; time.sleep(float(sys.argv[1]))", secs,
                          os.environ["STUB_TOKEN"]]) for _ in range(2)]   # the per-GPU workers
for k in kids:
    k.wait()
print(f"GPU 0: 100 TFLOP/s, 1980 MHz, 60 C, reasons 0x0{'  SLOW' if mode == 'fail' else ''}")
print(f"stubhost: median 100 TFLOP/s; {'UNHEALTHY GPUs 0' if mode == 'fail' else 'healthy'}", flush=True)
with open(log, "a") as f:
    f.write(f"end {time.time():.3f}\\n")
sys.exit(1 if mode == "fail" else 0)
'''
STUB_MAIN = LOAD_FIT + '''
# stub of baselines/PixelGen/main.py (tools/mfu/tests/test_launch_levers.py): STUB_MAIN = fit[:seconds] | die:<status>:<seconds>
import json
rec = {"t_start": time.time(), "argv": sys.argv[1:], "cwd": os.getcwd(),
       "env": {k: os.environ.get(k) for k in ("GPIC_GATE_" + "RC", "GPIC_GATE_" + "PID", "OMP_NUM_THREADS", "GPIC_VEC_ISA_MEMO")}}
def dump():
    with open(os.environ["STUB_OUT"] + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(os.environ["STUB_OUT"] + ".tmp", os.environ["STUB_OUT"])
dump()
mode = os.environ.get("STUB_MAIN", "fit").split(":")
if mode[0] == "die":
    time.sleep(float(mode[2]))
    sys.exit(int(mode[1]))
real_before_fit(os.environ["STUB_REAL_MAIN"])(None)
rec["t_fit"] = time.time()
dump()
if len(mode) > 1:
    time.sleep(float(mode[1]))   # the training
'''
assert "GPIC_GATE_RC" not in STUB_MAIN   # the launcher looks for this name in main.py; only the first worktree's has it
MEMO = "/scratch/gpfs/GROUP/USER/cache/inductor/vec_isa_memo"   # the launcher's default


def left(pattern, wait=3.0):   # processes whose command line holds the pattern, after up to <wait> s
    t = time.time()
    while True:
        pids = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        if not pids or time.time() - t > wait:
            return pids
        time.sleep(0.2)


def part_c():
    c = f"{OUT}/c"
    fw, fw_old = f"{c}/worktree", f"{c}/worktree_old"
    for path, knows in ((fw, True), (fw_old, False)):
        os.makedirs(f"{path}/baselines/PixelGen")
        os.makedirs(f"{path}/tools")
        with open(f"{path}/tools/gpu_health.py", "w") as f:
            f.write(STUB_GATE)
        with open(f"{path}/baselines/PixelGen/main.py", "w") as f:
            f.write(STUB_MAIN + ("# this main.py knows GPIC_GATE_RC\n" if knows else ""))
    count = [0]

    def launch(name, gate="pass:0.3", main="fit", env=None, args=(), wt=fw, nproc=96, cut=None):
        count[0] += 1
        d, tok = f"{c}/{name}", f"lvtok{os.getpid()}n{count[0]}"
        os.makedirs(f"{d}/tmp")
        os.makedirs(f"{d}/bin")
        with open(f"{d}/bin/nproc", "w") as f:   # the CPU count the launcher sees
            f.write(f"#!/bin/sh\necho {nproc}\n")
        os.chmod(f"{d}/bin/nproc", 0o755)
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("GPIC_", "LADDER_", "SLURM_", "STUB_", "OMP_", "PROBE_")) and k not in ("FETCHED_JOB_ID", "BENCH_OUT", "LOCAL_RANK")}
        e.update(PATH=f"{d}/bin:{os.environ['PATH']}", TMPDIR=f"{d}/tmp", LADDER_WT=wt, LADDER_RUN_ROOT=f"{d}/root",
                 LADDER_PARTIAL_OK="1", STUB_GATE=gate, STUB_MAIN=main, STUB_OUT=f"{d}/main.json", STUB_GATE_LOG=f"{d}/gate.log",
                 STUB_TOKEN=tok, STUB_REAL_MAIN=REAL_MAIN)
        e.update(env or {})
        t0 = time.time()
        p = subprocess.Popen(["bash", LAUNCHER, "d640", "640", "9", "2", "4096", "--tok", tok, *args], env=e,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        if cut:   # the pilot's cut: TERM to the whole process group, here once <cut> holds
            while not cut(d) and p.poll() is None and time.time() - t0 < 60:
                time.sleep(0.05)
            time.sleep(0.5)
            os.killpg(p.pid, signal.SIGTERM)
        try:
            out = p.communicate(timeout=90)[0]
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out = p.communicate()[0] + "\n[test] the launcher did not end in 90 s"
        r = {"rc": p.returncode, "out": out, "wall": time.time() - t0, "main": None, "levers": None}
        if os.path.exists(f"{d}/main.json"):
            with open(f"{d}/main.json") as f:
                r["main"] = json.load(f)
        lt = f"{d}/root/exp_ladder_d640/launches.txt"
        if os.path.exists(lt):
            with open(lt) as f:
                r["levers"] = next((l.split("launch levers: ")[1].strip() for l in f if "launch levers: " in l), None)
        g = []
        if os.path.exists(f"{d}/gate.log"):
            with open(f"{d}/gate.log") as f:
                g = f.read().splitlines()
        r["gate_omp"] = next((l.split("omp=")[1] for l in g if l.startswith("start")), None)
        r["gate_end"] = next((float(l.split()[1]) for l in g if l.startswith("end")), None)
        r["tmp"] = sorted(os.listdir(f"{d}/tmp"))
        r["left"] = left(tok) + left(f"{wt}/tools/gpu_health.py", 0.5)
        return r

    def expect(r, rc, levers=None, text=(), no_text=(), env=None, order=None):
        pr = []
        if r["rc"] != rc:
            pr.append(f"exit status {r['rc']}, expected {rc}")
        if levers is not None and r["levers"] != levers + " ranks=0":   # lever B (tests/test_launch_ranks.py) is off by default
            pr.append(f"launches.txt: levers {r['levers']!r}, expected {levers + ' ranks=0'!r}")
        pr += [f"no /{t}/ in the output" for t in text if not re.search(t, r["out"])]
        pr += [f"/{t}/ in the output" for t in no_text if re.search(t, r["out"])]
        if (env is not None or order) and r["main"] is None:
            pr.append("main.py did not start")
        elif env is not None:
            for k, v in env.items():
                got = r["main"]["env"].get(k)
                if (got is not None) if v is None else (got is None or not re.fullmatch(v, got)):
                    pr.append(f"main.py saw {k}={got!r}, expected {v!r}")
        if order and r["main"] is not None:
            if r["gate_end"] is None:
                pr.append("the gate did not reach its end")
            elif order == "beside" and not r["main"]["t_start"] < r["gate_end"] <= r["main"].get("t_fit", 0):
                pr.append(f"main.py start {r['main']['t_start']:.2f}, gate end {r['gate_end']:.2f}, fit {r['main'].get('t_fit')}: not beside")
            elif order == "front" and not r["gate_end"] <= r["main"]["t_start"]:
                pr.append(f"main.py start {r['main']['t_start']:.2f} before the gate's end {r['gate_end']:.2f}")
        if any(f.startswith("gpu_gate_") for f in r["tmp"]):
            pr.append(f"left in TMPDIR: {r['tmp']}")
        if r["left"]:
            pr.append(f"processes left: {r['left']}")
        return pr + (["output: " + r["out"][-700:].replace("\n", " | ")] if pr else [])

    on = f"gate_bg=1 vec_isa_memo={MEMO} omp=OMP_NUM_THREADS=12"
    off3 = {"GPIC_LAUNCH_GATE_BG": "0", "GPIC_LAUNCH_VEC_ISA_MEMO": "0", "GPIC_LAUNCH_OMP": "0"}
    r = launch("pass", gate="pass:3")
    check("c: the gate beside the start, healthy: fit waits for the verdict",
          expect(r, 0, on, [r"\[gate\] healthy; fit waited \d+\.\d s for the verdict", r"main\.py exit 0"], [r"unhealthy GPU"],
                 {"GPIC_GATE_RC": r".*/pass/tmp/gpu_gate_\d+\.rc", "GPIC_GATE_PID": r"\d+", "OMP_NUM_THREADS": "12",
                  "GPIC_VEC_ISA_MEMO": MEMO}, "beside")
          + ([] if r["gate_omp"] == "None" else [f"the gate itself ran with OMP_NUM_THREADS={r['gate_omp']}"]))
    r = launch("gate_fail", gate="fail:2")
    check("c: the gate beside the start, unhealthy: exit 98, no fit",
          expect(r, 98, on, [r"\[gate\] exit status 1 after", r"UNHEALTHY GPUs 0", r"unhealthy GPU on \S+; exit 98"], [r"main\.py exit"])
          + (["fit ran"] if r["main"] and "t_fit" in r["main"] else []))
    r = launch("gate_crash", gate="crash")
    check("c: the gate ends at once with status 3: exit 98, no fit",
          expect(r, 98, on, [r"\[gate\] exit status 3 after", r"unhealthy GPU on \S+; exit 98"], [r"main\.py exit"])
          + (["fit ran"] if r["main"] and "t_fit" in r["main"] else []))
    r = launch("main_first", gate="hang", main="die:3:1")
    check("c: main.py ends first with status 3: status kept, the gate and its workers stopped",
          expect(r, 3, on, [r"main\.py exit 3"], [r"unhealthy GPU", r"\[gate\]"])
          + ([] if r["wall"] < 30 else [f"the launcher took {r['wall']:.0f} s"]) + ([] if r["gate_end"] is None else ["the gate ran to its end"]))
    r = launch("off", gate="pass:2", env={**off3, "GPIC_VEC_ISA_MEMO": "/x/y"})
    check("c: the three switches off: the gate in front, no lever in main.py's environment",
          expect(r, 0, "gate_bg=0 vec_isa_memo=off omp=off", [r"main\.py exit 0"], [r"\[gate\]", r"unhealthy GPU"],
                 {"GPIC_GATE_RC": None, "GPIC_GATE_PID": None, "OMP_NUM_THREADS": None, "GPIC_VEC_ISA_MEMO": None}, "front"))
    r = launch("off_fail", gate="fail:1", env=off3)
    check("c: the three switches off, unhealthy: exit 98 before main.py",
          expect(r, 98, None, [r"UNHEALTHY GPUs 0", r"unhealthy GPU on \S+; exit 98"], [r"main\.py exit"])
          + ([] if r["main"] is None and r["levers"] is None else ["main.py started"]))
    r = launch("gate_off", gate="pass:2", env={"GPIC_LAUNCH_GATE_BG": "0", "GPIC_VEC_ISA_MEMO": "/x/y"})
    check("c: only the gate's switch off; a memo directory named by the caller stays",
          expect(r, 0, "gate_bg=0 vec_isa_memo=/x/y omp=OMP_NUM_THREADS=12", [r"main\.py exit 0"], [r"\[gate\]"],
                 {"GPIC_GATE_RC": None, "GPIC_GATE_PID": None, "OMP_NUM_THREADS": "12", "GPIC_VEC_ISA_MEMO": "/x/y"}, "front"))
    r = launch("old_main", gate="pass:2", wt=fw_old)
    check("c: a main.py without the hook: the gate in front",
          expect(r, 0, f"gate_bg=0 vec_isa_memo={MEMO} omp=OMP_NUM_THREADS=12", [r"main\.py exit 0"], [r"\[gate\]"],
                 {"GPIC_GATE_RC": None, "GPIC_GATE_PID": None}, "front"))
    r = launch("caller_omp", gate="pass:3", env={"OMP_NUM_THREADS": "5"})
    check("c: a thread count set by the caller stays",
          expect(r, 0, f"gate_bg=1 vec_isa_memo={MEMO} omp=off", [r"main\.py exit 0"], [], {"OMP_NUM_THREADS": "5"}, "beside")
          + ([] if r["gate_omp"] == "5" else [f"the gate ran with OMP_NUM_THREADS={r['gate_omp']}"]))
    for name, kw, want in (("devices_4", {"args": ("--trainer.devices", "4")}, "24"), ("devices_eq_2", {"args": ("--trainer.devices=2",)}, "48"),
                           ("devices_auto", {"args": ("--trainer.devices", "auto")}, None), ("cpus_4", {"nproc": 4}, "1"),
                           ("cpus_100", {"nproc": 100}, "12")):
        r = launch(name, **kw)
        lev = f"gate_bg=1 vec_isa_memo={MEMO} omp=" + (f"OMP_NUM_THREADS={want}" if want else "off")
        check(f"c: thread count, {name}: {want or 'left to Lightning'}", expect(r, 0, lev, [r"main\.py exit 0"], [], {"OMP_NUM_THREADS": want}))
    r = launch("cut_in_gate", gate="pass:25", cut=lambda d: os.path.exists(f"{d}/main.json") and os.path.exists(f"{d}/gate.log"))
    check("c: a cut while fit waits for the gate: exit 143, nothing left",
          expect(r, 143, on, [r"train_ladder\.sh: TERM", r"main\.py exit 143"], [r"unhealthy GPU"])
          + ([] if r["wall"] < 20 else [f"the launcher took {r['wall']:.0f} s"]) + ([] if r["gate_end"] is None else ["the gate ran to its end"]))

    def fit_started(d):
        try:
            with open(f"{d}/main.json") as f:
                return "t_fit" in json.load(f)
        except FileNotFoundError:
            return False
    r = launch("cut_in_fit", gate="pass:1", main="fit:60", cut=fit_started)
    check("c: a cut after the gate: exit 143, nothing left",
          expect(r, 143, on, [r"\[gate\] healthy", r"train_ladder\.sh: TERM", r"main\.py exit 143"], [r"unhealthy GPU"])
          + ([] if r["wall"] < 30 else [f"the launcher took {r['wall']:.0f} s"]))


for part in PARTS:
    {"a": part_a, "b": part_b, "c": part_c}[part]()
bad = [n for n, p in results if p]
print(f"TEST_LAUNCH_LEVERS {'PASS' if not bad else 'FAIL'}: {len(results)} cases" + (f", {len(bad)} failed" if bad else ""), flush=True)
sys.exit(1 if bad else 0)
