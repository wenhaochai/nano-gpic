#!/usr/bin/env python3
"""Test of tools/mfu/check_launch.py on made-up launches (no torch, no GPU): a manifest of arms that agree must PASS, and
each single fault that the checker is there to find must turn the verdict to FAIL with its own reason; a difference
that the base arms show among themselves in a soft field is a note, not a failure.
usage: python test_check_launch.py <scratch dir>"""
import contextlib
import copy
import io
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import check_launch  # noqa: E402

HARD = {"den": "d0", "den_buf": "b0", "ema": "e0", "ema_buf": "eb0", "opt": "o0", "sched": "s0", "isa": "VecAMX,VecAVX512,VecAVX2"}
SOFT = {"rng_cpu": "c", "rng_cuda": "g", "rng_np": "n", "rng_py": "p"}


def arm(name, group, lever, memo, step=0, run=None, n=1, first=100.0, step_ms=70.0, rk=None, role=None, gain=None, ema=None):
    """One launch as a dict that write() turns into the files the checker reads; every field can be changed after."""
    on = lever == "on"
    ranks = {}
    for r in range(8):
        f = {"step": str(step)}
        f.update({k: f"{v}r{r}/9" for k, v in HARD.items() if k != "isa"})
        f.update({k: f"{v}{r}" for k, v in SOFT.items()})
        f["isa"] = HARD["isa"]
        f["memo"] = {"off": "h0/m0/s0/e0", "miss": "h0/m4/s4/e0", "hit": "h4/m0/s0/e0"}[memo]
        f["threads"], f["omp"], f["cpus"] = "12/96", "12", "48"   # at the first step: Lightning's count in every arm
        f["thr0"], f["omp0"] = ("12", "12") if on else ("96", "unset")   # before fit: the lever's count, or every CPU's
        f.update({"t_proc": f"{(0.5 if on else 15.0) + (20.0 if r and rk != '1' else 0.01 * r):.2f}", "t_init": "9.0", "t_setup": "30.0", "t_fit": "31.0", "t_train": f"{first - 12:.2f}", "copy_s": "none" if ema == "skip" else "4.20", "copy_flt": "none" if ema == "skip" else "1000", "fp_s": "0.4"})
        ranks[r] = {"start": f, "batch": [(step + i, f"h{i}r{r}/7", first - 1 + 0.3 * r + i) for i in range(2)],
                    "rows": [{"step": step + i, "step_ms": 300.0 if i < 16 else step_ms, "data_wait_s": 0.001, "loss": 1.0 - 1e-3 * i + r}
                             for i in range(64)]}
    return {"name": name, "group": group, "lever": lever, "memo": memo, "step": step, "run": run or name, "n": n, "rc": 0,
            "main_rc": 0, "ranks": ranks, "gate": [380, 381, 379, 382, 380, 381, 380, 379], "gate_ok": True,
            "gate_line": on, "extra": "".join(f" {k}={v}" for k, v in (("ranks", rk), ("role", role), ("gain", gain), ("ema_copy", ema)) if v is not None),
            "levers": ("gate_bg=1 vec_isa_memo=/x/memo omp=OMP_NUM_THREADS=12" if on else "gate_bg=0 vec_isa_memo=off omp=off")
                      + ("" if rk is None else f" ranks={rk}") + ("" if ema is None else f" skip_ema_copy={int(ema == 'skip')}")}


def write(root, arms):
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    man = []
    for a in sorted(arms, key=lambda a: (a["run"], a["n"])):   # the launches of one run directory in their order
        run, bench, log = os.path.join(root, "runs", a["run"]), os.path.join(root, a["name"] + ".bench"), os.path.join(root, a["name"] + ".log")
        os.makedirs(run, exist_ok=True)
        os.makedirs(bench)
        with open(os.path.join(run, "launches.txt"), "a") as f:
            f.write("2026-11-09T00:00:00Z abc size=d640\n2026-11-09T00:00:00Z MFU levers: attn=fa3\n")
            f.write(f"2026-11-09T00:00:00Z launch levers: {a['levers']}\n")
        with open(os.path.join(run, f"gpu_health_{1000 + a['n']}.txt"), "w") as f:
            f.writelines(f"GPU {i}: {v} TFLOP/s, 1965 MHz, 55 C, reasons 0x4 \n" for i, v in enumerate(a["gate"]))
            f.write(f"node: median 380 TFLOP/s; {'healthy' if a['gate_ok'] else 'UNHEALTHY GPUs 3'}\n")
        out = ["[env] job= host=x\n"]
        if a["gate_line"]:
            out.append("[gate] healthy; fit waited 0.0 s for the verdict\n")
        out += [f"[warmup] T={t}: 5.0 s\n" for t in (64, 128, 320, 192)]
        for r, d in a["ranks"].items():   # a progress bar's text before the marker, as in a shared log
            out.append("Epoch 0:   0%|   | 0/384 [00:00<?, ?it/s]\r" + f"[launch-fp] rank {r} start " + " ".join(f"{k}={v}" for k, v in d["start"].items()) + "\n")
            out += [f"[launch-fp] rank {r} batch {i} step={s} hash={h} t={t:.2f}\n" for i, (s, h, t) in enumerate(d["batch"])]
            with open(os.path.join(bench, f"steps_rank{r}.json"), "w") as f:
                json.dump({"rank": r, "rows": d["rows"]}, f)
        out.append(f"main.py exit {a['main_rc']}\n")
        with open(log, "w") as f:
            f.writelines(out)
        man.append(f"arm name={a['name']} group={a['group']} lever={a['lever']} memo={a['memo']} step={a['step']} log={log} run={run} "
                   f"n={a['n']} bench={bench} rc={a['rc']} wall=100{a['extra']}\n")
    path = os.path.join(root, "manifest.txt")
    with open(path, "w") as f:
        f.writelines(man)
    return path


def base():
    return {a["name"]: a for a in (
        arm("d640_off1", "d640", "off", "off", first=110.0), arm("d640_miss", "d640", "on", "miss", first=95.0),
        arm("d640_hit", "d640", "on", "hit", first=80.0), arm("d640_off2", "d640", "off", "off", first=104.0),
        arm("r_off1", "res", "off", "off", step=128, run="parent", n=2, first=120.0),
        arm("r_hit", "res", "on", "hit", step=128, run="parent", n=3, first=90.0),
        arm("r_off2", "res", "off", "off", step=128, run="parent", n=4, first=118.0),
        arm("parent", "fresh", "off", "off", run="parent", n=1, first=130.0), arm("fresh_hit", "fresh", "on", "hit", first=90.0),
        arm("fresh_off2", "fresh", "off", "off", first=128.0))}


def each(a, fn, ranks=range(8)):
    for r in ranks:
        fn(a["ranks"][r])


CASES = []   # (name, change(arms), PASS?, a text the output must hold, the number of failures, the number of notes)


def case(name, ok, text, nfail=None, nnote=None):
    def deco(fn):
        CASES.append((name, fn, ok, text, (0 if ok else 1) if nfail is None else nfail,
                      (1 if "note: " in text else 0) if nnote is None else nnote))
        return fn
    return deco


@case("all arms agree", True, "CHECK_LAUNCH PASS: 10 arms in 3 groups, 0 failures, 0 notes")
def _(a): pass


@case("on arm: other weights on rank 3", False, "d640_hit: den differs from the base arms (rank 3)")
def _(a): a["d640_hit"]["ranks"][3]["start"]["den"] = "other"


@case("on arm: other EMA slot on every rank", False, "d640_miss: ema differs from the base arms (all ranks)")
def _(a): each(a["d640_miss"], lambda d: d["start"].update(ema="other"))


@case("off arms differ in the optimizer state", False, "d640: the base arms differ in opt (rank 0): the launcher itself does not repeat")
def _(a): a["d640_off2"]["ranks"][0]["start"]["opt"] = "other"


@case("off arms differ in a random stream: a note", True, "note: d640: rng_cpu differs between the base arms (rank 2): not testable")
def _(a): a["d640_off1"]["ranks"][2]["start"]["rng_cpu"] = "other"


@case("on arm: other GPU random stream", False, "r_hit: rng_cuda differs from the base arms (rank 7)")
def _(a): a["r_hit"]["ranks"][7]["start"]["rng_cuda"] = "other"


@case("on arm: other first batch", False, "d640_hit: batch0 differs from the base arms (rank 1)")
def _(a): a["d640_hit"]["ranks"][1]["batch"][0] = (0, "otherhash/7", 80.0)


@case("on arm: other second loss", False, "d640_hit: loss1 differs from the base arms (rank 4)")
def _(a): a["d640_hit"]["ranks"][4]["rows"][1]["loss"] = 0.5


@case("off arms differ in the first loss: a note", True, "note: d640: loss0 differs between the base arms (all ranks): not testable")
def _(a): each(a["d640_off2"], lambda d: d["rows"][0].update(loss=0.25))


@case("on arm: other vector sets", False, "fresh_hit: isa differs from the base arms (all ranks)")
def _(a): each(a["fresh_hit"], lambda d: d["start"].update(isa="VecAVX2"))


@case("hit arm only 2 s sooner", False, "d640_hit: first step after 103.1 s, the base arms after 105.1-111.1 s")
def _(a): each(a["d640_hit"], lambda d: d.update(batch=[(s, h, t + 22.0) for s, h, t in d["batch"]]))


@case("miss arm later than the base arms", False, "d640_miss: first step after 116.1 s, later than the base arms (111.1 s)")
def _(a): each(a["d640_miss"], lambda d: d.update(batch=[(s, h, t + 20.0) for s, h, t in d["batch"]]))


@case("on arm: steps 5% slower", False, "d640_hit: step 73.50 ms, the base arms 70.00-70.00 ms")
def _(a): each(a["d640_hit"], lambda d: [row.update(step_ms=73.5) for row in d["rows"][16:]])


@case("on arm: steps 1.5% slower passes", True, "CHECK_LAUNCH PASS")
def _(a): each(a["d640_hit"], lambda d: [row.update(step_ms=71.05) for row in d["rows"][16:]])


@case("on arm: rank 2 started with every CPU's thread", False,
      "d640_hit: not the launcher's 12 threads before fit on rank 2: thr0=96 omp0=unset")
def _(a): a["d640_hit"]["ranks"][2]["start"].update(thr0="96", omp0="unset")


@case("on arm: the variable was set before fit, the thread count was not", False,
      "r_hit: not the launcher's 12 threads before fit on all ranks: thr0=96 omp0=12")
def _(a): each(a["r_hit"], lambda d: d["start"].update(thr0="96"))


@case("on arm: the launcher gave another count than the ranks had", False,
      "fresh_hit: not the launcher's 6 threads before fit on all ranks: thr0=12 omp0=12")
def _(a): a["fresh_hit"]["levers"] = a["fresh_hit"]["levers"].replace("OMP_NUM_THREADS=12", "OMP_NUM_THREADS=6")


@case("off arm: OMP_NUM_THREADS was set before fit", False, "d640_off1: OMP_NUM_THREADS=12 before fit without the lever (all ranks)")
def _(a): each(a["d640_off1"], lambda d: d["start"].update(thr0="12", omp0="12"))


@case("on arm: another thread count at the first step", False, "d640_hit: thr differs from the base arms (rank 2): 6 against 12")
def _(a): a["d640_hit"]["ranks"][2]["start"].update(threads="6/96")


@case("on arm: another OMP_NUM_THREADS at the first step", False, "d640_miss: omp differs from the base arms (all ranks): 6 against 12")
def _(a): each(a["d640_miss"], lambda d: d["start"].update(omp="6"))


@case("off arms differ in the thread count at the first step: a note", True,
      "note: d640: thr differs between the base arms (rank 3): not testable")
def _(a): a["d640_off1"]["ranks"][3]["start"].update(threads="96/96")


@case("on arm: another inter-op count is no fault", True, "CHECK_LAUNCH PASS: 10 arms in 3 groups, 0 failures, 0 notes")
def _(a): each(a["d640_hit"], lambda d: d["start"].update(threads="12/12"))


@case("on arm: a loss differs in the last digits: a note", True,
      "note: d640_hit: loss1 agrees with the base arms within 0.001 but not to the last digit (rank 4): 4.999004999 against 4.999")
def _(a): a["d640_hit"]["ranks"][4]["rows"][1]["loss"] = 4.999004999


@case("on arm: a loss differs by 0.5%", False, "d640_hit: loss0 differs from the base arms (rank 6): 7.035 against 7.0")
def _(a): a["d640_hit"]["ranks"][6]["rows"][0]["loss"] = 7.035


@case("off arms differ in a loss in the last digits: each on arm gets a note", True,
      "note: d640_miss: loss0 agrees with the base arms within 0.001 but not to the last digit (rank 1)", nnote=2)
def _(a): a["d640_off2"]["ranks"][1]["rows"][0]["loss"] = 2.000001


@case("hit arm stored new answers", False, "d640_hit: memo counts of a launch after the first: [(0, 4, 4, 0)]")
def _(a): each(a["d640_hit"], lambda d: d["start"].update(memo="h0/m4/s4/e0"))


@case("hit arm with one False answer: a note", True, "note: d640_hit: 1 vector-set checks without a stored answer are asked at every launch")
def _(a): each(a["d640_hit"], lambda d: d["start"].update(memo="h3/m1/s0/e0"))


@case("miss arm with an error", False, "d640_miss: memo counts of the first launch with the memo")
def _(a): a["d640_miss"]["ranks"][6]["start"]["memo"] = "h0/m4/s3/e1"


@case("miss arm stored nothing", False, "d640_miss: memo counts of the first launch with the memo: [(0, 4, 0, 0)]")
def _(a): each(a["d640_miss"], lambda d: d["start"].update(memo="h0/m4/s0/e0"))


@case("off arm used the memo", False, "d640_off2: memo counts without the memo")
def _(a): a["d640_off2"]["ranks"][0]["start"]["memo"] = "h4/m0/s0/e0"


@case("off arm printed the gate's verdict line", False, "d640_off1: 1 gate verdict lines for lever off")
def _(a): a["d640_off1"]["gate_line"] = True


@case("on arm printed no verdict line", False, "r_hit: 0 gate verdict lines for lever on")
def _(a): a["r_hit"]["gate_line"] = False


@case("on arm: launches.txt says the gate blocked", False, "d640_hit: launches.txt has the levers")
def _(a): a["d640_hit"]["levers"] = a["d640_hit"]["levers"].replace("gate_bg=1", "gate_bg=0")


@case("on arm: launches.txt says no thread count", False, "fresh_hit: launches.txt has the levers")
def _(a): a["fresh_hit"]["levers"] = a["fresh_hit"]["levers"].replace("omp=OMP_NUM_THREADS=12", "omp=off")


@case("on arm: one GPU 10% under the gate's median", False, "d640_hit: the gate's slowest GPU is 342 of a median 380 TFLOP/s")
def _(a): a["d640_hit"]["gate"][5] = 342


@case("on arm: the gate's median 5% under the base arms'", False, "d640_miss: gate median 361 TFLOP/s, the base arms 380")
def _(a): a["d640_miss"]["gate"] = [361] * 8


@case("gate report unhealthy", False, "d640_off1: no healthy gate report of 8 GPUs")
def _(a): a["d640_off1"]["gate_ok"] = False


@case("arm ended with exit 1", False, "d640_off2: exit status 1, main.py ['0']")
def _(a): a["d640_off2"]["rc"] = 1


@case("main.py ended with 98", False, "d640_hit: exit status 0, main.py ['98']")
def _(a): a["d640_hit"]["main_rc"] = 98


@case("rank 5 printed no start line", False, "d640_hit: 7 start and 16 batch lines, 0 repeated (need 8 and 16)")
def _(a): del a["d640_hit"]["ranks"][5]["start"]; a["d640_hit"]["ranks"][5]["start"] = None


@case("resumed arm started at step 0", False, "r_hit: starts at step 0, not 128 (all ranks)", nfail=2)   # and: step differs from the base arms
def _(a): each(a["r_hit"], lambda d: d["start"].update(step="0"))


@case("resumed arm: first batch at the wrong step", False, "r_off1: the first two batches are not steps 128 and 129",
      nnote=1)   # and the note: batch0 differs between the base arms
def _(a): a["r_off1"]["ranks"][0]["batch"][0] = (127, "h0r0/7", 119.0)


@case("a group with one off arm", False, "fresh: needs two base arms and one test arm")
def _(a): del a["fresh_off2"]


@case("step timer file missing", False, "d640_off1: the step timer's files are missing or too short",
      nnote=2)   # and the notes: the two losses of that rank are not testable
def _(a): a["d640_off1"]["ranks"][3]["rows"] = None


def with_b(a, first=70.0, gain=None, base2=82.0, ema=None):
    """The second smoke's layout as one more group: the first three levers on in every arm, lever B (and, with ema="skip",
    lever G1) in the test arm."""
    a["b_base1"] = arm("b_base1", "b640", "on", "miss", first=95.0, rk="0", role="base", ema=ema and "keep")
    a["b_test"] = arm("b_test", "b640", "on", "hit", first=first, rk="1", role="test", gain=gain, ema=ema)
    a["b_base2"] = arm("b_base2", "b640", "on", "hit", first=base2, rk="0", role="base", ema=ema and "keep")


@case("lever B: base arms with the three levers on, a test arm with every rank started at once", True, "CHECK_LAUNCH PASS: 13 arms in 4 groups, 0 failures, 0 notes")
def _(a): with_b(a)


@case("lever B: the test arm's launches.txt says ranks=0", False, "b_test: launches.txt has ranks=0, the manifest 1")
def _(a): with_b(a); a["b_test"]["levers"] = a["b_test"]["levers"].replace("ranks=1", "ranks=0")


@case("lever B: launches.txt of an older launcher (no ranks field) for the test arm", False, "b_test: launches.txt has ranks=0, the manifest 1")
def _(a): with_b(a); a["b_test"]["levers"] = a["b_test"]["levers"].replace(" ranks=1", "")


@case("lever B: the test arm's ranks 1-7 started 20 s after rank 0", False, "b_test: the 8 rank processes started 20.0 s apart with ranks=1")
def _(a): with_b(a); each(a["b_test"], lambda d: d["start"].update(t_proc="20.5"), range(1, 8))


@case("lever B: a base arm's ranks started together", False, "b_base2: the 8 rank processes started 0.0 s apart with ranks=0")
def _(a): with_b(a); each(a["b_base2"], lambda d: d["start"].update(t_proc="0.5"))


@case("lever B: the test arm differs in the weights", False, "b_test: den differs from the base arms (rank 6)")
def _(a): with_b(a); a["b_test"]["ranks"][6]["start"]["den"] = "other"


@case("lever B: only 4 s sooner than the fastest base arm", False, "b_test: first step after 79.1 s, the base arms after 83.1-96.1 s")
def _(a): with_b(a, first=78.0)


@case("lever B: 4 s sooner with gain=3 in the manifest", True, "CHECK_LAUNCH PASS: 13 arms in 4 groups, 0 failures, 0 notes")
def _(a): with_b(a, first=78.0, gain="3")


@case("lever B: 2 s sooner with gain=3 in the manifest", False, "b_test: first step after 81.1 s, the base arms after 83.1-96.1 s")
def _(a): with_b(a, first=80.0, gain="3")


@case("lever B: the test arm's steps 3% slower", False, "b_test: step 72.10 ms, the base arms 70.00-70.00 ms")
def _(a): with_b(a); each(a["b_test"], lambda d: [x.update(step_ms=72.1) for x in d["rows"][16:]])


@case("lever B: a group with one base arm", False, "b640: needs two base arms and one test arm")
def _(a): with_b(a); del a["b_base1"]


@case("lever G1: the test arm leaves out the EMA slot's second copy", True, "CHECK_LAUNCH PASS: 13 arms in 4 groups, 0 failures, 0 notes")
def _(a): with_b(a, ema="skip")


@case("lever G1: asked for, but rank 2 made the copy", False, "b_test: the EMA slot's second copy on rank 2: copy_s=4.20 with ema_copy=skip")
def _(a): with_b(a, ema="skip"); a["b_test"]["ranks"][2]["start"]["copy_s"] = "4.20"


@case("lever G1: a base arm made no copy", False, "b_base2: the EMA slot's second copy on all ranks: copy_s=none with ema_copy=keep")
def _(a): with_b(a, ema="skip"); each(a["b_base2"], lambda d: d["start"].update(copy_s="none"))


@case("lever G1: launches.txt does not say it", False, "b_test: launches.txt has skip_ema_copy=0, the manifest ema_copy=skip")
def _(a): with_b(a, ema="skip"); a["b_test"]["levers"] = a["b_test"]["levers"].replace("skip_ema_copy=1", "skip_ema_copy=0")


@case("lever G1: the test arm's EMA slot differs", False, "b_test: ema differs from the base arms (all ranks)")
def _(a): with_b(a, ema="skip"); each(a["b_test"], lambda d: d["start"].update(ema="other"))


def main(scratch):
    ok = True
    for i, (name, change, want_pass, text, nfail, nnote) in enumerate(CASES):
        arms = copy.deepcopy(base())
        change(arms)
        root = os.path.join(scratch, f"case{i:02d}")
        for a in arms.values():   # the two cases that drop a file
            for r, d in a["ranks"].items():
                if d["start"] is None:
                    d["start"], d["drop_start"] = {}, True
        man = write_case(root, arms)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = check_launch.main(man)
        out = buf.getvalue()
        nf = sum(l.startswith("FAIL: ") for l in out.splitlines())   # one fault, its own reasons: nothing else may fail with it
        nn = sum(l.startswith("note: ") for l in out.splitlines())
        good = (rc == 0) == want_pass and text in out and ("CHECK_LAUNCH PASS" in out) == want_pass and nf == nfail and nn == nnote
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: exit {rc}, {nf} failures, {nn} notes")
        if not good:
            print("     wanted:", "PASS" if want_pass else "FAIL", f"with {nfail} failures, {nnote} notes and:", text)
            print("     " + "\n     ".join(l for l in out.splitlines() if l.startswith(("FAIL", "note", "CHECK"))))
        else:
            shutil.rmtree(root)
    print(f"TEST_CHECK_LAUNCH {'PASS' if ok else 'FAIL'}: {len(CASES)} cases")
    return 0 if ok else 1


def write_case(root, arms):
    """write(), then the faults that are a missing line or file."""
    man = write(root, list(arms.values()))
    for a in arms.values():
        log = os.path.join(root, a["name"] + ".log")
        for r, d in a["ranks"].items():
            if d.get("drop_start"):
                lines = [l for l in open(log) if f"[launch-fp] rank {r} start" not in l]
                with open(log, "w") as f:
                    f.writelines(lines)
            if d["rows"] is None:
                os.remove(os.path.join(root, a["name"] + ".bench", f"steps_rank{r}.json"))
    return man


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
