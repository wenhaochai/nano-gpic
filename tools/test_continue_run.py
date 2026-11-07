"""CPU test of jobs/continue_run.sh on stubs: which source checkpoint it copies, the step check, the resume, and the training-code guard.
No env.sh, no GPU, no training: the script runs from stub worktrees whose train_ladder.sh only reports the checkpoint it would resume from.
Run on a vis node with the project env (torch writes and reads the tiny checkpoints):
  OMP_NUM_THREADS=6 envs/pixgen/bin/python -I tools/test_continue_run.py [script under test, default jobs/continue_run.sh]"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = "source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh"
STUB = """#!/bin/bash
RUN=$LADDER_RUN_ROOT/exp_ladder_$1
CK=$(ls -t $RUN/last*.ckpt | head -1)
echo "TRAIN $(basename $LADDER_WT) steps $5 from $(basename $CK) step $(python -c 'import sys, torch; print(torch.load(sys.argv[1], weights_only=False)["global_step"])' $CK)"
"""


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def commit(path, msg):
    git(path, "add", "-A"); git(path, "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", msg)
    return git(path, "rev-parse", "HEAD")


def repo(path, files):
    os.makedirs(path)
    for p, text in files.items():
        os.makedirs(os.path.dirname(os.path.join(path, p)), exist_ok=True)
        open(os.path.join(path, p), "w").write(text)
    git(path, "init", "-q")
    return commit(path, "x")


def ckpt(path, step, age=0.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"global_step": step, "state_dict": {}}, path)
    t = time.time() - age
    os.utime(path, (t, t))


def main():
    script = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../jobs/continue_run.sh"))
    src = open(script).read()
    assert src.count(ENV) == 1, "continue_run.sh changed shape"
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    body = src.replace(ENV, f"WC={tmp}")
    gm = os.path.join(tmp, "project/gpic-wt/gmflow_b2")
    repo(gm, {"jobs/continue_run.sh": body, "jobs/train_ladder.sh": STUB, "baselines/model.py": "# model\n"})
    repo(os.path.join(tmp, "project/gpic-wt/ladder_run"), {"jobs/train_ladder.sh": STUB})
    runs = os.path.join(tmp, "runs/gpic-speed")
    # a fresh source run (one launch, last.ckpt), a continued one (the copy it started from as last.ckpt, its own state in last-v1.ckpt), a bad one
    ckpt(f"{runs}/fresh/exp_ladder_d64/last.ckpt", 100)
    open(f"{runs}/fresh/exp_ladder_d64/launches.txt", "w").write("t sha size=d64 max_steps=100 ms_every=25\n")
    ckpt(f"{runs}/cont/exp_ladder_d64/last.ckpt", 100, age=3600)
    ckpt(f"{runs}/cont/exp_ladder_d64/last-v1.ckpt", 200)
    open(f"{runs}/cont/exp_ladder_d64/launches.txt", "w").write("t sha size=d64 max_steps=200 ms_every=50\nt sha size=d64 max_steps=200 ms_every=50\n")
    ckpt(f"{runs}/bad/exp_ladder_d64/last.ckpt", 150)
    open(f"{runs}/bad/exp_ladder_d64/launches.txt", "w").write("t sha size=d64 max_steps=200 ms_every=50\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LADDER_")}
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

    def run(k, source, name, steps):
        r = subprocess.run(["bash", os.path.join(gm, "jobs/continue_run.sh"), str(k), "d64", "64", "2", "1", f"{runs}/{source}/exp_ladder_d64", name, str(steps), name],
                           env=env, capture_output=True, text=True)
        log = open(f"{runs}/{name}/train.log").read().strip().splitlines() if os.path.exists(f"{runs}/{name}/train.log") else []
        return r.returncode, r.stdout.strip().splitlines(), log

    def copies(name):
        p = f"{runs}/{name}/source.txt"
        return open(p).read().strip().splitlines() if os.path.exists(p) else []

    cases = []
    rc, out, log = run(0, "fresh", "p2", 200)
    cases.append(("K = 0, fresh source: copies last.ckpt", (rc, log[-1:], len(copies("p2")), "last.ckpt (" in copies("p2")[0] and "step 100" in copies("p2")[0]),
                  (0, ["TRAIN ladder_run steps 200 from last.ckpt step 100"], 1, True)))
    rc, out, log = run(8, "cont", "g4", 400)
    cases.append(("K = 8, continued source: copies last-v1.ckpt", (rc, log[-1:], "last-v1.ckpt (" in copies("g4")[0] and "step 200" in copies("g4")[0]),
                  (0, ["TRAIN gmflow_b2 steps 400 from last.ckpt step 200"], True)))
    rc, out, log = run(8, "bad", "b4", 400)
    cases.append(("source checkpoint short of its max_steps: refused", (rc, any("holds step 150" in l for l in out), os.path.exists(f"{runs}/b4/exp_ladder_d64/last.ckpt")),
                  (2, True, False)))
    ckpt(f"{runs}/g4/exp_ladder_d64/last-v1.ckpt", 300)   # the continued run saved its own state beside the copy
    rc, out, log = run(8, "cont", "g4", 400)
    cases.append(("resume: no second copy, the newest last", (rc, log[-1:], len(copies("g4"))), (0, ["TRAIN gmflow_b2 steps 400 from last-v1.ckpt step 300"], 1)))
    first = open(f"{runs}/g4/commit.txt").read().strip()
    open(os.path.join(gm, "jobs/continue_run.sh"), "a").write("# a later fix of this script\n")
    os.makedirs(os.path.join(gm, "docs"), exist_ok=True)
    open(os.path.join(gm, "docs/notes.md"), "w").write("x\n")
    commit(gm, "fix the script")
    rc, out, log = run(8, "cont", "g4", 400)
    cases.append(("a later commit with the same training code: resumes, commit.txt kept",
                  (rc, any("has the same training code" in l for l in out), open(f"{runs}/g4/commit.txt").read().strip() == first), (0, True, True)))
    open(os.path.join(gm, "baselines/model.py"), "a").write("# changed\n")
    commit(gm, "change the model")
    rc, out, log = run(8, "cont", "g4", 400)
    cases.append(("a later commit with other training code: refused", (rc, any("other training code" in l for l in out)), (2, True)))
    open(os.path.join(gm, "jobs/continue_run.sh"), "a").write("# a local edit\n")
    rc, out, log = run(8, "cont", "g8", 800)
    cases.append(("local changes in the worktree: refused", (rc, any("local changes" in l for l in out)), (2, True)))
    git(gm, "checkout", "-q", "--", "jobs/continue_run.sh")
    ckpt(f"{runs}/p2/exp_ladder_d64/ms-step=000200.ckpt", 200)
    rc, out, log = run(0, "fresh", "p2", 200)
    cases.append(("final ms checkpoint present: done", (rc, out[-1:]), (0, ["training done before"])))
    ok = True
    for name, got, want in cases:
        good = got == want
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {got}", flush=True)
    shutil.rmtree(tmp)
    print("ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
