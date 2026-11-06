"""CPU test of the per-width code choice in jobs/train_ladder.sh (docs/ladder_plan.md section 12.9), on stubs: no env.sh, no GPU, no training.
The script's head (everything before the step check) runs from a stub copy of this branch; the rest is replaced by a line that names this code, and
the MFU worktree's train_ladder.sh is a stub that names itself. Run on a vis node: python3 -I tools/test_dispatch.py"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ENV = "source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh"
CUT = "if [ $((STEPS % 32)) -ne 0 ]"


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def repo(path, files):
    os.makedirs(path)
    for p, text in files.items():
        os.makedirs(os.path.dirname(os.path.join(path, p)), exist_ok=True)
        open(os.path.join(path, p), "w").write(text)
    git(path, "init", "-q"); git(path, "add", "-A"); git(path, "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "x")
    return git(path, "rev-parse", "HEAD")


def main():
    src = open(os.path.join(ROOT, "jobs/train_ladder.sh")).read()
    assert src.count(ENV) == 1 and src.count(CUT) == 1, "train_ladder.sh changed shape"
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    head = src.split(CUT)[0].replace(ENV, f"WC={tmp}")
    this = os.path.join(tmp, "project/gpic-wt/this")
    this_sha = repo(this, {"jobs/train_ladder.sh": head + 'echo "THIS CODE d$D LADDER_WT=${LADDER_WT:-}"\n'})
    mfu = os.path.join(tmp, "project/gpic-wt/ladder_full_mfu")
    mfu_sha = repo(mfu, {"jobs/train_ladder.sh": 'echo "MFU CODE d$2 steps $5 args $6 $7 LADDER_WT=$LADDER_WT"\n'})
    runs = os.path.join(tmp, "runs")
    env0 = {k: v for k, v in os.environ.items() if not k.startswith("LADDER_")}

    def run(d, first=None, **extra):
        size = f"d{d}"
        shutil.rmtree(runs, ignore_errors=True); os.makedirs(os.path.join(runs, f"exp_ladder_{size}"))
        if first:
            open(os.path.join(runs, f"exp_ladder_{size}/launches.txt"), "w").write(f"2026-11-06T00:00:00Z {first} size={size}\n")
        env = {**env0, "LADDER_WT": this, "LADDER_RUN_ROOT": runs, **extra}
        r = subprocess.run(["bash", os.path.join(this, "jobs/train_ladder.sh"), size, str(d), "14", "3", "800", "--a", "b"],
                           env=env, capture_output=True, text=True)
        return r.returncode, (r.stdout.strip().splitlines() or [r.stderr.strip()])[-1]

    ok = True
    cases = [
        ("d1024, new run: this code", run(1024), (0, f"THIS CODE d1024 LADDER_WT={this}")),
        ("d1280, new run: the MFU worktree", run(1280), (0, f"MFU CODE d1280 steps 800 args --a b LADDER_WT={mfu}")),
        ("d1792, first launch on the MFU commit: the MFU worktree again", run(1792, mfu_sha), (0, f"MFU CODE d1792 steps 800 args --a b LADDER_WT={mfu}")),
        ("d1536, first launch on this branch: this code", run(1536, this_sha), (0, f"THIS CODE d1536 LADDER_WT={this}")),
        ("d640, first launch on this branch: this code", run(640, this_sha), (0, f"THIS CODE d640 LADDER_WT={this}")),
    ]
    rc, out = run(1280, "0" * 40)
    cases.append(("d1280, first launch on another commit: refused", (rc, "neither this branch" in out and "refused"), (2, "refused")))
    rc, out = run(1280, LADDER_MFU_WT=os.path.join(tmp, "missing"))
    cases.append(("d1280, no MFU worktree: refused", (rc, "no worktree" in out and "refused"), (2, "refused")))
    open(os.path.join(mfu, "jobs/train_ladder.sh"), "a").write("# local edit\n")
    rc, out = run(1792)
    cases.append(("d1792, MFU worktree with local changes: refused", (rc, "local changes" in out and "refused"), (2, "refused")))
    git(mfu, "checkout", "-q", "--", "jobs/train_ladder.sh")
    cases.append(("LADDER_WT is the MFU worktree: no second dispatch", run(1280, LADDER_WT=mfu), (0, f"THIS CODE d1280 LADDER_WT={mfu}")))
    for name, got, want in cases:
        good = got == want
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {got}", flush=True)
    shutil.rmtree(tmp)
    print("ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
