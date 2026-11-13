"""CPU test of jobs/train_full_gmflow.sh (docs/gmflow.md section 17) on stubs: no env.sh, no GPU, no training. The script runs against a stub worktree whose
train_ladder.sh reports its arguments and environment; the checks are the command line (the GMFlow overlay before the full ladder's data and logger arguments),
the run root, the shard seed, last.ckpt every C0/8, and the refusals. Run on a vis node: python3 -I tools/test_full_gmflow.py"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "../jobs/train_full_gmflow.sh")
STUB = 'echo "ARGS $*"; echo "ENV LADDER_WT=$LADDER_WT LADDER_RUN_ROOT=$LADDER_RUN_ROOT LADDER_LAST_EVERY=$LADDER_LAST_EVERY"\n'


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main():
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    wt = os.path.join(tmp, "wt")
    for p, text in (("jobs/train_ladder.sh", STUB), ("baselines/PixelGen/src/models/transformer/JiT_T2I.py", "gm_k\n"),
                    ("baselines/PixelGen/configs_h100/gmflow_train.yaml", "model: {}\n")):
        os.makedirs(os.path.dirname(os.path.join(wt, p)), exist_ok=True)
        open(os.path.join(wt, p), "w").write(text)
    git(wt, "init", "-q"); git(wt, "add", "-A"); git(wt, "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "x")
    root = os.path.join(tmp, "root")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LADDER_", "FULL_"))}
    env.update(FULL_WT=wt, FULL_ROOT=root)

    def run(*a, **extra):
        r = subprocess.run(["bash", SCRIPT, *map(str, a)], env={**env, **extra}, capture_output=True, text=True)
        return r.returncode, r.stdout.strip().splitlines()

    cases = []
    rc, out = run(8, "d640", 640, 9, 2, 499200, 1)
    want_args = ("ARGS d640 640 9 2 499200 --config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k 8"
                 " --data.train_dataset.init_args.tar_pattern gpic_train_{00000..07999}.tar"
                 " --trainer.logger.init_args.name ladder_full_gmflow_k8_d640 --trainer.logger.init_args.id ladder-full-gmflow-k8-d640")
    seed = open(os.path.join(root, "exp_ladder_d640/shard_seed.txt")).read().strip()
    cases.append(("d640 1C0: the command line, the root, last.ckpt every C0/8, the point model's shard seed",
                  (rc, out, seed), (0, [want_args, f"ENV LADDER_WT={wt} LADDER_RUN_ROOT={root} LADDER_LAST_EVERY=62400"], "20261106")))
    rc, out = run(8, "d1024", 1024, 12, 3, 155200, 8)
    cases.append(("d1024 8C0: steps 8 x C0", (rc, out[0].split()[:6]), (0, ["ARGS", "d1024", "1024", "12", "3", "1241600"])))
    open(os.path.join(root, "exp_ladder_d640/shard_seed.txt"), "w").write("123\n")
    rc, out = run(8, "d640", 640, 9, 2, 499200, 2)
    cases.append(("a run's shard seed is kept", (rc, open(os.path.join(root, "exp_ladder_d640/shard_seed.txt")).read().strip()), (0, "123")))
    src = open(SCRIPT).read()   # the defaults, read from the source: running without FULL_ROOT would write into the production root
    cases.append(("the defaults: root runs/gpic-speed/full_ladder_gmflow_k<K>, worktree gpic-wt/ladder_full_gmflow_run",
                  ("LADDER_RUN_ROOT=${FULL_ROOT:-$WC/runs/gpic-speed/full_ladder_gmflow_k$K}" in src, "WT=${FULL_WT:-$WC/project/gpic-wt/ladder_full_gmflow_run}" in src),
                  (True, True)))
    for name, args, msg in (("d1280 refused", (8, "d1280", 1280, 14, 3, 800, 1), "runs d640 to d1024"),
                            ("d1792 refused", (8, "d1792", 1792, 18, 4, 800, 1), "runs d640 to d1024"),
                            ("size and width disagree", (8, "d768", 640, 9, 2, 800, 1), "does not match"),
                            ("multiple 3 refused", (8, "d640", 640, 9, 2, 800, 3), "multiple must be"),
                            ("C0 steps not a multiple of 800", (8, "d640", 640, 9, 2, 499328, 1), "multiple of 800"),
                            ("K = 0 refused", (0, "d640", 640, 9, 2, 800, 1), "K must be"),
                            ("K = 100 refused", (100, "d640", 640, 9, 2, 800, 1), "K must be"),
                            ("six arguments refused", (8, "d640", 640, 9, 2, 800), "usage")):
        rc, out = run(*args)
        cases.append((name, (rc, any(msg in line for line in out)), (2, True)))
    open(os.path.join(wt, "jobs/train_ladder.sh"), "a").write("# local edit\n")
    rc, out = run(8, "d640", 640, 9, 2, 800, 1)
    cases.append(("a worktree with local changes refused", (rc, any("local changes" in line for line in out)), (2, True)))
    git(wt, "checkout", "-q", "--", "jobs/train_ladder.sh")
    os.remove(os.path.join(wt, "baselines/PixelGen/configs_h100/gmflow_train.yaml")); git(wt, "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qam", "y")
    rc, out = run(8, "d640", 640, 9, 2, 800, 1)
    cases.append(("a worktree without GMFlow refused", (rc, any("has no GMFlow" in line for line in out)), (2, True)))
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
