"""Runs a Python file of this worktree under `python -I` (no PYTHONPATH, no user site): puts baselines/PixelGen and tools/ on sys.path,
changes into baselines/PixelGen (the tests read configs and run git from there) and runs the file as __main__.
usage (vis node): CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 nice -n 19 <pixgen python> -I tools/gmflow_ladder/run.py [--pix <dir>] <file relative to the repo> [args...]
--pix: another baselines/PixelGen (test (d) runs the harness on an export of 5ac69ab); the file still comes from this worktree."""
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIX = os.path.join(REPO, "baselines", "PixelGen")
if sys.argv[1] == "--pix":
    PIX = os.path.abspath(sys.argv[2])
    del sys.argv[1:3]
for p in (os.path.join(REPO, "tools"), PIX):
    sys.path.insert(0, p)
os.environ["PYTHONPATH"] = PIX   # for the subprocesses some tests start
target = os.path.join(REPO, sys.argv[1])
sys.argv = [target] + sys.argv[2:]
os.chdir(PIX)
runpy.run_path(target, run_name="__main__")
