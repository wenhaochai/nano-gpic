#!/usr/bin/env python3
"""CPU test: AsyncShmCheckpointIO (MFU lever F) finishes its writes at interpreter exit when nobody calls teardown, the
case of Lightning 2.6.6's ModelParallelStrategy (lever B). A child process saves one checkpoint whose write takes 3 s
(GPIC_ASYNC_CKPT_TEST_DELAY) and exits at once without teardown:
  old: the exit wait unregistered (the code before the fix): the file must be missing (the race, reproduced),
  new: the file must be complete, loadable and equal to the saved dict, exit code 0,
  dead: the writer is killed with the write pending: exit code 97 within the wait's timeout (no hang, no silent exit 0).
usage: python test_async_ckpt_exit.py <scratch dir>"""
import os, subprocess, sys, time

CHILD = r'''
import atexit, os, sys, torch
sys.path.insert(0, sys.argv[3])
from src.plugins.async_ckpt_io import AsyncShmCheckpointIO
mode, path = sys.argv[1], sys.argv[2]
io_ = AsyncShmCheckpointIO()
g = torch.Generator().manual_seed(0)
ck = {"state_dict": {f"w{i}": torch.randn(256, 256, generator=g) for i in range(8)}, "global_step": 4800}
io_.save_checkpoint(ck, path)
if mode == "old":
    atexit.unregister(io_._teardown_at_exit)
if mode == "dead":
    io_._proc.kill()
print("child exits", flush=True)
'''


def main():
    d = sys.argv[1]
    os.makedirs(d, exist_ok=True)
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
    env = dict(os.environ, GPIC_ASYNC_CKPT_TEST_DELAY="3", CUDA_VISIBLE_DEVICES="")
    import torch
    g = torch.Generator().manual_seed(0)
    ref = {f"w{i}": torch.randn(256, 256, generator=g) for i in range(8)}
    ok = True
    for mode in ("old", "new", "dead"):
        path = os.path.join(d, f"{mode}.ckpt")
        for f in os.listdir(d):
            if f.startswith(mode):
                os.remove(os.path.join(d, f))
        t = time.time()
        r = subprocess.run([sys.executable, "-c", CHILD, mode, path, src], env=env, capture_output=True, text=True, timeout=120)
        dt = time.time() - t
        exists = os.path.exists(path)
        if mode == "old":
            good = r.returncode == 0 and not exists
            what = f"file {'present' if exists else 'missing'} (the race {'reproduced' if not exists else 'NOT reproduced'})"
        elif mode == "new":
            same = exists and all(torch.equal(a, b) for a, b in zip(torch.load(path, weights_only=False)["state_dict"].values(), ref.values()))
            good = r.returncode == 0 and same and dt >= 3
            what = f"file {'complete and equal' if same else 'MISSING or different'}"
        else:
            good = r.returncode == 97 and not exists
            what = f"file {'present' if exists else 'missing'}, stderr: {r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else ''}"
        left = [f for f in os.listdir(d) if f.startswith(mode) and ".tmp" in f]
        print(f"{mode:4s}: exit {r.returncode} after {dt:.1f} s, {what}, temporary files {left}: {'ok' if good else 'FAIL'}")
        ok &= good
    print("ok" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
