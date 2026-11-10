#!/usr/bin/env python3
"""CPU test: the checkpoint writer of AsyncShmCheckpointIO survives a TERM that comes while it starts.

The writer ignores TERM from the first line of _writer, but a spawned process needs seconds until then. In smoke w2514
(2026-11-10) the pilot's cut came 6 s after a launch's first save: the writer ended with -15 with the writes pending, the
cut's own save failed, rank 0 left with 97 and ranks 1-7 hung on their GPUs. A child process in its own process group
saves one checkpoint (the writer starts) and at once sends TERM, to the writer every 50 ms for 4 s and twice to its whole
process group, as the pilot does; it catches the TERM itself, as a rank does under save_on_term. Then it saves a second
checkpoint (the cut's save), waits for both writes and ends.
  new (this code):                 both files complete and equal to what was saved, the child got its TERMs, exit 0
  old (the start without the mask): the writer is gone, the wait for the writes raises (the fault, reproduced)
usage: python test_async_ckpt_term.py <scratch dir>     Result: TEST_ASYNC_CKPT_TERM PASS | FAIL"""
import os
import subprocess
import sys

CHILD = r'''
import os, signal, sys, time, torch
sys.path.insert(0, sys.argv[3])
import src.plugins.async_ckpt_io as A
mode, d = sys.argv[1], sys.argv[2]
got = []
signal.signal(signal.SIGTERM, lambda *a: got.append(time.time()))
if mode == "old":   # the start as it was: no mask around it
    A.signal = type("S", (), {**{k: getattr(signal, k) for k in dir(signal)}, "pthread_sigmask": staticmethod(lambda how, s: set())})
io_ = A.AsyncShmCheckpointIO()
g = torch.Generator().manual_seed(0)
ck = lambda: {"state_dict": {f"w{i}": torch.randn(256, 256, generator=g) for i in range(8)}, "global_step": 256}
io_.save_checkpoint(ck(), d + "/first.ckpt")      # the launch's first save: the writer starts
pid = io_._proc.pid
t = time.time()
os.killpg(os.getpgid(0), signal.SIGTERM)          # the pilot's cut: the whole process group, at once
while time.time() - t < 4:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        break
    time.sleep(0.05)
os.killpg(os.getpgid(0), signal.SIGTERM)
print("writer alive after the TERMs:", io_._proc.is_alive(), "exit code:", io_._proc.exitcode, "TERMs caught by this process:", len(got), flush=True)
io_.save_checkpoint(ck(), d + "/cut.ckpt")        # save_on_term's save
while any(b[1] is not None for b in io_._bufs):
    io_._reap(block=True)
io_.teardown()
print("child done", flush=True)
'''


def main():
    d = sys.argv[1]
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
    import torch
    g = torch.Generator().manual_seed(0)
    ref = [{f"w{i}": torch.randn(256, 256, generator=g) for i in range(8)} for _ in range(2)]
    ok = True
    for mode in ("new", "old"):
        dd = os.path.join(d, mode)
        os.makedirs(dd)
        r = subprocess.run([sys.executable, "-c", CHILD, mode, dd, src], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True,
                           text=True, timeout=180, start_new_session=True)
        files = [os.path.join(dd, f) for f in ("first.ckpt", "cut.ckpt")]
        same = [os.path.exists(f) and all(torch.equal(a, b) for a, b in zip(torch.load(f, weights_only=False)["state_dict"].values(), x.values()))
                for f, x in zip(files, ref)]
        line = next((l for l in r.stdout.splitlines() if l.startswith("writer alive")), "no line")
        caught = int(line.rsplit(":", 1)[1]) if "caught" in line else 0
        if mode == "new":
            good = r.returncode == 0 and all(same) and "alive after the TERMs: True" in line and caught >= 2 and "child done" in r.stdout
        else:
            good = r.returncode != 0 and "alive after the TERMs: False" in line and "exit code: -15" in line and "writes pending" in r.stderr
        left = [f for f in os.listdir(dd) if ".tmp" in f]
        print(f"{mode}: exit {r.returncode}; {line}; files equal {same}; temporary files {left}: {'ok' if good else 'FAIL'}", flush=True)
        if not good:
            print("   stderr: " + r.stderr.strip()[-600:].replace("\n", " | "))
        ok &= good
    print(f"TEST_ASYNC_CKPT_TERM {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
