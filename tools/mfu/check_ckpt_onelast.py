#!/usr/bin/env python3
"""Check of jobs/smoke_ckpt_onelast.sh (2026-11-09): the saves of each launch, read from the hook's own log lines
("[ckpt] step <s> (<n> micro-batches): <file>", src/callbacks/model_checkpoint.py), and the files the launches leave.
Manifest lines (the smoke writes them):
  launch <name> <log> <start step> <max steps> <ms every> <last every> <micro-batches a step>
      the log holds exactly the scheduled saves after <start step>, in order, the ms file before last.ckpt, each at the
      end of an accumulation window (n = s x micro-batches a step); so no save at the step the launch resumed from
  files <name> <run dir> <final step> <ms steps, comma separated>
      the run directory holds last.ckpt, these ms files and no other *.ckpt (no -vN file, no last-vN.ckpt); last.ckpt
      is the full state of <final step> and names itself; the ms file of <final step> holds the model only, with
      last.ckpt's weights bit for bit
  same <name> <listing before> <listing after> kept|exact
      listings of files (name, size, mtime, inode): kept = every file of the first is unchanged in the second (a later
      launch left the earlier ms files alone); exact = the two are equal (a branch wrote nothing into its parent)
  fault <name> <run dir> <step>
      the old code under the same launches: ms-step=<step>-v1.ckpt and last-v1.ckpt exist and last.ckpt is stale at
      <step>. Without the fault the smoke's launches test nothing.
usage: check_ckpt_onelast.py <manifest>; one line per check, then CKPT_ONELAST_CHECK PASS | FAIL <names>"""
import glob, os, re, sys
import torch

LINE = re.compile(r"\[ckpt\] step (\d+) \((\d+) micro-batches\): (\S+\.ckpt)")
FULL = ("optimizer_states", "lr_schedulers", "loops", "callbacks")


def load(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def ckpts(d):
    return sorted(os.path.basename(f) for f in glob.glob(f"{d}/*.ckpt"))


def launch(log, start, steps, ms, last, acc):
    start, steps, ms, last, acc = map(int, (start, steps, ms, last, acc))
    got = [(int(m[1]), int(m[2]), m[3]) for m in LINE.finditer(open(log, errors="replace").read())]
    want = []
    for s in range(start + 1, steps + 1):
        if s % ms == 0:
            want.append((s, s * acc, f"ms-step={s:06d}.ckpt"))
        if s % last == 0:
            want.append((s, s * acc, "last.ckpt"))
    if got == want:
        first = f"first at step {want[0][0]} ({want[0][1]} micro-batches)" if want else "none"
        return True, f"{len(got)} saves, as scheduled after step {start}: {first}, every save at the end of a window of {acc}"
    return False, f"saves differ from the schedule after step {start}: log {got}, schedule {want}"


def files(d, final, ms_steps):
    final = int(final)
    want = sorted(["last.ckpt"] + [f"ms-step={int(s):06d}.ckpt" for s in ms_steps.split(",")])
    have = ckpts(d)
    if have != want:
        return False, f"files {have}, expected {want}"
    ck = load(f"{d}/last.ckpt")
    if ck["global_step"] != final or not all(k in ck for k in FULL):
        return False, f"last.ckpt: step {ck['global_step']}, keys {sorted(ck)}"
    own = {os.path.basename(v["last_model_path"]) for v in ck["callbacks"].values() if isinstance(v, dict) and v.get("last_model_path")}
    if own != {"last.ckpt"}:
        return False, f"last.ckpt's hook state names {sorted(own)}"
    m = load(f"{d}/ms-step={final:06d}.ckpt")
    if m["global_step"] != final or any(k in m for k in FULL):
        return False, f"final ms file: step {m['global_step']}, keys {sorted(m)}"
    a, b = ck["state_dict"], m["state_dict"]
    bad = [k for k in a if k not in b or not torch.equal(a[k], b[k])]
    if a.keys() != b.keys() or bad:
        return False, f"final ms file and last.ckpt differ in {len(bad)} of {len(a)} tensors ({bad[:3]})"
    return True, f"{len(have)} files, no -vN file; last.ckpt = the full state of step {final}; the final ms file holds its {len(a)} tensors bit for bit, model only"


def same(before, after, mode):
    a, b = (open(f).read().split("\n") for f in (before, after))
    a, b = [x for x in a if x], [x for x in b if x]
    if not a:
        return False, "the first listing is empty"
    changed = [x.split()[0] for x in a if x not in b]
    if changed:
        return False, f"changed or gone: {changed}"
    if mode == "exact" and sorted(a) != sorted(b):
        return False, f"new or changed: {[x.split()[0] for x in b if x not in a]}"
    return True, f"{len(a)} files unchanged (size, mtime, inode)" + (f", {len(b) - len(a)} written beside them" if mode == "kept" else ", none added")


def fault(d, step):
    step = int(step)
    have = ckpts(d)
    need = [f"ms-step={step:06d}-v1.ckpt", "last-v1.ckpt", "last.ckpt"]
    if not all(f in have for f in need):
        return False, f"no fault with the old code: files {have}"
    stale, live = load(f"{d}/last.ckpt")["global_step"], load(f"{d}/last-v1.ckpt")["global_step"]
    if stale != step or live <= step:
        return False, f"last.ckpt at step {stale}, last-v1.ckpt at step {live}"
    size = sum(os.path.getsize(f"{d}/{f}") for f in need[:1] + ["last.ckpt"]) / 2**30
    return True, f"the fault: {need[0]} (a second copy of step {step}), last.ckpt stale at step {stale} beside last-v1.ckpt at step {live}; {size:.1f} GiB of waste"


CHECKS = {"launch": launch, "files": files, "same": same, "fault": fault}
failed = []
for line in open(sys.argv[1]):
    kind, name, *args = line.split()
    try:
        ok, text = CHECKS[kind](*args)
    except Exception as e:   # a missing file or key is a failed check, with the others still reported
        ok, text = False, f"{type(e).__name__}: {e}"
    print(f"{'ok  ' if ok else 'FAIL'} {kind} {name}: {text}", flush=True)
    if not ok:
        failed.append(name)
print("CKPT_ONELAST_CHECK " + ("PASS" if not failed else "FAIL " + " ".join(failed)))
sys.exit(1 if failed else 0)
