#!/usr/bin/env python3
"""CPU checks of jobs/smoke_ckpt_dedup.sh on its two runs <out>/d9 (width 640) and <out>/d16 (width 1536), trained with
the twins layout (GPIC_MFU_NO_EMA: the EMA slot's copy of the raw weights is written once):
  files   every *.ckpt loads; an ms file's global_step is its name's step; every non-empty ema_denoiser.X shares its
          memory with denoiser.X; unique storage bytes / logical bytes (1/2 for an ms file, about 2/3 for a last file);
          the file's size against the old layout's; the final ms file's state_dict equals last.ckpt's bit for bit
  resume  launch 2 redoes 0 steps; the mean loss of the 40 steps before the cut and after the resume; no non-finite loss
  stall   the step that holds a scheduled save (bench row step = save step - 1) minus the launch's median step, against
          the old layout's smoke <ref> (information, no criterion: it depends on the node and on the file system)
Prints one line per check and CKPT_DEDUP_CHECK PASS|FAIL; exit 1 on FAIL.
usage: check_ckpt_dedup.py <out> [<ref smoke dir>]"""
import glob, json, math, os, re, statistics, sys
import torch

RUNS = {"d9": ("d640", 4096, 512), "d16": ("d1536", 1024, 128)}          # tag: size, steps, last every
OLD = {"d640": dict(ms=877451079, last=1327596107), "d1536": dict(ms=8979589163, last=13501660404)}   # bytes, old layout
BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("ok    " if cond else "FAIL  ") + msg, flush=True)


def tensors(o):
    if isinstance(o, torch.Tensor):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from tensors(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from tensors(v)


def bits_equal(a, b):
    if (a.dtype, a.shape) != (b.dtype, b.shape):
        return False
    if a.numel() == 0:
        return True
    i = BITS[a.element_size()]
    return torch.equal(a.contiguous().view(i), b.contiguous().view(i))


def files(tag):
    size, S, last_every = RUNS[tag]
    run = f"{out}/{tag}/exp_ladder_{size}"
    names = sorted(os.path.basename(f) for f in glob.glob(f"{run}/*.ckpt"))
    want = [f"ms-step={S // 4 * k:06d}.ckpt" for k in (1, 2, 3, 4)] + ["last.ckpt"]
    check(all(w in names for w in want) and not [n for n in names if re.search(r"-v\d+\.ckpt$", n)], f"{tag}: files {names}")
    sds = {}
    for n in names:
        ck = torch.load(f"{run}/{n}", map_location="cpu", mmap=True, weights_only=False)
        sd = ck["state_dict"]
        ema = [k for k in sd if k.startswith("ema_denoiser.")]
        shared = sum(1 for k in ema if k[4:] in sd and (sd[k].numel() == 0 or (sd[k].data_ptr() == sd[k[4:]].data_ptr()
                                                                                and sd[k].shape == sd[k[4:]].shape)))
        seen, logical, unique = set(), 0, 0
        for t in tensors(ck):
            b = t.numel() * t.element_size()
            logical += b
            if b and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                unique += b
        kind = "ms" if n.startswith("ms-step=") else "last"
        size_b = os.path.getsize(f"{run}/{n}")
        rs, ru = size_b / OLD[size][kind], unique / logical
        lo, hi = (0.48, 0.53) if kind == "ms" else (0.62, 0.71)   # the old layout gives 1.0
        step_ok = kind != "ms" or ck["global_step"] == int(n[8:14])
        full = kind == "ms" or all(k in ck for k in ("optimizer_states", "lr_schedulers", "loops", "callbacks"))
        check(ema and shared == len(ema) and lo < rs < hi and lo < ru < hi and step_ok and full,
              f"{tag} {n}: step {ck['global_step']}, {shared} of {len(ema)} EMA tensors share memory with the raw ones, "
              f"unique / logical bytes {ru:.4f}, {size_b} bytes = {rs:.4f} of the old layout's {OLD[size][kind]}"
              + ("" if full else ", TRAINING STATE MISSING"))
        if n in (want[3], "last.ckpt"):
            sds[n] = (ck["global_step"], sd)
    if len(sds) == 2:
        (s1, a), (s2, b) = sds[want[3]], sds["last.ckpt"]
        same = a.keys() == b.keys() and all(bits_equal(a[k], b[k]) for k in a)
        check(s1 == s2 == S and same, f"{tag}: {want[3]} and last.ckpt hold the same state_dict bit for bit ({len(a)} tensors, steps {s1}, {s2})")


def rows_of(d):
    return json.load(open(f"{d}/steps_rank0.json"))["rows"]


def resume(tag):
    bs = sorted(glob.glob(f"{out}/{tag}/bench_*"))
    if len(bs) != 2:
        return check(False, f"{tag}: {len(bs)} bench directories, 2 expected")
    a, b = ({r["step"]: r for r in rows_of(d)} for d in bs)
    cut = min(b)
    la = [a[s]["loss"] for s in range(max(cut - 40, min(a)), cut) if s in a]
    lb = [b[s]["loss"] for s in range(cut, cut + 40) if s in b]
    bad = sum(1 for r in list(a.values()) + list(b.values()) if not math.isfinite(r["loss"]))
    ma, mb = sum(la) / len(la), sum(lb) / len(lb)
    check(max(a) + 1 == cut and max(b) == RUNS[tag][1] - 1 and bad == 0 and mb < 1.1 * ma,
          f"{tag}: launch 1 steps {min(a)}-{max(a)}, launch 2 steps {cut}-{max(b)}, steps redone {max(a) + 1 - cut}; "
          f"mean loss 40 steps before the cut {ma:.4f}, 40 after the resume {mb:.4f}; non-finite {bad}")


def stalls(root, tag):
    """{save step: seconds above the launch's median step}, and the median step in ms"""
    res, meds = {}, []
    for d in sorted(glob.glob(f"{root}/{tag}/bench_*")):
        rows = [r for r in rows_of(d) if r.get("step_ms") is not None]
        if not rows:
            continue
        med = statistics.median(r["step_ms"] for r in rows)
        meds.append(med)
        for r in rows:
            if (r["step"] + 1) % RUNS[tag][2] == 0:
                res[r["step"] + 1] = (r["step_ms"] - med) / 1000
    return res, (statistics.mean(meds) if meds else float("nan"))


out = sys.argv[1].rstrip("/")
ref = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else None
for tag in RUNS:
    files(tag)
    resume(tag)
    new, med = stalls(out, tag)
    old, omed = stalls(ref, tag) if ref else ({}, float("nan"))
    both = sorted(set(new) & set(old))
    print(f"info  {tag}: median step {med:.1f} ms (old layout's smoke {omed:.1f} ms); stall of the step with a scheduled save, s, new (old): "
          + ", ".join(f"{s}: {new[s]:.2f} ({old[s]:.2f})" if s in old else f"{s}: {new[s]:.2f}" for s in sorted(new)))
    if both:
        sn, so = sum(new[s] for s in both), sum(old[s] for s in both)
        print(f"info  {tag}: sum over the {len(both)} save steps in both smokes: {sn:.1f} s new, {so:.1f} s old ({100 * (sn / so - 1):+.0f}%)")
print("CKPT_DEDUP_CHECK " + ("PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
