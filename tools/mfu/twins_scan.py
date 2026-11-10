#!/usr/bin/env python3
"""Which ms-step and raw-view files hold the raw weights twice (ema_denoiser.X bitwise equal to denoiser.X, old layout)?
Sampled: 3 tensor pairs per file through mmap, so a file costs a few MB of reads. Writes one line per file (resumable):
path, bytes, mtime, result (equal = old layout with EMA = raw, twins-layout, different = a real EMA, no-ema-keys), keys.
tools/mfu/rewrite_twins.py rewrites the "equal" files outside raw_view/ to the twins layout.
usage: twins_scan.py <list> <out.tsv>
  list: "<bytes> <mtime> <path>" lines, paths relative to runs/gpic-speed:
        find . -maxdepth 6 -name "*.ckpt" -type f -printf "%s %T@ %p\n"   (run in runs/gpic-speed)"""
import os, sys, torch
R = "/scratch/gpfs/GROUP/USER/runs/gpic-speed"
BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def eq(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    i = BITS[a.element_size()]
    x, y = a.contiguous().view(i).reshape(-1), b.contiguous().view(i).reshape(-1)
    n = x.numel()                       # 3 slices of 16,384 elements: an EMA that was updated differs everywhere
    return all(torch.equal(x[o:o + 16384], y[o:o + 16384]) for o in (0, n // 2, max(n - 16384, 0)))


done = set()
if os.path.exists(sys.argv[2]):
    done = {l.split("\t")[0] for l in open(sys.argv[2])}
out = open(sys.argv[2], "a")
n = 0
for line in open(sys.argv[1]):
    s, t, p = line.rstrip("\n").split(" ", 2)
    if "/ms-step=" not in p or p in done:
        continue
    f = os.path.normpath(os.path.join(R, p))
    keys = ""
    try:
        if not os.path.isfile(f) or os.path.islink(f):
            res = "gone-or-link"
        else:
            ck = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            sd = ck["state_dict"]
            ema = [k for k in sd if k.startswith("ema_denoiser.") and k[4:] in sd and sd[k].numel() >= 4096]
            keys = ",".join(sorted(k for k in ck if k != "state_dict"))[:120]
            if not ema:
                res = "no-ema-keys"
            else:
                pick = [ema[0], ema[len(ema) // 2], ema[-1]]
                shared = all(sd[k].data_ptr() == sd[k[4:]].data_ptr() for k in pick)
                res = "twins-layout" if shared else ("equal" if all(eq(sd[k], sd[k[4:]]) for k in pick) else "different")
            del ck, sd
    except Exception as e:
        res = "error:" + repr(e)[:100].replace("\t", " ").replace("\n", " ")
    out.write(f"{p}\t{s}\t{t}\t{res}\t{keys}\n")
    out.flush()
    n += 1
print("scanned", n)
