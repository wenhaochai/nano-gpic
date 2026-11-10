#!/usr/bin/env python3
"""Rewrite old-layout ms-step checkpoints to the twins layout of src/plugins/async_ckpt_io.py (gpic-28 2026-11-09).
Old layout: ema_denoiser.X is stored beside a denoiser.X with the same bytes (runs under GPIC_MFU_NO_EMA before the
ckpt-dedup branches). Twins layout: each such pair is stored once, so the file halves, and torch.load still gives both
keys. Owner 2026-11-09: approved for the old-layout files of the scan (twins_scan.py: "equal"), after smoke w2443.
Per file:
  1 the file is as the scan listed it: a regular file with one hard link, the listed size and mtime
  2 load it through mmap; each ema_denoiser.X whose samples equal denoiser.X points at that tensor (no other change:
    the same dict objects, key order and state-dict _metadata); torch.save to <file>.twinsrw_tmp; fsync
  3 load the old file again and the new file, and compare them in full: the same keys in the same order, every tensor
    the same dtype, shape and bits, every other value equal; each pointed pair shares its storage in the new file
  4 the old file is still the file of step 1 (inode, size, mtime); the new file gets its mode and mtime;
    os.replace (atomic: a reader holds the old file or the new one, never a part)
Any difference or error: the new file is removed and the old file stays as it is.
usage: rewrite_twins.py [--go] [--part i/n] <log.tsv> <scan.tsv>...
  without --go: step 1 only, and the totals. TWINS_ROOT replaces the run root (tests).
  log line: path, status, old bytes, new bytes, tensors, pairs, seconds. A path with status ok in the log is skipped."""
import os, stat, sys, time
import numpy as np
import torch

ROOT = os.environ.get("TWINS_ROOT", "/scratch/gpfs/GROUP/USER/runs/gpic-speed")
BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
SUF = ".twinsrw_tmp"
E = "ema_denoiser."


def load(f):
    return torch.load(f, map_location="cpu", mmap=True, weights_only=False)


def bits(t):
    return t.detach().contiguous().reshape(-1).view(BITS[t.element_size()])


def same_tensor(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and (a.numel() == 0 or bool(torch.equal(bits(a), bits(b))))


def sampled(a, b):
    """the scan's test: 3 slices of 16,384 elements. Step 3 compares in full, so a pair that passes here by chance fails there."""
    if a.dtype != b.dtype or a.shape != b.shape or a.numel() == 0 or a.element_size() not in BITS:
        return False
    x, y = bits(a), bits(b)
    n = x.numel()
    return all(bool(torch.equal(x[o:o + 16384], y[o:o + 16384])) for o in (0, n // 2, max(n - 16384, 0)))


def same(a, b, where="checkpoint"):
    """None when a and b are equal in full, else the place of the first difference"""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        ok = isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and same_tensor(a, b)
    elif type(a) is not type(b):
        ok = False
    elif isinstance(a, dict):
        if list(a.keys()) != list(b.keys()) or getattr(a, "_metadata", None) != getattr(b, "_metadata", None):
            return where + " (keys, their order or _metadata)"
        for k in a:
            bad = same(a[k], b[k], f"{where}[{k!r}]")
            if bad:
                return bad
        return None
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return where + " (length)"
        for i, (x, y) in enumerate(zip(a, b)):
            bad = same(x, y, f"{where}[{i}]")
            if bad:
                return bad
        return None
    elif isinstance(a, np.ndarray):
        ok = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    else:
        ok = bool(a == b)
    return None if ok else where


def listed(f, size, mtime):
    """step 1: (lstat, None) when the file is as the scan listed it, else (None, reason)"""
    try:
        st = os.lstat(f)
    except OSError as e:
        return None, f"skip-gone:{type(e).__name__}"
    if not stat.S_ISREG(st.st_mode):
        return None, "skip-not-regular"
    if st.st_nlink != 1:
        return None, f"skip-hard-links:{st.st_nlink}"
    if st.st_size != size:
        return None, f"skip-size-changed:{st.st_size}"
    if abs(st.st_mtime - mtime) > 0.01:
        return None, "skip-mtime-changed"
    return st, None


def rewrite(f, st):
    """steps 2-4: (status, new bytes, tensors, pairs)"""
    tmp = f + SUF
    try:
        ck = load(f)
        sd = ck["state_dict"]
        pairs = []
        for k in list(sd):
            r = sd.get("denoiser." + k[len(E):]) if isinstance(k, str) and k.startswith(E) else None
            if (isinstance(r, torch.Tensor) and isinstance(sd[k], torch.Tensor) and sd[k].data_ptr() != r.data_ptr()
                    and sampled(sd[k], r)):   # a pair that shares its storage is in the twins layout already
                sd[k] = r
                pairs.append(k)
        n = sum(isinstance(v, torch.Tensor) for v in sd.values())
        if not pairs:
            return "skip-no-pairs", 0, n, 0
        torch.save(ck, tmp)
        del ck, sd
        with open(tmp, "rb") as h:
            os.fsync(h.fileno())
        old, new = load(f), load(tmp)
        bad = same(old, new)
        if not bad:
            nsd = new["state_dict"]
            apart = [k for k in pairs if nsd[k].data_ptr() != nsd["denoiser." + k[len(E):]].data_ptr()]
            bad = f"{len(apart)} pairs not shared in the new file ({apart[:2]})" if apart else None
            del nsd
        del old, new
        size = os.path.getsize(tmp)
        if bad:
            os.remove(tmp)
            return "FAIL-compare:" + bad.replace("\t", " ")[:200], size, n, len(pairs)
        now = os.lstat(f)
        if (now.st_ino, now.st_size, now.st_mtime_ns, now.st_nlink) != (st.st_ino, st.st_size, st.st_mtime_ns, st.st_nlink):
            os.remove(tmp)
            return "skip-changed-during-rewrite", size, n, len(pairs)
        os.chmod(tmp, stat.S_IMODE(st.st_mode))
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, f)
        return "ok", size, n, len(pairs)
    except BaseException as e:   # also a TERM or ^C: the old file stays, the part-written new file goes
        try:
            os.remove(tmp)
        except OSError:
            pass
        if not isinstance(e, Exception):
            raise
        return f"FAIL-error:{type(e).__name__}:{str(e)[:160]}".replace("\t", " ").replace("\n", " "), 0, 0, 0


def main():
    args = sys.argv[1:]
    go = "--go" in args
    args = [a for a in args if a != "--go"]
    part, of = 0, 1
    if "--part" in args:
        i = args.index("--part")
        part, of = map(int, args[i + 1].split("/"))
        del args[i:i + 2]
    log, scans = args[0], args[1:]
    files = {}
    for s in scans:
        for line in open(s):
            p, size, mtime, res, _ = line.rstrip("\n").split("\t")
            if res == "equal" and not p.startswith("./raw_view/"):
                files[p] = (int(size), float(mtime))
    done = set()
    if os.path.exists(log):
        done = {l.split("\t")[0] for l in open(log) if l.split("\t")[1:2] == ["ok"]}
    todo = [p for i, p in enumerate(sorted(files)) if i % of == part and p not in done]
    print(f"{len(files)} old-layout files in the scan, {sum(v[0] for v in files.values()) / 2**40:.3f} TiB; "
          f"part {part}/{of}: {len(todo)} to do, {len(done)} already ok in {log}", flush=True)
    out = open(log, "a") if go else None
    count, t_all = {}, time.time()
    b_old = b_new = 0
    for j, p in enumerate(todo):
        f = os.path.normpath(os.path.join(ROOT, p))
        size = files[p][0]
        t0 = time.time()
        st, status = listed(f, *files[p])
        new = n = k = 0
        if go and st is not None and os.path.exists(f + SUF):
            os.remove(f + SUF)   # my own part-written file of a run that was killed
        if st is not None:
            status, new, n, k = rewrite(f, st) if go else ("would-rewrite", 0, 0, 0)
        key = status.split(":")[0]
        count[key] = count.get(key, 0) + 1
        if key in ("ok", "would-rewrite"):
            b_old += size
            b_new += new
        if out:
            out.write(f"{p}\t{status}\t{size}\t{new}\t{n}\t{k}\t{time.time() - t0:.1f}\n")
            out.flush()
        if key not in ("ok", "would-rewrite") or (j + 1) % 20 == 0:
            print(f"[{j + 1}/{len(todo)}] {p}: {status}; so far {b_old / 2**30:.1f} GiB -> {b_new / 2**30:.1f} GiB "
                  f"in {time.time() - t_all:.0f} s", flush=True)
    print(f"TWINS_REWRITE part {part}/{of} {'done' if go else 'dry'}: {count}; {b_old / 2**40:.4f} TiB"
          + (f" -> {b_new / 2**40:.4f} TiB, saved {(b_old - b_new) / 2**40:.4f} TiB" if go else " would be rewritten")
          + f"; {time.time() - t_all:.0f} s", flush=True)
    sys.exit(0 if all(c in ("ok", "would-rewrite") or c.startswith("skip") for c in count) else 1)


if __name__ == "__main__":
    main()
