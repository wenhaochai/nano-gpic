#!/usr/bin/env python3
"""Do two predicts give the same samples, and when not, is the difference the predict's own run-to-run difference?
(2026-11-09, smoke w2430: at width 1536 the predict from a twins-layout checkpoint and the predict from an old-layout
copy of the same 500 tensors gave different samples. That predict was already known to differ run to run, FD spread
0.005% over five runs, while the width-640 and width-768 predicts repeat bit for bit; logs/mfu_eval_idle.yaml.)
Manifest lines: <group> <name> <samples .npy>. A main group holds link (the twins file), old and old2 (the old-layout
copy, predicted twice; the smoke predicts old, link, old2 in this order, so a first predict that differs from all later
ones does not pass for an effect of the layout). A group <main>:<setting> holds link and old under one changed setting;
it is reported only.
Rule for a main group:
  link = old bit for bit: pass
  else old = old2 bit for bit: fail, the predict repeats and the layout changes its samples
  else link = old2 bit for bit: pass (the two old-layout predicts differ, and the twins file gives one of them)
  else: link against old and against old2 must look like old against old2: in the images that differ, the share of
        different values and the mean difference at most 2 times those of old against old2
Sample n is of batch n // 256 and rank (n // 32) % 8 (8 ranks, 32 samples each a batch: lite_ladder_sample.yaml).
usage: check_pred_repro.py <manifest>; one line per pair, then PRED_REPRO_CHECK PASS | FAIL <groups>"""
import collections, hashlib, sys
import numpy as np

groups = collections.OrderedDict()
for line in open(sys.argv[1]):
    g, name, path = line.split()
    groups.setdefault(g, {})[name] = path
arr, md5 = {}, {}
for g, members in groups.items():
    for name, path in members.items():
        a = np.load(path)
        a = a if isinstance(a, np.ndarray) else a["arr_0"]   # output.npz of the evaluations before run6
        arr[g, name], md5[g, name] = a, hashlib.md5(a.tobytes()).hexdigest()
        print(f"{g} {name}: {a.shape} {a.dtype}, md5 {md5[g, name]}")


def pair(g, x, y):
    """None when equal bit for bit, else the statistics of the difference; prints one line"""
    a, b = arr[g, x], arr[g, y]
    if a.shape == b.shape and md5[g, x] == md5[g, y]:
        print(f"{g} {x} against {y}: equal bit for bit")
        return None
    if a.shape != b.shape:
        print(f"{g} {x} against {y}: shapes differ, {a.shape} and {b.shape}")
        return {"share": float("inf"), "mean": float("inf")}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).reshape(len(a), -1)
    idx = np.nonzero(d.max(1) > 0)[0]
    dd = d[idx]
    s = {"images": len(idx), "share": float((dd > 0).mean()), "mean": float(dd.mean()), "largest": int(dd.max())}
    rank = np.bincount((idx // 32) % 8, minlength=8).tolist()
    batch = np.bincount(idx // 256, minlength=-(-len(a) // 256)).tolist()
    print(f"{g} {x} against {y}: {s['images']} of {len(a)} images differ; in them, share of different values {s['share']:.4f}, "
          f"mean difference {s['mean']:.4f}, largest {s['largest']}; images by rank {rank}; by batch {batch}")
    return s


failed = []
for g, members in groups.items():
    if ":" in g:
        if not {"link", "old"} <= members.keys():
            print(f"{g}: not compared, a predict of this setting failed")
        elif pair(g, "link", "old") is None:
            print(f"{g}: with this setting the samples of the two layouts are equal bit for bit")
        continue
    if not {"link", "old", "old2"} <= members.keys():
        print(f"FAIL {g}: predicts are missing, have {sorted(members)}")
        failed.append(g)
        continue
    lo = pair(g, "link", "old")
    oo = pair(g, "old", "old2")
    if lo is None:
        print(f"ok   {g}: the twins file and the old-layout copy give the same samples bit for bit"
              + ("" if oo is None else "; the second predict of the old-layout copy differs, so that equality is one draw of a predict that does not repeat"))
    elif oo is None:
        print(f"FAIL {g}: the predict repeats bit for bit, and the layout changes its samples")
        failed.append(g)
    elif (lo2 := pair(g, "link", "old2")) is None:
        print(f"ok   {g}: the predict does not repeat (old against old2), and the twins file gives the samples of the second old-layout predict bit for bit")
    else:
        ok = all(s[k] <= 2 * oo[k] for s in (lo, lo2) for k in ("share", "mean"))
        ratio = max(s[k] / oo[k] for s in (lo, lo2) for k in ("share", "mean"))
        print(f"{'ok  ' if ok else 'FAIL'} {g}: the predict does not repeat (old against old2); the twins file against either differs "
              f"at most {ratio:.2f} times as much (limit 2)")
        if not ok:
            failed.append(g)
print("PRED_REPRO_CHECK " + ("PASS" if not failed else "FAIL " + " ".join(failed)))
sys.exit(1 if failed else 0)
