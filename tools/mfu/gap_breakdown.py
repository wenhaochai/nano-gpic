#!/usr/bin/env python3
"""Per-step breakdown of an arm's profiled window (jobs/mfu/gap_*.sh: BENCH_PROFILE, light): GPU time per kernel class,
GPU idle time, and the waits, per rank, averaged over the profiled steps after the first.

Kernel classes (by name; Inductor's fused_cat kernels count as other: they compute): gemm (cuBLAS nvjet / cutlass GEMMs), attention (FlashAttention-3), qk_prep (lever K2),
norm (lever K), copy_index (concatenations, index copies and fills: the unfused packed path's key buffer), optimizer
(multi_tensor_apply: AdamW / EMA), nccl, memcpy, other (the rest of the elementwise and reduction kernels).
Waits: idle = the GPU window minus the union of its kernels; straggler = this rank's NCCL time minus the smallest NCCL
time over the ranks (the last rank to arrive waits least); data = the step timer's DataLoader wait (rows of the second
half); bucket = the shared bucket's DataLoader and all-reduce waits per micro-batch from train.log ([shared_bucket]).
usage: gap_breakdown.py <arm dir> [...]   (prints one block per arm and writes <arm dir>/bench/breakdown.json)"""
import glob, json, os, re, statistics as st, sys

CLASSES = ("gemm", "attention", "qk_prep", "norm", "copy_index", "optimizer", "nccl", "memcpy", "other")


def klass(n):
    if "nccl" in n:
        return "nccl"
    if "flash" in n.lower():
        return "attention"
    if n.startswith(("_fwd_image", "_fwd_text", "_bwd_image", "_bwd_text")):
        return "qk_prep"
    if n.startswith(("_fwd_kernel", "_bwd_kernel")):
        return "norm"
    if n.startswith("nvjet") or "gemm" in n.lower() or "cutlass::Kernel" in n or "xmma" in n:
        return "gemm"
    if "multi_tensor_apply" in n:
        return "optimizer"
    if n.startswith("Memcpy") or n.startswith("Memset"):
        return "memcpy"
    if re.search(r"index|scatter|Fill", n):   # Inductor's "fused_cat" kernels compute (e.g. the SwiGLU backward writes both
        return "copy_index"                   # halves of its input gradient into one buffer): they stay in "other"
    return "other"


def union(iv):
    iv.sort()
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if cs is not None else 0.0)


def rank_window(path):
    d = json.load(open(path))
    ev = d["traceEvents"] if isinstance(d, dict) else d
    steps = sorted((e for e in ev if str(e.get("name", "")).startswith("ProfilerStep#")), key=lambda e: e["ts"])
    t0 = steps[1]["ts"] if len(steps) > 1 else steps[0]["ts"]
    ks = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e["ts"] >= t0]
    n = max(len(steps) - 1, 1)
    per = {c: 0.0 for c in CLASSES}
    for e in ks:
        per[klass(e["name"])] += e.get("dur", 0)
    wall = max(e["ts"] + e.get("dur", 0) for e in ks) - min(e["ts"] for e in ks)
    busy = union([(e["ts"], e["ts"] + e.get("dur", 0)) for e in ks])
    out = {c: per[c] / n / 1000 for c in CLASSES}
    out.update(window=wall / n / 1000, busy=busy / n / 1000, idle=(wall - busy) / n / 1000)
    return out


def arm(d):
    tr = sorted(glob.glob(os.path.join(d, "bench", "trace", "rank*.json")))
    R = [rank_window(p) for p in tr]
    nmin = min(r["nccl"] for r in R)
    for r in R:
        r["straggler"] = r["nccl"] - nmin
    mean = {k: st.mean(r[k] for r in R) for k in R[0]}
    rows = [json.load(open(f))["rows"] for f in sorted(glob.glob(os.path.join(d, "bench", "steps_rank*.json")))]
    n = min(len(x) for x in rows)
    mean["data_wait"] = st.mean(st.mean(1000 * x["data_wait_s"] for x in rr[n // 2:n]) for rr in rows)
    mean["step_ms"] = st.mean(x.get("step_ms", 1000 * (x["t_end"] - x["t_start"])) for x in rows[0][n // 2:n])
    log = open(os.path.join(d, "train.log"), errors="replace").read()
    sb = re.findall(r"\[shared_bucket\].*DataLoader ([\d.]+) ms, all-reduce ([\d.]+) ms", log)
    if sb:
        mean["bucket_pull"] = st.mean(float(a) for a, _ in sb)
        mean["bucket_allreduce"] = st.mean(float(b) for _, b in sb)
    mean["ranks"] = len(R)
    json.dump(mean, open(os.path.join(d, "bench", "breakdown.json"), "w"), indent=1)
    return mean


if __name__ == "__main__":
    for d in sys.argv[1:]:
        m = arm(d)
        gpu = " ".join(f"{c} {m[c]:.1f}" for c in CLASSES)
        print(f"{os.path.basename(d)}: step {m['step_ms']:.1f} ms (second half); profiled window per step {m['window']:.1f}: "
              f"GPU busy {m['busy']:.1f} [{gpu}], idle {m['idle']:.1f}, straggler wait in NCCL {m['straggler']:.1f}; "
              f"data wait {m['data_wait']:.1f}" + (f"; bucket waits per micro-batch: pull {m['bucket_pull']:.1f}, "
                                                   f"all-reduce {m['bucket_allreduce']:.1f}" if "bucket_pull" in m else ""))
