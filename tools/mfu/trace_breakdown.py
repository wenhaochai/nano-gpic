#!/usr/bin/env python3
"""Where the GPU time of a training step goes, from a torch.profiler chrome trace (src/callbacks/bench.py BENCH_PROFILE).

For the profiled steps (ProfilerStep#N on the GPU timeline, or the whole window): the step's wall span on the GPU, the
time the compute stream is busy, and the kernel time per category (GEMM, attention, Inductor element-wise and
reduction kernels, optimizer and EMA, NCCL, copies), split by stage (stage:fwd / bwd / after_bwd / opt / between
annotations). NCCL runs on its own stream, so its time overlaps compute; "idle" is the step span minus the union of
non-NCCL kernel time.
usage: python trace_breakdown.py <rank<r>.json> [--top 40]
"""
import argparse, bisect, collections, json, re

CATS = [
    ("nccl", r"nccl"),
    ("attn_fwd", r"FlashAttnFwd|flash_fwd|fmha.*fwd|attention.*fwd|cutlass.*Fmha.*Fwd|fmha_cutlassF|_efficient_attention_forward|cudnn.*fprop|flex_attention(?!_backward)|triton_tem_fused_flex_attention(?!_backward)|sdpa.*fwd|fwd_kernel"),
    ("attn_bwd", r"FlashAttnBwd|flash_bwd|fmha.*bwd|attention.*bwd|cutlass.*Fmha.*Bwd|fmha_cutlassB|_efficient_attention_backward|cudnn.*dgrad|flex_attention_backward|sdpa.*bwd|bwd_kernel|convert_dq"),
    ("gemm", r"nvjet|gemm|sm90_xmma|cutlass|ampere_.*gemm|sm80_xmma|cublas|triton_tem_fused_(?!flex)|_mm_|splitK"),
    ("optim", r"multi_tensor_apply|fused_adam|foreach|_amp_|lerp"),
    ("triton_red", r"triton_red|triton_per"),
    ("triton_poi", r"triton_poi"),
    ("copy", r"Memcpy|Memset|memcpy|memset|copy_kernel|CatArrayBatched|direct_copy"),
    ("reduce_aten", r"reduce_kernel|norm_kernel|layer_norm|softmax"),
    ("elem_aten", r"elementwise_kernel|vectorized_elementwise|unrolled_elementwise|index|scatter|gather"),
]


def cat_of(name):
    for c, rx in CATS:
        if re.search(rx, name, re.I):
            return c
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
    if cs is not None:
        tot += ce - cs
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    ev = json.load(open(a.trace))["traceEvents"]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    ann = [e for e in ev if e.get("cat") == "gpu_user_annotation" and "dur" in e]
    steps = sorted([e for e in ann if e["name"].startswith("ProfilerStep#")], key=lambda e: e["ts"])
    stages = sorted([e for e in ann if e["name"].startswith("stage:")], key=lambda e: e["ts"])
    if steps:   # one annotation per stream that ran kernels in the step (the EMA's side stream too): merge them by name
        sw = {}
        for e in steps:
            s0, e0 = sw.get(e["name"], (e["ts"], e["ts"] + e["dur"]))
            sw[e["name"]] = (min(s0, e["ts"]), max(e0, e["ts"] + e["dur"]))
        win = sorted(sw.values())
    else:
        t0, t1 = min(e["ts"] for e in gpu), max(e["ts"] + e["dur"] for e in gpu)
        win = [(t0, t1)]
    n = len(win)
    span = sum(e - s for s, e in win)
    st_starts = [e["ts"] for e in stages]

    def stage_of(ts):
        i = bisect.bisect_right(st_starts, ts) - 1
        if i >= 0 and ts < stages[i]["ts"] + stages[i]["dur"]:
            return stages[i]["name"][6:]
        return "-"

    by_cat = collections.Counter()
    by_cat_stage = collections.defaultdict(collections.Counter)
    by_name = collections.defaultdict(lambda: [0, 0.0])
    busy_iv = []
    nker = 0
    for e in gpu:
        s, d = e["ts"], e["dur"]
        if not any(ws <= s < we for ws, we in win):
            continue
        c = cat_of(e["name"])
        by_cat[c] += d
        by_cat_stage[stage_of(s)][c] += d
        k = re.sub(r"\(.*", "", e["name"].replace("(anonymous namespace)::", ""))[:110]
        by_name[k][0] += 1
        by_name[k][1] += d
        if c != "nccl":
            busy_iv.append((s, s + d))
            nker += 1
    busy = union(busy_iv)
    print(f"{a.trace}: {n} steps, GPU span {span / n / 1e3:.2f} ms/step, compute busy {busy / n / 1e3:.2f} ms/step "
          f"({100 * busy / span:.1f}%), idle {(span - busy) / n / 1e3:.2f} ms/step, {nker / n:.0f} non-NCCL kernels/step")
    tot = sum(v for c, v in by_cat.items() if c != "nccl")
    print(f"{'category':12s} {'ms/step':>8s} {'% of busy-sum':>13s}")
    for c, v in by_cat.most_common():
        print(f"{c:12s} {v / n / 1e3:8.2f} {100 * v / tot if c != 'nccl' else float('nan'):13.1f}")
    print("\nby stage (ms/step):")
    cats = [c for c, _ in by_cat.most_common()]
    print(f"{'stage':10s} " + " ".join(f"{c[:10]:>10s}" for c in cats))
    for sname, cc in sorted(by_cat_stage.items(), key=lambda x: -sum(x[1].values())):
        print(f"{sname:10s} " + " ".join(f"{cc[c] / n / 1e3:10.2f}" for c in cats))
    print(f"\ntop {a.top} kernels: count/step, ms/step, us each, category")
    for k, (cnt, d) in sorted(by_name.items(), key=lambda x: -x[1][1])[:a.top]:
        print(f"{cnt / n:7.1f} {d / n / 1e3:7.2f} {d / cnt:8.1f}  {cat_of(k):10s} {k}")


if __name__ == "__main__":
    main()
