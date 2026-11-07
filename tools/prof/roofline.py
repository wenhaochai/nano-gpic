"""Per-kernel roofline of a torch.profiler trace: is every operator, in every stage of the step, efficient?

For each GPU kernel in the profiled steps: the op that launched it (External id), its shapes and dtypes, the stage it
was launched in (stage:fwd / bwd / after_bwd / opt / between markers from the bench callback), and an analytic
lower bound on its time: max(FLOPs / peak compute, minimum bytes / HBM bandwidth). Efficiency = bound / measured,
lost = measured - bound. FLOPs and bytes:
  GEMM (aten::mm/addmm/bmm)          2MNK(B); bytes = A + B + C at their dtypes
  SDPA forward / backward            4BHNSd / 10BHNSd; bytes = q,k,v,o (+do,dq,dk,dv)
  Triton (Inductor) kernels          bytes = sum of the launch's tensor args (in_out args counted twice)
  other aten kernels                 bytes = inputs + output (output ~ largest input): approximate, flagged
  NCCL                               bus bandwidth from the collective's message size, against NVLink
Peaks (H100 SXM): bf16/fp16 989.4 TFLOP/s dense, TF32 494.7, FP32 66.9, HBM 3.35 TB/s, NVLink 450 GB/s per direction.
usage: python roofline.py <trace rank<r>.json> <out_dir> [--steps N]
"""
import argparse, bisect, collections, csv, json, math, os, re

PEAK = {"bf16": 989.4e12, "fp16": 989.4e12, "tf32": 494.7e12, "fp32": 66.9e12}
HBM = 3.35e12
NVLINK = 450e9
ELEM = {"c10::BFloat16": 2, "c10::Half": 2, "float": 4, "double": 8, "long int": 8, "int": 4, "bool": 1,
        "unsigned char": 1, "signed char": 1, "c10::Float8_e4m3fn": 1, "short int": 2}
PREC = {"c10::BFloat16": "bf16", "c10::Half": "fp16", "float": "tf32"}  # fp32 matmuls run as TF32 (matmul precision medium)


def numel(d):
    n = 1
    for x in d:
        n *= x
    return n


def tensor_args(dims, types):
    out = []
    for d, t in zip(dims or [], types or []):
        if t == "TensorList" and isinstance(d, list):
            out.extend((x, "float") for x in d if isinstance(x, list) and all(isinstance(y, int) for y in x))
            continue
        if isinstance(d, list) and d and all(isinstance(x, int) for x in d) and t in ELEM:
            out.append((d, t))
        elif isinstance(d, list) and d == [] and t in ELEM:  # scalar tensor
            out.append(([1], t))
    return out


def op_cost(name, a, kname):
    """(flops, bytes, precision, method) for one kernel, from its launching op's args."""
    dims, types = a.get("Input Dims") or [], a.get("Input type") or []
    ta = tensor_args(dims, types)
    nm = name or ""
    try:
        if nm in ("aten::mm",) and len(dims) >= 2 and len(dims[0]) == 2:
            (m, k), (_, n) = dims[0], dims[1]
            e = ELEM.get(types[0], 2)
            return 2 * m * n * k, (m * k + k * n + m * n) * e, PREC.get(types[0], "bf16"), "gemm"
        if nm == "aten::addmm" and len(dims) >= 3 and len(dims[1]) == 2:
            (m, k), (_, n) = dims[1], dims[2]
            e = ELEM.get(types[1], 2)
            return 2 * m * n * k, (m * k + k * n + 2 * m * n) * e, PREC.get(types[1], "bf16"), "gemm"
        if nm == "aten::bmm" and len(dims) >= 2 and len(dims[0]) == 3:
            (b, m, k), (_, _, n) = dims[0], dims[1]
            e = ELEM.get(types[0], 2)
            return 2 * b * m * n * k, b * (m * k + k * n + m * n) * e, PREC.get(types[0], "bf16"), "gemm"
        if re.match(r"aten::_(scaled_dot_product|flash|efficient|cudnn).*attention", nm) and len(dims) >= 3 and len(dims[0]) == 4:
            bwd = "backward" in nm
            if bwd:  # (grad_out, q, k, v, out, ...)
                q, k = dims[1], dims[2]
            else:
                q, k = dims[0], dims[1]
            if re.match(r"aten::_(flash|efficient)_attention", nm):   # [B, N, H, d]
                b, n, h, d = q
                s, hk = k[1], k[2]
            else:                                                       # [B, H, N, d]
                b, h, n, d = q
                s, hk = k[2], k[1]
            e = ELEM.get(types[0], 2)
            qb, kb = b * h * n * d * e, b * hk * s * d * e
            if bwd:
                return 10 * b * h * n * s * d, 4 * qb + 4 * kb, PREC.get(types[0], "bf16"), "sdpa_bwd"
            return 4 * b * h * n * s * d, 2 * qb + 2 * kb, PREC.get(types[0], "bf16"), "sdpa_fwd"
        if kname.startswith("triton") or (a.get("kernel_backend") == "triton"):
            by = sum(numel(d) * ELEM[t] for d, t in ta)
            return 0, by, "bf16", "triton_args"
        if ta:
            ins = sum(numel(d) * ELEM[t] for d, t in ta)
            outb = max(numel(d) * ELEM[t] for d, t in ta)
            return 0, ins + outb, "bf16", "aten_approx"
    except Exception:
        pass
    return 0, 0, "bf16", "unknown"


def classify(kname, opname):
    s = (kname + " " + (opname or "")).lower()
    for cat, pat in (("nccl", r"nccl"), ("attention", r"flash|fmha|attention|sdpa"),
                     ("gemm", r"nvjet|gemm|cutlass|xmma|sm90_|ampere|cublas|aten::(mm|addmm|bmm)"),
                     ("optimizer", r"multi_tensor|foreach|adam|_amp_"), ("triton_reduction", r"triton_(red|per)"),
                     ("triton_pointwise", r"triton_poi"), ("copy/cast", r"copy|cast|to_copy|convert"),
                     ("memset/fill", r"fill|memset|zero"), ("reduce/norm", r"reduce|norm|sum|mean"),
                     ("elementwise", r"elementwise|vectorized|unrolled|cudafunctor|index|cat|where")):
        if re.search(pat, s):
            return cat
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("out"); ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ev = json.load(open(args.trace))["traceEvents"]
    ext_op, corr_launch, stages, steps, kernels, others = {}, {}, [], [], [], []
    for e in ev:
        c, a = e.get("cat"), e.get("args", {}) or {}
        if c == "cpu_op":
            ext_op[a.get("External id")] = e
        elif c == "cuda_runtime" or c == "cuda_driver":
            if "correlation" in a:
                corr_launch[a["correlation"]] = e
        elif c == "user_annotation":
            if e["name"].startswith("stage:"):
                stages.append((e["ts"], e["ts"] + e.get("dur", 0), e["name"][6:]))
            elif e["name"].startswith("ProfilerStep#"):
                steps.append((e["ts"], e["ts"] + e.get("dur", 0), e["name"]))
        elif c == "kernel":
            kernels.append(e)
        elif c in ("gpu_memcpy", "gpu_memset"):
            others.append(e)
    stages.sort(); starts = [s[0] for s in stages]
    nsteps = args.steps or max(1, len(steps))

    def stage_of(ts):
        i = bisect.bisect_right(starts, ts) - 1
        if i >= 0 and stages[i][0] <= ts <= stages[i][1]:
            return stages[i][2]
        return "outside"

    ext_dur = collections.defaultdict(float)
    for k in kernels:
        ext_dur[(k.get("args", {}) or {}).get("External id")] += k.get("dur", 0)
    rows = []
    for k in kernels + others:
        a = k.get("args", {}) or {}
        op = ext_op.get(a.get("External id"))
        share = (k.get("dur", 0) / ext_dur[a.get("External id")]) if (op and ext_dur[a.get("External id")]) else 1.0
        opname = op["name"] if op else ""
        oa = (op or {}).get("args", {}) or {}
        launch = corr_launch.get(a.get("correlation"))
        st = stage_of(launch["ts"]) if launch else "outside"
        dur_s = k.get("dur", 0) / 1e6
        if k.get("cat") == "gpu_memcpy":
            by = a.get("bytes", 0); fl, prec, meth = 0, "bf16", "memcpy"
        elif k.get("cat") == "gpu_memset":
            by = a.get("bytes", 0); fl, prec, meth = 0, "bf16", "memset"
        else:
            fl, by, prec, meth = op_cost(opname, oa, k["name"])
            fl, by = fl * share, by * share
        cat = "memcpy" if k.get("cat") == "gpu_memcpy" else classify(k["name"], opname)
        if cat == "nccl":
            bound = by / NVLINK if by else 0.0
        else:
            bound = max(fl / PEAK[prec] if fl else 0.0, by / HBM if by else 0.0)
        rows.append({"kernel": k["name"], "op": opname, "stage": st, "cat": cat, "method": meth,
                     "dims": json.dumps(oa.get("Input Dims"))[:200] if oa else "", "dur_s": dur_s, "flops": fl, "bytes": by,
                     "bound_s": bound, "stream": a.get("stream"), "grid": str(a.get("grid")), "block": str(a.get("block")),
                     "occ": a.get("est. achieved occupancy %"), "regs": a.get("registers per thread"),
                     "kernel_file": oa.get("kernel_file", "")})
    # aggregate per kernel signature
    agg = collections.OrderedDict()
    for r in rows:
        key = (r["kernel"][:120], r["op"], r["dims"], r["stage"])
        g = agg.setdefault(key, {"kernel": r["kernel"], "op": r["op"], "dims": r["dims"], "stage": r["stage"],
                                 "cat": r["cat"], "method": r["method"], "n": 0, "dur_s": 0.0, "flops": 0, "bytes": 0,
                                 "bound_s": 0.0, "occ": r["occ"], "grid": r["grid"], "block": r["block"],
                                 "kernel_file": r["kernel_file"]})
        g["n"] += 1; g["dur_s"] += r["dur_s"]; g["flops"] += r["flops"]; g["bytes"] += r["bytes"]; g["bound_s"] += r["bound_s"]
    out = []
    for g in agg.values():
        g["per_step_ms"] = 1e3 * g["dur_s"] / nsteps
        g["calls_per_step"] = g["n"] / nsteps
        g["tflops"] = g["flops"] / g["dur_s"] / 1e12 if g["dur_s"] else 0
        g["gbps"] = g["bytes"] / g["dur_s"] / 1e9 if g["dur_s"] else 0
        g["eff"] = g["bound_s"] / g["dur_s"] if g["dur_s"] else 0
        g["lost_ms_per_step"] = 1e3 * (g["dur_s"] - g["bound_s"]) / nsteps
        out.append(g)
    out.sort(key=lambda g: -g["lost_ms_per_step"])
    with open(os.path.join(args.out, "kernels.csv"), "w", newline="") as f:
        cols = ["lost_ms_per_step", "per_step_ms", "eff", "tflops", "gbps", "calls_per_step", "stage", "cat", "method",
                "op", "kernel", "dims", "occ", "grid", "block", "kernel_file"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(out)
    # summaries
    tot = sum(r["dur_s"] for r in rows) / nsteps
    bnd = sum(r["bound_s"] for r in rows) / nsteps
    by_cs = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        x = by_cs[(r["stage"], r["cat"])]; x[0] += r["dur_s"]; x[1] += r["bound_s"]; x[2] += 1
    step_wall = sum(e - s for s, e, _ in steps) / max(1, len(steps)) / 1e6 if steps else float("nan")
    lines = [f"# Roofline of {os.path.basename(args.trace)}: {nsteps} steps",
             "", f"step wall (ProfilerStep, CPU span) {1e3 * step_wall:.1f} ms; GPU kernel time per step {1e3 * tot:.1f} ms; "
             f"sum of per-kernel lower bounds {1e3 * bnd:.1f} ms (kernel-level speed of light, same kernels, same math)",
             "", "## Per stage and category (ms per step)", "",
             "| stage | category | kernels/step | measured | bound | efficiency | lost |", "|---|---|---|---|---|---|---|"]
    for (st, cat), (d, b, n) in sorted(by_cs.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
        lines.append(f"| {st} | {cat} | {n / nsteps:.0f} | {1e3 * d / nsteps:.2f} | {1e3 * b / nsteps:.2f} | "
                     f"{(b / d if d else 0):.0%} | {1e3 * (d - b) / nsteps:.2f} |")
    lines += ["", "## Top 60 kernel signatures by lost time (ms per step)", "",
              "| lost | measured | eff | TFLOP/s | GB/s | calls | stage | category | op | kernel | dims |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in out[:60]:
        lines.append(f"| {g['lost_ms_per_step']:.2f} | {g['per_step_ms']:.2f} | {g['eff']:.0%} | {g['tflops']:.0f} | "
                     f"{g['gbps']:.0f} | {g['calls_per_step']:.0f} | {g['stage']} | {g['cat']} | {g['op'][:40]} | "
                     f"`{g['kernel'][:60]}` | `{g['dims'][:70]}` |")
    open(os.path.join(args.out, "roofline.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
